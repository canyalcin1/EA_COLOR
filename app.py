import gradio as gr
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import time

# ==========================================
# 1. ARKA PLAN: MODEL VE ALGORİTMA MOTORU
# ==========================================

class SystemConfig:
    MODEL_PATH = "3DeltaEModel.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    STEP_1_COUNT = 20 # Havuz genişliği sabit kalsın (20 iyidir)

# Model Sınıfı
class PigmentColorNet(nn.Module):
    def __init__(self, num_pigments, out_dim, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.pigment_embedding = nn.Linear(num_pigments, embed_dim, bias=False)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        embedding_mix = self.pigment_embedding(x) 
        output = self.regressor(embedding_mix)
        return output

# Modeli Yükle
print(f"⚙️ Model Yükleniyor... ({SystemConfig.DEVICE})")
try:
    checkpoint = torch.load(SystemConfig.MODEL_PATH, map_location=SystemConfig.DEVICE, weights_only=False)
    saved_config = checkpoint.get('config', {})
    INPUT_COLS = checkpoint['input_cols']
    TARGET_COLS = checkpoint['target_cols']
    SCALER = checkpoint['scaler']
    
    MODEL = PigmentColorNet(
        num_pigments=len(INPUT_COLS), 
        out_dim=len(TARGET_COLS),
        embed_dim=saved_config.get('embed_dim', 24),
        hidden_dim=saved_config.get('hidden_dim', 128),
        dropout=saved_config.get('dropout', 0.0)
    ).to(SystemConfig.DEVICE)
    
    MODEL.load_state_dict(checkpoint['model_state'])
    MODEL.eval()
    print("✅ Model Hazır!")
except FileNotFoundError:
    print("❌ HATA: 'pigment_model_bundle.pt' bulunamadı.")
    exit()

# --- Yardımcı Fonksiyonlar ---
def predict_delta_e(recipe_norm, target_lab_values):
    with torch.no_grad():
        input_tensor = torch.tensor(recipe_norm, dtype=torch.float32).unsqueeze(0).to(SystemConfig.DEVICE)
        pred_scaled = MODEL(input_tensor).cpu().numpy()
    pred_real = SCALER.inverse_transform(pred_scaled)
    diff = target_lab_values.reshape(1, 5, 3) - pred_real.reshape(1, 5, 3)
    return np.mean(np.sqrt(np.sum(diff**2, axis=2)))

def objective_wrapper(recipe_candidate, target_lab_values):
    recipe_candidate = np.maximum(recipe_candidate, 0)
    if np.sum(recipe_candidate) == 0: return 9999.0
    recipe_norm = recipe_candidate / np.sum(recipe_candidate)
    return predict_delta_e(recipe_norm, target_lab_values)

def objective_subset(reduced_ratios, selected_indices, full_size, target_lab_values):
    full_recipe = np.zeros(full_size, dtype=np.float32)
    reduced_ratios = np.maximum(reduced_ratios, 0)
    if np.sum(reduced_ratios) == 0: return 9999.0
    full_recipe[selected_indices] = reduced_ratios
    full_recipe /= np.sum(full_recipe)
    return predict_delta_e(full_recipe, target_lab_values)

# --- ANA MOTOR (HUNİ SİSTEMİ - DİNAMİK) ---
def run_ai_engine(target_lab, allowed_pigments, max_pigment_count):
    start_time = time.time()
    num_pigments = len(INPUT_COLS)
    
    # Kısıtlamalar
    active_bounds = [(0, 1)] * num_pigments
    if len(allowed_pigments) < num_pigments:
        for i, col in enumerate(INPUT_COLS):
            if col not in allowed_pigments:
                active_bounds[i] = (0, 0) # Yasakla

    yield "🌊 Aşama 1: Geniş Havuz Taranıyor...", pd.DataFrame(), ""
    
    # AŞAMA 1: Geniş Tarama
    res_1 = differential_evolution(
        func=objective_wrapper,
        bounds=active_bounds,
        args=(target_lab,),
        strategy='best1bin',
        maxiter=30,
        popsize=4,
        workers=1
    )
    
    recipe_1 = res_1.x
    # İlk 20 taneyi seç (veya kısıtlı sayıdan fazla değilse hepsini)
    top_20_indices = np.argsort(recipe_1)[-SystemConfig.STEP_1_COUNT:]
    
    yield f"🌪️ Aşama 2: Huni Daralıyor (En iyi {max_pigment_count} seçilecek)...", pd.DataFrame(), ""

    # AŞAMA 2: Yarı Final
    res_2 = differential_evolution(
        func=objective_subset,
        bounds=[(0, 1)] * len(top_20_indices),
        args=(top_20_indices, num_pigments, target_lab),
        strategy='best1bin',
        maxiter=50,
        popsize=6,
        workers=1
    )
    
    recipe_2_vals = res_2.x
    # KULLANICININ SEÇTİĞİ SAYI KADAR AL (Dynamic Slider)
    # Slider 7 geldiyse 7 tane, 4 geldiyse 4 tane.
    sorted_local = np.argsort(recipe_2_vals)[-int(max_pigment_count):]
    top_final_indices = top_20_indices[sorted_local]

    yield f"✨ Aşama 3: İnce İşçilik (Final {int(max_pigment_count)})...", pd.DataFrame(), ""

    # AŞAMA 3: Final
    res_final = differential_evolution(
        func=objective_subset,
        bounds=[(0, 1)] * len(top_final_indices),
        args=(top_final_indices, num_pigments, target_lab),
        strategy='best1bin',
        maxiter=150,
        popsize=10,
        tol=0.01,
        workers=1
    )

    # Sonucu Hazırla
    final_recipe = np.zeros(num_pigments)
    final_recipe[top_final_indices] = res_final.x
    final_recipe = np.maximum(final_recipe, 0)
    final_recipe /= np.sum(final_recipe)
    
    final_de = predict_delta_e(final_recipe, target_lab)
    duration = time.time() - start_time
    
    # DataFrame
    result_data = []
    for i, val in enumerate(final_recipe):
        if val > 0.001: 
            result_data.append([INPUT_COLS[i], round(val * 100, 2)])
    
    df_result = pd.DataFrame(result_data, columns=["Pigment", "Oran (%)"])
    df_result = df_result.sort_values(by="Oran (%)", ascending=False)
    
    status_msg = f"✅ Bitti! ({duration:.2f} sn)"
    delta_msg = f"🏆 Tahmini Delta E: {final_de:.4f}"
    
    yield status_msg, df_result, delta_msg

# ==========================================
# 2. ARAYÜZ (FRONTEND)
# ==========================================

def interface_fn(
    l15, a15, b15, 
    l25, a25, b25, 
    l45, a45, b45, 
    l75, a75, b75, 
    l110, a110, b110,
    selected_pigments,
    max_pigment_count
):
    try:
        target_lab = np.array([
            l15, a15, b15, 
            l25, a25, b25, 
            l45, a45, b45, 
            l75, a75, b75, 
            l110, a110, b110
        ], dtype=np.float32)
    except:
        return "❌ Hata: Sayısal değerler girin.", pd.DataFrame(), "---"

    for status, df, de_msg in run_ai_engine(target_lab, selected_pigments, max_pigment_count):
        yield status, df, de_msg

# Theme parametresini kaldırdım (Hata çözümü için)
with gr.Blocks(title="RenkAI Pro") as demo:
    gr.Markdown("# 🎨 RenkAI: Akıllı Reçete Motoru")
    
    with gr.Row():
        # SOL KOLON
        with gr.Column(scale=1):
            gr.Markdown("### 1. Hedef Renk (L, a, b)")
            
            with gr.Row():
                l15 = gr.Number(label="15° L", value=65.22)
                a15 = gr.Number(label="a", value=-6.39)
                b15 = gr.Number(label="b", value=-5.53)
            with gr.Row():
                l25 = gr.Number(label="25° L", value=49.45)
                a25 = gr.Number(label="a", value=-5.42)
                b25 = gr.Number(label="b", value=-4.85)
            with gr.Row():
                l45 = gr.Number(label="45° L", value=25.29)
                a45 = gr.Number(label="a", value=-3.19)
                b45 = gr.Number(label="b", value=-3.52)
            with gr.Row():
                l75 = gr.Number(label="75° L", value=10.47)
                a75 = gr.Number(label="a", value=-0.51)
                b75 = gr.Number(label="b", value=-2.39)
            with gr.Row():
                l110 = gr.Number(label="110° L", value=5.70)
                a110 = gr.Number(label="a", value=0.62)
                b110 = gr.Number(label="b", value=-1.25)
            
            gr.Markdown("### 2. Ayarlar")
            
            # YENİ ÖZELLİK: Slider
            max_pigment_count = gr.Slider(
                minimum=3, maximum=12, value=7, step=1, 
                label="Maksimum Pigment Sayısı",
                info="Final reçetede en fazla kaç çeşit pigment olsun?"
            )
            
            pigment_selector = gr.CheckboxGroup(
                choices=INPUT_COLS, 
                value=INPUT_COLS, 
                label="Kullanılabilir Stok",
                info="Elinizde olmayan pigmentleri çıkarın."
            )
            
            calc_btn = gr.Button("🧪 Reçeteyi Hesapla", variant="primary", size="lg")

        # SAĞ KOLON
        with gr.Column(scale=1):
            gr.Markdown("### 🧬 Analiz Sonucu")
            
            status_lbl = gr.Label(value="Hazır", label="Durum")
            delta_lbl = gr.Label(value="---", label="Tahmini Delta E")
            
            result_df = gr.Dataframe(
                headers=["Pigment", "Oran (%)"],
                datatype=["str", "number"],
                label="Önerilen Reçete"
            )

    calc_btn.click(
        fn=interface_fn,
        inputs=[
            l15, a15, b15, l25, a25, b25, l45, a45, b45, l75, a75, b75, l110, a110, b110,
            pigment_selector, max_pigment_count # Yeni input
        ],
        outputs=[status_lbl, result_df, delta_lbl]
    )

if __name__ == "__main__":
    demo.queue().launch()