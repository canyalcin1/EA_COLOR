import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import time

# --- 1. AYARLAR ---
class SystemConfig:
    MODEL_PATH = "3DeltaEModel.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # KADEMELER (Huni Sistemi)
    STEP_1_COUNT = 20  # Önce 20'ye düş
    STEP_2_COUNT = 7   # Sonra 7'ye düş

# --- 2. MODEL SINIFI ---
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

# --- 3. YÜKLEME ---
def load_trained_model():
    print(f"⚙️ Model Yükleniyor... ({SystemConfig.DEVICE})")
    try:
        checkpoint = torch.load(SystemConfig.MODEL_PATH, map_location=SystemConfig.DEVICE, weights_only=False)
    except FileNotFoundError:
        print("❌ HATA: Model dosyası yok.")
        exit()

    saved_config = checkpoint.get('config', {})
    input_cols = checkpoint['input_cols']
    target_cols = checkpoint['target_cols']
    scaler = checkpoint['scaler']
    
    model = PigmentColorNet(
        num_pigments=len(input_cols), 
        out_dim=len(target_cols),
        embed_dim=saved_config.get('embed_dim', 24),
        hidden_dim=saved_config.get('hidden_dim', 128),
        dropout=saved_config.get('dropout', 0.0)
    ).to(SystemConfig.DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, scaler, input_cols, target_cols

MODEL, SCALER, INPUT_COLS, TARGET_COLS = load_trained_model()

# --- 4. TAHMİN ---
def predict_delta_e(recipe_norm, target_lab_values):
    with torch.no_grad():
        input_tensor = torch.tensor(recipe_norm, dtype=torch.float32).unsqueeze(0).to(SystemConfig.DEVICE)
        pred_scaled = MODEL(input_tensor).cpu().numpy()
    pred_real = SCALER.inverse_transform(pred_scaled)
    diff = target_lab_values.reshape(1, 5, 3) - pred_real.reshape(1, 5, 3)
    return np.mean(np.sqrt(np.sum(diff**2, axis=2)))

# --- 5. OPTİMİZASYON FONKSİYONLARI ---

# Fonksiyon A: Tamamı ile (Soup)
def objective_full(recipe_candidate, target_lab_values):
    recipe_candidate = np.maximum(recipe_candidate, 0)
    if np.sum(recipe_candidate) == 0: return 9999.0
    recipe_norm = recipe_candidate / np.sum(recipe_candidate)
    
    error = predict_delta_e(recipe_norm, target_lab_values)
    
    # Hafif ceza (ki hepsi 0.0001 olmasın, bazıları öne çıksın)
    return error + (np.sum(recipe_norm > 0.001) * 0.02)

# Fonksiyon B: Sadece Seçilen İndekslerle
def objective_subset(reduced_ratios, selected_indices, full_size, target_lab_values):
    full_recipe = np.zeros(full_size, dtype=np.float32)
    reduced_ratios = np.maximum(reduced_ratios, 0)
    
    if np.sum(reduced_ratios) == 0: return 9999.0
    
    full_recipe[selected_indices] = reduced_ratios
    full_recipe /= np.sum(full_recipe) # Normalize
    
    return predict_delta_e(full_recipe, target_lab_values)

# --- 6. HUNİ ALGORİTMASI ---
def find_smart_recipe(target_lab):
    print(f"\n🧪 Reçete Aranıyor (Huni Tekniği)... Hedef: {target_lab[0:3]}...")
    start_time = time.time()
    num_pigments = len(INPUT_COLS)
    
    # --- AŞAMA 1: GENİŞ TARAMA (SOUP) ---
    print(f"🌊 Aşama 1: Geniş Havuz Taranıyor...")
    res_1 = differential_evolution(
        func=objective_full,
        bounds=[(0, 1)] * num_pigments,
        args=(target_lab,),
        strategy='best1bin',
        maxiter=120,    # Hızlı olsun
        popsize=6,
        tol=0.1,
        workers=1,
        disp=False
    )
    
    # İlk elemenin yapılması (Top 20)
    recipe_1 = res_1.x
    top_20_indices = np.argsort(recipe_1)[-SystemConfig.STEP_1_COUNT:]
    print(f"   -> Delta E: {res_1.fun:.2f} | En iyi {SystemConfig.STEP_1_COUNT} aday seçildi.")
    
    # --- AŞAMA 2: YARI FİNAL (20 PIGMENT) ---
    print(f"🌪️ Aşama 2: Daraltma ({SystemConfig.STEP_1_COUNT} Pigment)...")
    res_2 = differential_evolution(
        func=objective_subset,
        bounds=[(0, 1)] * len(top_20_indices),
        args=(top_20_indices, num_pigments, target_lab),
        strategy='best1bin',
        maxiter=200,
        popsize=8,
        tol=0.05,
        workers=1,
        disp=False
    )
    
    # İkinci eleme (Top 7) - İşte burada 980 öne çıkmış olmalı
    # Burada oranları tekrar kontrol ediyoruz, çünkü 20 pigment içindeki oranlar değişti
    recipe_2_vals = res_2.x
    # Kendi içindeki sıralamayı bul
    sorted_local_indices = np.argsort(recipe_2_vals)[-SystemConfig.STEP_2_COUNT:]
    # Global indekslere çevir
    top_7_indices = top_20_indices[sorted_local_indices]
    
    print(f"   -> Delta E: {res_2.fun:.2f} | Finalist {SystemConfig.STEP_2_COUNT} pigment belirlendi.")
    print(f"   -> Finalistler: {[INPUT_COLS[i] for i in top_7_indices]}")

    # --- AŞAMA 3: FİNAL (7 PIGMENT) ---
    print(f"✨ Aşama 3: İnce İşçilik (Final)...")
    res_final = differential_evolution(
        func=objective_subset,
        bounds=[(0, 1)] * len(top_7_indices),
        args=(top_7_indices, num_pigments, target_lab),
        strategy='best1bin',
        maxiter=400,   # İyice otursun
        popsize=12,
        tol=0.01,
        workers=1,
        disp=True
    )

    # Sonucu hazırla
    final_recipe = np.zeros(num_pigments)
    final_recipe[top_7_indices] = res_final.x
    final_recipe = np.maximum(final_recipe, 0)
    final_recipe /= np.sum(final_recipe)
    
    duration = time.time() - start_time
    final_de = predict_delta_e(final_recipe, target_lab)
    
    print(f"\n✅ İŞLEM TAMAMLANDI ({duration:.1f} sn)")
    print(f"🏆 Final Delta E: {final_de:.4f}")
    
    return final_recipe

if __name__ == "__main__":
    df_test = pd.read_csv("eval_dataset_clean.csv", sep=None, engine='python')
    for c in TARGET_COLS + INPUT_COLS:
        if df_test[c].dtype == object:
            df_test[c] = pd.to_numeric(df_test[c].str.replace(',', '.'), errors='coerce')
    df_test = df_test.fillna(0.0)

    # Aynı örnek (20)
    sample_idx = 30
    print(f"📌 Örnek No: {sample_idx}")
    
    real_target = df_test[TARGET_COLS].iloc[sample_idx].values.astype(np.float32)
    real_recipe = df_test[INPUT_COLS].iloc[sample_idx].values.astype(np.float32)
    if real_recipe.sum() > 0: real_recipe /= real_recipe.sum()

    found_recipe = find_smart_recipe(real_target)
    
    print(f"\n📊 HUNİ SİSTEMİ SONUCU (Max {SystemConfig.STEP_2_COUNT} Pigment):")
    print(f"{'Pigment':<10} | {'Gerçek':<10} | {'Yapay Zeka':<10} | {'Fark'}")
    print("-" * 50)
    
    for i, col in enumerate(INPUT_COLS):
        r_val = real_recipe[i]
        f_val = found_recipe[i]
        if r_val > 0.005 or f_val > 0.005:
            print(f"{col:<10} | {r_val:.4f}     | {f_val:.4f}     | {abs(r_val-f_val):.4f}")