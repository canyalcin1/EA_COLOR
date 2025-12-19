import gradio as gr
import pandas as pd
import numpy as np
import torch
import glob
from ga_engine import GeneticOptimizer

# --- 1. META VERİ VE MODEL HAZIRLIĞI ---
def load_meta_data():
    # Modelleri bul
    model_files = glob.glob("models/model_fold_*.pt")
    if not model_files: 
        model_files = glob.glob("models_production/model_fold_*.pt")
    
    if not model_files: 
        print("⚠️ Uyarı: Model dosyası bulunamadı!")
        return [], []
    
    # İlk modelden pigment listesini çek ve SIRALA
    chk = torch.load(model_files[0], map_location='cpu',weights_only=False)
    pigments = sorted(chk['input_cols']) # Alfabetik/Nümerik sıralama
    targets = chk['target_cols']
    return pigments, targets

ALL_PIGMENTS, TARGET_COLS = load_meta_data()

# Varsayılan Hedef Tablosu (Boş gelmesin, kullanıcı formatı anlasın)
DEFAULT_TARGETS = pd.DataFrame(
    [
        [30.00, 0.00, 0.00], # 15 Derece
        [30.00, 0.00, 0.00], # 25 Derece
        [30.00, 0.00, 0.00], # 45 Derece
        [30.00, 0.00, 0.00], # 75 Derece
        [30.00, 0.00, 0.00]  # 110 Derece
    ],
    columns=["L*", "a*", "b*"],
    index=["15°", "25°", "45°", "75°", "110°"]
)

# --- 2. HESAPLAMA MOTORU BAĞLANTISI ---
# app_ga.py içindeki run_ai_engine fonksiyonunun YENİ HALİ

def run_ai_engine(target_df, selected_pigments, max_pigment_count):
    # Veri Hazırlığı
    try:
        target_matrix = target_df.values.astype(np.float32)
        target_lab = target_matrix.flatten()
    except ValueError:
        yield "❌ Hata: Tabloya sadece sayı giriniz.", pd.DataFrame(), "---"
        return

    if not selected_pigments:
        selected_pigments = ALL_PIGMENTS # Eğer boşsa hepsini seç
    
    yield "🏎️ Turbo Motor Başlatılıyor...", pd.DataFrame(), ""
    
    # Optimizer Başlat
    optimizer = GeneticOptimizer(target_lab, ALL_PIGMENTS, TARGET_COLS)
    
    # Kısıtlamalar
    model_input_cols = optimizer.input_cols 
    allowed_indices = [i for i, col in enumerate(model_input_cols) if col in selected_pigments]
    optimizer.set_constraints(allowed_indices, max_pigment_count)
    
    # Başlatma
    population = optimizer.initialize_population()
    best_recipe = None
    best_de = float('inf')
    
    # -----------------------------------------------
    # OPTİMİZASYON DÖNGÜSÜ (ARTIK ÇOK DAHA HIZLI)
    # -----------------------------------------------
    GENERATIONS = 80 # Hızlı olduğu için 80-100 yeterli
    
    for gen in range(GENERATIONS):
        # Tek adımda tüm işlemler (Fitness, Sort, Crossover, Mutate)
        population, current_best_recipe, current_best_de = optimizer.evolve_step(population)
        
        # En iyiyi güncelle
        if current_best_de < best_de:
            best_de = current_best_de
            best_recipe = current_best_recipe
        
        # UI Güncelleme (Her 5 gen'de bir - daha sık güncelleme yapabiliriz artık)
        if gen % 5 == 0:
            active_count = (best_recipe > 0.001).sum().item()
            status_msg = f"🚀 Hızlanıyor... Gen {gen}/{GENERATIONS} | Delta E: {best_de:.3f}"
            yield status_msg, pd.DataFrame(), f"{best_de:.3f}"

    # Sonuç Formatlama (Aynı)
    final_np = best_recipe.cpu().numpy()
    result_data = []
    
    for i, val in enumerate(final_np):
        if val > 0.001: 
            pigment_name = model_input_cols[i]
            result_data.append([pigment_name, round(val * 100, 2)])
    
    df_result = pd.DataFrame(result_data, columns=["Pigment Kodu", "Oran (%)"])
    df_result = df_result.sort_values(by="Oran (%)", ascending=False)
    
    yield "✅ Hesaplama Bitti!", df_result, f"🏆 {best_de:.4f}"

# --- 3. ARAYÜZ TASARIMI (PRO) ---
# CSS ile tabloyu ve başlıkları güzelleştirelim
custom_css = """
#target_table {font-size: 16px !important;} 
.gradio-container {background-color: #f9fafb;}
"""

with gr.Blocks(title="RenkAI Pro", css=custom_css) as demo:
    gr.Markdown("# 🎨 RenkAI: Akıllı Reçete Asistanı")
    gr.Markdown("Hedeflenen L, a, b değerlerini tabloya girin. Tab tuşu ile hücreler arasında hızlıca gezebilirsiniz.")

    with gr.Row():
        # --- SOL PANEL: GİRİŞLER ---
        with gr.Column(scale=4):
            
            # 1. HEDEF RENK TABLOSU (En Hızlı Giriş Yöntemi)
            with gr.Group():
                gr.Markdown("### 1. Hedef Renk Verileri")
                target_input = gr.Dataframe(
                    value=DEFAULT_TARGETS,
                    headers=["L*", "a*", "b*"],
                    row_count=(5, "fixed"),
                    col_count=(3, "fixed"),
                    interactive=True,
                    label="L, a, b Değerleri (15°, 25°, 45°, 75°, 110°)",
                    elem_id="target_table"
                )
            
            # 2. AYARLAR & STOK
            gr.Markdown("### 2. Stok ve Kısıtlamalar")
            with gr.Accordion("⚙️ Pigment Seçimi ve Ayarlar", open=False):
                max_pigment_slider = gr.Slider(
                    minimum=3, maximum=8, value=5, step=1, 
                    label="Maksimum Pigment Sayısı",
                    info="Reçetenin en fazla kaç çeşit pigmentten oluşmasını istersiniz?"
                )
                
                gr.Markdown("---")
                # Pigment Seçimi Butonları
                with gr.Row():
                    select_all_btn = gr.Button("✅ Tümünü Seç", size="sm")
                    deselect_btn = gr.Button("❌ Seçimi Temizle", size="sm")
                
                pigment_selector = gr.CheckboxGroup(
                    choices=ALL_PIGMENTS, 
                    value=ALL_PIGMENTS, 
                    label="Kullanılabilir Stok Listesi",
                    info="Elinizde olmayan veya kullanmak istemediğiniz pigmentlerin işaretini kaldırın."
                )
                
                # Buton Fonksiyonları
                select_all_btn.click(lambda: ALL_PIGMENTS, outputs=pigment_selector)
                deselect_btn.click(lambda: [], outputs=pigment_selector)

            # HESAPLA BUTONU
            calc_btn = gr.Button("🚀 Reçeteyi Hesapla", variant="primary", size="lg")

        # --- SAĞ PANEL: SONUÇLAR ---
        with gr.Column(scale=3):
            gr.Markdown("### 🧬 Analiz Sonucu")
            
            with gr.Group():
                delta_lbl = gr.Label(value="---", label="Tahmini Delta E")
                status_lbl = gr.Label(value="Hazır", label="Durum")
            
            result_df = gr.Dataframe(
                headers=["Pigment Kodu", "Oran (%)"],
                datatype=["str", "number"],
                label="Önerilen Reçete",
                interactive=False
            )

    # --- BAĞLANTILAR ---
    calc_btn.click(
        fn=run_ai_engine,
        inputs=[target_input, pigment_selector, max_pigment_slider],
        outputs=[status_lbl, result_df, delta_lbl]
    )

if __name__ == "__main__":
    demo.queue().launch()