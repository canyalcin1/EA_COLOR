import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os

# --- MODEL SINIFI (Kayıtlı ayarlara göre kendini kurar) ---
class PigmentColorNet(nn.Module):
    def __init__(self, num_pigments, out_dim, config):
        super().__init__()
        self.pigment_embedding = nn.Linear(num_pigments, config['EMBED_DIM'], bias=False)
        
        layers = []
        input_size = config['EMBED_DIM']
        
        for _ in range(config['N_LAYERS']):
            layers.append(nn.Linear(input_size, config['HIDDEN_DIM']))
            if config['USE_BATCHNORM']: 
                layers.append(nn.BatchNorm1d(config['HIDDEN_DIM']))
            
            layers.append(nn.SiLU())
            
            if config['DROPOUT'] > 0: 
                layers.append(nn.Dropout(config['DROPOUT']))
            
            input_size = config['HIDDEN_DIM']
            
        layers.append(nn.Linear(input_size, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(self.pigment_embedding(x))

def calculate_delta_e(targets, preds):
    """
    [N, 15] boyutundaki hedef ve tahminleri alır,
    satır bazında ortalama Delta E döndürür.
    """
    diff = targets.reshape(-1, 5, 3) - preds.reshape(-1, 5, 3)
    # Her açı için Delta E
    de_per_angle = np.sqrt(np.sum(diff**2, axis=2))
    # Tüm açıların ortalaması (Her numune için tek değer)
    return np.mean(de_per_angle, axis=1)

def main():
    # --- AYARLAR ---
    INPUT_CSV = "RS400_Clean.csv" 
    OUTPUT_CSV = "Detailed_Analysis_Result_RS.csv"
    MODEL_FOLDER = "models_final" # Son eğitimde buraya kaydettik
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Modelleri Bul
    model_files = glob.glob(f"{MODEL_FOLDER}/model_fold_*.pt")
    if not model_files:
        print(f"❌ '{MODEL_FOLDER}' klasöründe model bulunamadı!")
        return
    
    print(f"⚙️ DETAYLI ANALİZ BAŞLIYOR... ({len(model_files)} Model İncelenecek)")
    
    # 1. VERİYİ OKU VE HAZIRLA
    try:
        df = pd.read_csv(INPUT_CSV, sep=None, engine='python')
    except Exception as e:
        print(f"Hata: {e}")
        return

    # İlk modelden meta verileri çek
    chk = torch.load(model_files[0], map_location=DEVICE, weights_only=False)
    in_cols, tgt_cols = chk['input_cols'], chk['target_cols']
    
    # Temizlik
    for c in in_cols + tgt_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].str.replace(',', '.', regex=False), errors='coerce')
    df[in_cols] = df[in_cols].fillna(0.0)
    
    # Tensor Hazırlığı (V1 - Ağırlık Bazlı)
    X_raw = df[in_cols].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1.0
    tensor_in = torch.tensor(X_raw/row_sums).to(DEVICE)
    
    # Hedefler (Delta E için)
    if set(tgt_cols).issubset(df.columns):
        targets = df[tgt_cols].values.astype(np.float32)
    else:
        print("⚠️ Uyarı: Hedef sütunlar (L,a,b) eksik, Delta E hesaplanamaz.")
        return

    # 2. TEKİL MODEL ANALİZİ
    print("\n" + "="*50)
    print(f"{'MODEL':<15} | {'ORTALAMA ΔE':<15} | {'DURUM':<10}")
    print("="*50)
    
    all_preds = []
    results = []
    
    for i, m_path in enumerate(model_files):
        # Yükle
        checkpoint = torch.load(m_path, map_location=DEVICE, weights_only=False)
        model = PigmentColorNet(len(in_cols), len(tgt_cols), checkpoint['config']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        scaler = checkpoint['scaler']
        
        # Tahmin
        with torch.no_grad():
            pred_scaled = model(tensor_in).cpu().numpy()
            pred_real = scaler.inverse_transform(pred_scaled)
            all_preds.append(pred_real)
            
        # Delta E Hesapla
        de_scores = calculate_delta_e(targets, pred_real)
        mean_de = np.mean(de_scores)
        
        # Sonucu Kaydet
        model_name = os.path.basename(m_path)
        print(f"{model_name:<15} | {mean_de:<15.4f} | {'✅' if mean_de < 3.0 else '⚠️'}")
        
        results.append({
            "Model": model_name,
            "Mean_DeltaE": mean_de,
            "Predictions": pred_real
        })
        
        # CSV'ye her modelin Delta E'sini ekleyelim (Analiz için)
        df[f"DE_{model_name}"] = np.round(de_scores, 3)

    # 3. ENSEMBLE ANALİZİ
    avg_preds = np.mean(all_preds, axis=0)
    ensemble_de_scores = calculate_delta_e(targets, avg_preds)
    ensemble_mean_de = np.mean(ensemble_de_scores)
    
    print("-" * 50)
    print(f"{'ENSEMBLE (ORT)':<15} | {ensemble_mean_de:<15.4f} | 🌟")
    print("=" * 50)
    
    # 4. KAZANANI BELİRLE
    # Tekil modellerden en iyisini bul
    best_single_model = min(results, key=lambda x: x["Mean_DeltaE"])
    
    print("\n🏆 KARŞILAŞTIRMA RAPORU:")
    print(f"🥇 En İyi Tekil Model: {best_single_model['Model']} (ΔE: {best_single_model['Mean_DeltaE']:.4f})")
    print(f"🤝 Ensemble Modeli   : (ΔE: {ensemble_mean_de:.4f})")
    
    if best_single_model['Mean_DeltaE'] < ensemble_mean_de:
        diff = ensemble_mean_de - best_single_model['Mean_DeltaE']
        print(f"\n🚀 SONUÇ: Tekil model, Ensemble'dan {diff:.4f} puan DAHA İYİ!")
        print(f"👉 Tavsiye: Sadece '{best_single_model['Model']}' dosyasını kullanabilirsin.")
    else:
        diff = ensemble_mean_de - best_single_model['Mean_DeltaE'] # Negatif olacak
        print(f"\n🛡️ SONUÇ: Ensemble, en iyi tekil modelden {abs(diff):.4f} puan DAHA GÜVENLİ.")
        print("👉 Tavsiye: Ensemble yapısını kullanmaya devam et.")

    # Kaydet
    df["Ensemble_DeltaE"] = np.round(ensemble_de_scores, 3)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📄 Detaylı Rapor Kaydedildi: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()