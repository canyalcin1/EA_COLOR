import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os

# --- MODEL SINIFI (Eğitimdekiyle aynı yapı) ---
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
            
            # Mimari eşleşmesi için Dropout katmanını ekliyoruz
            if config['DROPOUT'] > 0: 
                layers.append(nn.Dropout(config['DROPOUT']))
            
            input_size = config['HIDDEN_DIM']
            
        layers.append(nn.Linear(input_size, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(self.pigment_embedding(x))

# --- HACİM DÖNÜŞÜM FONKSİYONU ---
def convert_weight_to_volume(df, input_cols, density_map):
    """
    Model dosyasından gelen density_map'i kullanarak
    Ağırlık -> Hacim dönüşümü yapar.
    """
    print("⚗️ Ağırlık -> Hacim (Density) Dönüşümü Yapılıyor...")
    X_vol = df[input_cols].copy()
    
    for col in input_cols:
        # Density haritasından değeri çek (Yoksa varsayılan 1.5)
        # String'e çevirerek arıyoruz çünkü JSON/Dict anahtarları genelde stringdir
        density = density_map.get(str(col), density_map.get("DEFAULT", 1.5))
        X_vol[col] = X_vol[col] / density
        
    # Satır bazında 100'e normalize et (Hacimce Yüzde)
    row_sums = X_vol.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    X_vol = X_vol.div(row_sums, axis=0)
    
    return X_vol.values.astype(np.float32)

def main():
    # GİRİŞ DOSYASI (Test etmek istediğin dosya)
    INPUT_CSV = "RS400_Clean.csv" 
    OUTPUT_CSV = "Ensemble_V2_Density_Sonuc_RS.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_files = glob.glob("models_v2/model_fold_*.pt")
    if not model_files:
        print("❌ Model dosyaları bulunamadı! Önce train_ensemble_v2.py çalıştır.")
        return
        
    print(f"⚙️ Ensemble V2 (Density Mode) Tahmin Başlıyor... ({len(model_files)} Model)")
    
    # 1. Veri Oku
    try:
        df = pd.read_csv(INPUT_CSV, sep=None, engine='python')
    except Exception as e:
        print(f"Hata: {e}")
        return

    # İlk modelden konfigürasyonu ve DENSITY MAP'i çek
    # weights_only=False çünkü içinde dictionary ve scaler var
    chk = torch.load(model_files[0], map_location=DEVICE, weights_only=False)
    
    in_cols = chk['input_cols']
    tgt_cols = chk['target_cols']
    density_map = chk.get('density_map', None) # Density map'i al
    
    if density_map is None:
        print("⚠️ UYARI: Model dosyasında density_map bulunamadı! Eski model olabilir.")
        # Acil durum varsayılanı (ama training kodunu çalıştırdıysan bu olmaz)
        density_map = {"DEFAULT": 1.5}

    # Sayısal Temizlik
    for c in in_cols + tgt_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].str.replace(',', '.', regex=False), errors='coerce')
    df[in_cols] = df[in_cols].fillna(0.0)
    
    # --- KRİTİK ADIM: HACİM DÖNÜŞÜMÜ ---
    # Model eğitimi hacim verisiyle yapıldığı için, testi de hacme çeviriyoruz
    X_vol = convert_weight_to_volume(df, in_cols, density_map)
    tensor_in = torch.tensor(X_vol).to(DEVICE)
    
    # 2. Ensemble Döngüsü
    all_preds = []
    
    for i, m_path in enumerate(model_files):
        checkpoint = torch.load(m_path, map_location=DEVICE, weights_only=False)
        
        model = PigmentColorNet(len(in_cols), len(tgt_cols), checkpoint['config']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        scaler = checkpoint['scaler']
        
        with torch.no_grad():
            pred_scaled = model(tensor_in).cpu().numpy()
            pred_real = scaler.inverse_transform(pred_scaled)
            all_preds.append(pred_real)
        
        print(f"✅ Model {i+1} Tahmini Tamam.")
            
    # 3. Ortalamayı Al
    avg_preds = np.mean(all_preds, axis=0)
    
    # 4. Sonuçları Raporla
    if set(tgt_cols).issubset(df.columns):
        targets = df[tgt_cols].values.astype(np.float32)
        diff = targets.reshape(-1,5,3) - avg_preds.reshape(-1,5,3)
        delta_e = np.mean(np.sqrt(np.sum(diff**2, axis=2)), axis=1)
        
        print("="*40)
        print(f"📊 ENSEMBLE V2 (DENSITY) ORTALAMA DELTA E: {np.mean(delta_e):.3f}")
        print("="*40)
        df["EnsembleV2_Density_DeltaE"] = np.round(delta_e, 3)
    
    # Tahminleri dosyaya yaz
    for i, col in enumerate(tgt_cols):
        df[f"Pred_{col}"] = np.round(avg_preds[:, i], 2)
        
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Sonuçlar Kaydedildi: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()