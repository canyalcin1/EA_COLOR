import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob

# --- DÜZELTİLMİŞ MODEL SINIFI ---
# (Eğitimdekiyle birebir aynı yapıya getirildi)
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
            
            # --- EKSİK OLAN KISIM BURASIYDI ---
            # Eğitimde Dropout kullandıysak, mimaride yeri ayrılmalı.
            # model.eval() yapınca zaten devre dışı kalır ama yapıda bulunması şarttır.
            if config['DROPOUT'] > 0: 
                layers.append(nn.Dropout(config['DROPOUT']))
            # ----------------------------------
            
            input_size = config['HIDDEN_DIM']
            
        layers.append(nn.Linear(input_size, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(self.pigment_embedding(x))

def main():
    # TEST ETMEK İSTEDİĞİN DOSYA
    INPUT_CSV = "eval_dataset_clean.csv" 
    OUTPUT_CSV = "Ensemble_Sonuc_RS.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_files = glob.glob("models/model_fold_*.pt")
    if not model_files:
        print("❌ Hiç model bulunamadı! Önce train_ensemble.py çalıştır.")
        return
        
    print(f"⚙️ {len(model_files)} Model ile Ensemble Tahmin Yapılıyor...")
    
    # Veriyi Oku
    try:
        df = pd.read_csv(INPUT_CSV, sep=None, engine='python')
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return

    # İlk modelden konfigürasyonu çekip hazırlık yapalım
    chk = torch.load(model_files[0], map_location=DEVICE, weights_only=False)
    in_cols, tgt_cols = chk['input_cols'], chk['target_cols']
    
    # Sayısal Temizlik
    for c in in_cols + tgt_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].str.replace(',', '.', regex=False), errors='coerce')
    df[in_cols] = df[in_cols].fillna(0.0)
    
    # Tensor Hazırlığı
    X_raw = df[in_cols].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1.0
    tensor_in = torch.tensor(X_raw/row_sums).to(DEVICE)
    
    # --- ENSEMBLE DÖNGÜSÜ ---
    all_preds = []
    
    for i, m_path in enumerate(model_files):
        # weights_only=False önemli
        checkpoint = torch.load(m_path, map_location=DEVICE, weights_only=False)
        
        # Modeli başlat ve ağırlıkları yükle
        model = PigmentColorNet(len(in_cols), len(tgt_cols), checkpoint['config']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        scaler = checkpoint['scaler']
        
        with torch.no_grad():
            pred_scaled = model(tensor_in).cpu().numpy()
            pred_real = scaler.inverse_transform(pred_scaled)
            all_preds.append(pred_real)
        
        print(f"✅ Model {i+1} tahmini tamam.")
            
    # ORTALAMA AL (Ensemble Mantığı)
    avg_preds = np.mean(all_preds, axis=0)
    
    # DELTA E HESAPLA
    if set(tgt_cols).issubset(df.columns):
        targets = df[tgt_cols].values.astype(np.float32)
        diff = targets.reshape(-1,5,3) - avg_preds.reshape(-1,5,3)
        delta_e = np.mean(np.sqrt(np.sum(diff**2, axis=2)), axis=1)
        
        print("="*40)
        print(f"📊 ENSEMBLE ORTALAMA DELTA E: {np.mean(delta_e):.3f}")
        print("="*40)
        df["Ensemble_DeltaE"] = np.round(delta_e, 3)
        
    # Tahminleri dosyaya yaz
    # (İstersen tahmin sütunlarını da ekleyebilirsin ama dosya çok şişmesin diye sadece DeltaE ekledim)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Sonuçlar kaydedildi: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()