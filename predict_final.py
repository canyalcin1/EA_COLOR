import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# --- DİNAMİK MODEL YÜKLEYİCİ ---
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
            
            # Tahmin modunda Dropout kapalıdır ama mimari eşleşmesi için ekliyoruz
            if config['DROPOUT'] > 0:
                layers.append(nn.Dropout(config['DROPOUT']))
                
            input_size = config['HIDDEN_DIM']
            
        layers.append(nn.Linear(input_size, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        emb = self.pigment_embedding(x)
        out = self.net(emb)
        return out

def main():
    MODEL_PATH = "FinalModel.pt"
    # İster eğitim (RS400) ister test (eval_dataset) dosyanı buraya yaz
    INPUT_CSV = "eval_dataset_clean.csv" 
    OUTPUT_CSV = "Final_Sonuc_1M.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"⚙️ Final Tahmin Başlıyor... ({DEVICE})")

    # 1. Modeli Yükle (Config bilgisiyle beraber)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        print("❌ Model dosyası bulunamadı. Önce train_final.py çalıştır!")
        return

    config = checkpoint['config']
    input_cols = checkpoint['input_cols']
    target_cols = checkpoint['target_cols']
    scaler = checkpoint['scaler']

    # Model mimarisini kaydedilen ayarlara göre kur
    model = PigmentColorNet(len(input_cols), len(target_cols), config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("✅ Model ve Ayarlar Yüklendi.")

    # 2. Veri Okuma
    df = pd.read_csv(INPUT_CSV, sep=None, engine='python')
    
    cols_to_numeric = input_cols + target_cols
    for col in cols_to_numeric:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[input_cols] = df[input_cols].fillna(0.0)

    # 3. Tahmin
    X_raw = df[input_cols].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X_norm = X_raw / row_sums
    
    tensor_in = torch.tensor(X_norm).to(DEVICE)
    
    print("🧠 Hesaplamalar yapılıyor...")
    with torch.no_grad():
        preds_scaled = model(tensor_in).cpu().numpy()
        preds_real = scaler.inverse_transform(preds_scaled)

    # 4. Delta E Hesapla
    if set(target_cols).issubset(df.columns):
        targets_real = df[target_cols].values.astype(np.float32)
        
        # Delta E
        n = len(df)
        t_reshaped = targets_real.reshape(n, 5, 3)
        p_reshaped = preds_real.reshape(n, 5, 3)
        diff = t_reshaped - p_reshaped
        delta_e = np.mean(np.sqrt(np.sum(diff**2, axis=2)), axis=1)
        
        print(f"📊 Ortalama Delta E: {np.mean(delta_e):.3f}")
        df["Mean_DeltaE"] = np.round(delta_e, 3)
        df["Type"] = "REAL" # Orijinal satırlar
    
    # Sonuçları Kaydet
    # Tahminleri de sütun olarak ekleyelim (Opsiyonel: ayrı satır yerine sütun ekliyorum daha temiz olsun)
    for i, col in enumerate(target_cols):
        df[f"Pred_{col}"] = np.round(preds_real[:, i], 2)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Kaydedildi: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()