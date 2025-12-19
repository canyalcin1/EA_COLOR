import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os

# --- MODEL CLASS ---
class PigmentColorNet(nn.Module):
    def __init__(self, num_pigments, out_dim, config):
        super().__init__()
        self.pigment_embedding = nn.Linear(num_pigments, config['EMBED_DIM'], bias=False)
        layers = []
        input_size = config['EMBED_DIM']
        for _ in range(config['N_LAYERS']):
            layers.append(nn.Linear(input_size, config['HIDDEN_DIM']))
            if config['USE_BATCHNORM']: layers.append(nn.BatchNorm1d(config['HIDDEN_DIM']))
            layers.append(nn.SiLU())
            if config['DROPOUT'] > 0: layers.append(nn.Dropout(config['DROPOUT']))
            input_size = config['HIDDEN_DIM']
        layers.append(nn.Linear(input_size, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(self.pigment_embedding(x))

def predict_color(input_csv, output_csv="Results.csv"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_FOLDER = "models_production"
    
    model_files = glob.glob(f"{MODEL_FOLDER}/model_fold_*.pt")
    if not model_files:
        print(f"❌ '{MODEL_FOLDER}' boş! Önce train_production.py çalıştır.")
        return

    print(f"⚙️ TAHMİN MOTORU BAŞLATILIYOR... ({len(model_files)} Uzman Model)")
    
    # Veri Okuma
    try:
        df = pd.read_csv(input_csv, sep=None, engine='python')
    except Exception as e:
        print(f"Dosya hatası: {e}")
        return

    # Meta verileri ilk modelden çek
    chk = torch.load(model_files[0], map_location=DEVICE, weights_only=False)
    in_cols = chk['input_cols']
    tgt_cols = chk['target_cols']
    
    # Temizlik
    for c in in_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c].str.replace(',', '.', regex=False), errors='coerce')
            df[c] = df[c].fillna(0.0)
        else:
            # Eğer CSV'de pigment sütunu eksikse 0.0 olarak ekle
            df[c] = 0.0
            
    # Girdi Tensörü
    X_raw = df[in_cols].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1.0 # 0'a bölme hatası önle
    tensor_in = torch.tensor(X_raw/row_sums).to(DEVICE)
    
    # Ensemble Tahmin
    all_preds = []
    print("🧠 Renkler Hesaplanıyor...", end="")
    
    for m_path in model_files:
        checkpoint = torch.load(m_path, map_location=DEVICE, weights_only=False)
        model = PigmentColorNet(len(in_cols), len(tgt_cols), checkpoint['config']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        scaler = checkpoint['scaler']
        
        with torch.no_grad():
            pred_scaled = model(tensor_in).cpu().numpy()
            pred_real = scaler.inverse_transform(pred_scaled)
            all_preds.append(pred_real)
        print(".", end="")
            
    print(" ✅")
    
    # Ortalama
    avg_preds = np.mean(all_preds, axis=0)
    
    # Sonuçları Kaydet
    # Sadece SampleNo ve Tahminleri içeren temiz bir çıktı verelim
    result_df = pd.DataFrame()
    if "SampleNo" in df.columns:
        result_df["SampleNo"] = df["SampleNo"]
    
    for i, col in enumerate(tgt_cols):
        result_df[col] = np.round(avg_preds[:, i], 2)
        
    # Eğer gerçek değerler varsa (Test amaçlı) Delta E ekle
    if set(tgt_cols).issubset(df.columns):
        # Gerçek değerlerdeki virgülleri de temizle
        for c in tgt_cols:
             if df[c].dtype == object: df[c] = pd.to_numeric(df[c].str.replace(',', '.'), errors='coerce')
             
        targets = df[tgt_cols].values.astype(np.float32)
        diff = targets.reshape(-1,5,3) - avg_preds.reshape(-1,5,3)
        de = np.mean(np.sqrt(np.sum(diff**2, axis=2)), axis=1)
        result_df["Delta_E"] = np.round(de, 3)
        print(f"📊 Ortalama Hata (Delta E): {np.mean(de):.3f}")

    result_df.to_csv(output_csv, index=False)
    print(f"📄 Tahminler Kaydedildi: {output_csv}")

if __name__ == "__main__":
    # Kullanım: Hangi dosyayı tahmin etmek istiyorsan buraya yaz
    predict_color("eval_dataset_clean.csv", "Final_Report_Prod.csv")