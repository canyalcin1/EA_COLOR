import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. MİMARİ TANIMI (Eğitimdekiyle BİREBİR AYNI olmalı) ---
class Config:
    # Agresif eğitimdeki parametreler
    EMBED_DIM = 64
    HIDDEN_DIM = 512
    DROPOUT = 0.0 # Eğitimde 0.0 yapmıştık

class PigmentColorNet(nn.Module):
    def __init__(self, num_pigments, out_dim):
        super().__init__()
        
        # Giriş
        self.pigment_embedding = nn.Linear(num_pigments, Config.EMBED_DIM, bias=False)
        
        # Derin Fizik Motoru (4 Katman + BatchNorm)
        self.net = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.SiLU(),
            
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.SiLU(),
            
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.SiLU(),
            
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.SiLU(),
            
            nn.Linear(Config.HIDDEN_DIM, out_dim)
        )
        
    def forward(self, x):
        emb = self.pigment_embedding(x)
        out = self.net(emb)
        return out

# --- 2. YARDIMCI FONKSİYONLAR ---
def calculate_delta_e_matrix(y_true, y_pred):
    n_samples = y_true.shape[0]
    t_reshaped = y_true.reshape(n_samples, 5, 3)
    p_reshaped = y_pred.reshape(n_samples, 5, 3)
    diff = t_reshaped - p_reshaped
    de_per_angle = np.sqrt(np.sum(diff**2, axis=2))
    return np.mean(de_per_angle, axis=1)

# --- 3. ANA TAHMİN DÖNGÜSÜ ---
def main():
    MODEL_PATH = "AgresifModel.pt"
    INPUT_CSV = "eval_dataset_clean.csv"  # Test etmek istediğin dosya
    OUTPUT_CSV = "EvalAgresif_Sonuc.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"⚙️ Tahmin Başlıyor... ({DEVICE})")

    # 1. Modeli Yükle
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE,weights_only=False)
    except FileNotFoundError:
        print(f"❌ HATA: '{MODEL_PATH}' bulunamadı.")
        return

    # Checkpoint'ten meta verileri al
    input_cols = checkpoint['input_cols']
    target_cols = checkpoint['target_cols']
    scaler = checkpoint['scaler']

    # Modeli başlat
    model = PigmentColorNet(len(input_cols), len(target_cols)).to(DEVICE)
    
    # Ağırlıkları yükle (Strict=True önemlidir, mimari eşleşmezse hata verir)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except RuntimeError as e:
        print("❌ Model mimarisi uyuşmuyor! Train.py'daki sınıf ile buradaki sınıf aynı olmalı.")
        print(e)
        return
        
    model.eval() # BatchNorm'ları test moduna al
    print("✅ Model Başarıyla Yüklendi.")

    # 2. Veriyi Oku
    print(f"📂 Veri okunuyor: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, sep=None, engine='python')

    # Temizlik (Sayıya çevirme)
    cols_to_numeric = input_cols + target_cols
    for col in cols_to_numeric:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[input_cols] = df[input_cols].fillna(0.0)

    # 3. Model Girişini Hazırla
    X_raw = df[input_cols].values.astype(np.float32)
    # Row Normalization
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X_norm = X_raw / row_sums
    
    tensor_in = torch.tensor(X_norm).to(DEVICE)

    # 4. Tahmin Yap
    print("🧠 Hesaplamalar yapılıyor...")
    with torch.no_grad():
        preds_scaled = model(tensor_in).cpu().numpy()
        preds_real = scaler.inverse_transform(preds_scaled)

    # 5. Delta E Hesapla ve Raporla
    targets_real = df[target_cols].values.astype(np.float32)
    delta_e_scores = calculate_delta_e_matrix(targets_real, preds_real)
    global_mean_de = np.mean(delta_e_scores)

    # Çıktı DataFrame'i
    output_rows = []
    id_col = "SampleNo" if "SampleNo" in df.columns else None
    final_columns = ([id_col] if id_col else []) + ["Type", "Mean_DeltaE"] + target_cols + input_cols

    for i in range(len(df)):
        # Real
        row_real = df.iloc[i].to_dict()
        row_real["Type"] = "REAL"
        row_real["Mean_DeltaE"] = 0.0
        for col in input_cols: 
            if col not in row_real: row_real[col] = 0.0
        output_rows.append(row_real)

        # Pred
        row_pred = row_real.copy()
        row_pred["Type"] = "PRED"
        row_pred["Mean_DeltaE"] = round(delta_e_scores[i], 3)
        for j, t_col in enumerate(target_cols):
            row_pred[t_col] = round(preds_real[i, j], 2)
        output_rows.append(row_pred)

    # Kaydet
    result_df = pd.DataFrame(output_rows, columns=final_columns)
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    print("=" * 50)
    print(f"✅ SONUÇLAR KAYDEDİLDİ: {OUTPUT_CSV}")
    print(f"📊 Ortalama Delta E: {global_mean_de:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    main()