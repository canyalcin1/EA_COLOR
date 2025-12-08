import torch
import pandas as pd
import numpy as np
from Train_DeltaE_3.train import PigmentColorNet, Config

# --- AYARLAR ---
MODEL_PATH = "3DeltaEModel.pt"
INPUT_CSV = "RS400_Clean.csv"   #"eval_dataset_clean.csv"     
OUTPUT_CSV = "RS400_Clean_output.csv"   # Dosya adı güncellendi

def calculate_delta_e_matrix(y_true, y_pred):
    """
    y_true ve y_pred: [N, 15] boyutunda numpy dizileri.
    Her satır için 5 açının ortalama Delta E'sini hesaplar.
    Dönüş: [N] boyutunda dizi.
    """
    # [Batch, 15] -> [Batch, 5, 3] (5 Açı, 3 Kanal: L,a,b)
    # Veri setindeki sütun sırasının 15L, 15a, 15b, 25L... şeklinde olduğunu varsayıyoruz.
    n_samples = y_true.shape[0]
    
    t_reshaped = y_true.reshape(n_samples, 5, 3)
    p_reshaped = y_pred.reshape(n_samples, 5, 3)
    
    # Farklar
    diff = t_reshaped - p_reshaped
    
    # Delta E = sqrt(dL^2 + da^2 + db^2)
    # axis=2 (L,a,b ekseni) boyunca norm alıyoruz
    de_per_angle = np.sqrt(np.sum(diff**2, axis=2)) # Sonuç: [N, 5]
    
    # 5 Açının Ortalaması (Her numune için tek bir skor)
    mean_de_per_sample = np.mean(de_per_angle, axis=1) # Sonuç: [N]
    
    return mean_de_per_sample

def batch_predict_and_compare():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Sistem Başlatılıyor... ({device})")

    # 1. MODEL YÜKLEME
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"❌ HATA: '{MODEL_PATH}' bulunamadı.")
        return

    input_cols = checkpoint['input_cols']
    target_cols = checkpoint['target_cols']
    scaler = checkpoint['scaler']

    model = PigmentColorNet(len(input_cols), len(target_cols)).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("✅ Model Yüklendi.")

    # 2. VERİ OKUMA
    try:
        df = pd.read_csv(INPUT_CSV, sep=None, engine='python')
    except FileNotFoundError:
        print(f"❌ HATA: '{INPUT_CSV}' bulunamadı.")
        return

    # Sayısal Temizlik
    cols_to_numeric = input_cols + target_cols
    for col in cols_to_numeric:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[input_cols] = df[input_cols].fillna(0.0)

    # 3. TAHMİN HAZIRLIĞI
    X_raw = df[input_cols].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X_norm = X_raw / row_sums
    
    tensor_in = torch.tensor(X_norm).to(device)

    # 4. TAHMİN ALMA
    print("🧠 Tahminler hesaplanıyor...")
    with torch.no_grad():
        preds_scaled = model(tensor_in).cpu().numpy()
        preds_real = scaler.inverse_transform(preds_scaled)

    # 5. DELTA E HESAPLAMA (TÜM MATRİS İÇİN)
    # Gerçek değerleri DataFrame'den çekelim
    targets_real = df[target_cols].values.astype(np.float32)
    
    # Her satır için Delta E hesapla
    delta_e_scores = calculate_delta_e_matrix(targets_real, preds_real)
    
    # Genel Ortalama (Analiz Sonucu)
    global_mean_de = np.mean(delta_e_scores)

    # 6. RAPORLAMA
    print("📝 Rapor hazırlanıyor...")
    
    output_rows = []
    id_col = "SampleNo" if "SampleNo" in df.columns else None
    
    # Rapor Sütun Sırası: SampleNo | Type | Delta_E | Hedefler... | Girdiler...
    final_columns = ([id_col] if id_col else []) + ["Type", "Mean_DeltaE"] + target_cols + input_cols

    for i in range(len(df)):
        # --- A. REAL SATIRI ---
        row_real = df.iloc[i].to_dict()
        row_real["Type"] = "REAL"
        row_real["Mean_DeltaE"] = 0.0 # Gerçeğin kendine uzaklığı 0'dır
        
        for col in input_cols:
            if col not in row_real: row_real[col] = 0.0
        output_rows.append(row_real)

        # --- B. PRED SATIRI ---
        row_pred = row_real.copy()
        row_pred["Type"] = "PRED"
        
        # HESAPLANAN DELTA E'Yİ BURAYA YAZIYORUZ
        row_pred["Mean_DeltaE"] = round(delta_e_scores[i], 3)

        # Tahmin Değerlerini Yaz
        for j, t_col in enumerate(target_cols):
            row_pred[t_col] = round(preds_real[i, j], 2)
            
        output_rows.append(row_pred)

    # 7. KAYDETME
    result_df = pd.DataFrame(output_rows, columns=final_columns)
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    print("=" * 50)
    print(f"✅ İŞLEM BAŞARIYLA TAMAMLANDI")
    print(f"📄 Rapor Dosyası: {OUTPUT_CSV}")
    print("-" * 30)
    print(f"📊 ANALİZ SONUCU (5 Açı Ortalaması):")
    print(f"🌍 Tüm Veri Seti Ortalama Delta E: {global_mean_de:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    batch_predict_and_compare()