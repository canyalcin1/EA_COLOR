import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. AYARLAR (YÖNETİCİ PANELİ) ---
class Config:
    CSV_PATH = "RS400_Clean.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Hiperparametreleri
    EMBED_DIM = 24    # Pigmentlerin "Kimyasal Karakter" vektör boyutu
    HIDDEN_DIM = 128  # Karışım motorunun nöron sayısı
    DROPOUT = 0.05    # Overfitting engelleyici
    
    # Eğitim Ayarları
    LR = 1e-3         # Öğrenme hızı
    EPOCHS = 3000     # Toplam tur sayısı
    BATCH_SIZE = 64   # Her adımda işlenen veri sayısı
    PATIENCE = 300    # Erken durdurma sabrı

    # HEDEFLER (5 Açı x 3 Değer = 15 Çıkış)
    ANGLES = ['15', '25', '45', '75', '110']
    TARGET_COLS = [f"{ang}{ch}" for ang in ANGLES for ch in ['L', 'a', 'b']]

    # GİRİŞLER (Pigmentler) - CSV Header'dan alınmıştır
 
    INPUT_COLS = [
        "726C", "718Y", "783S", "772S", "755C", "744S", "717C", "980", "910", "135", 
        "828", "835", "831", "826", "836", "838", "815", "856", "892", "895", "855", 
        "851", "858", "839", "896", "862", "865", "891", "898", "845", "832", "816", 
        "818", "233", "940", "424", "356", "384", "306", "318", "331", "198", "154", 
        "290", "522", "322", "570", "410", "550", "530", "371", "482", "343", "030", 
        "960", "580", "670", "632", "110", "821", "190"
    ]

# --- 2. VERİ HAZIRLIĞI ---
def load_and_prep_data():
    print(f"📂 Veri Yükleniyor: {Config.CSV_PATH}")
    
    # 1. Önce dosyayı düz okuyalım (Ayırıcıyı otomatik algılasın)
    # Motoru 'python' yapıyoruz ki daha esnek olsun.
    df = pd.read_csv(Config.CSV_PATH, sep=None, engine='python')

    # 2. VİRGÜL / NOKTA TEMİZLİĞİ (Hem pigmentler hem hedefler için)
    # Tüm sayısal olması gereken sütunları geziyoruz
    cols_to_fix = Config.INPUT_COLS + Config.TARGET_COLS
    
    for col in cols_to_fix:
        if col in df.columns:
            # Eğer sütun string (object) tipindeyse temizlik yap
            if df[col].dtype == object:
                # Virgülleri noktaya çevir
                df[col] = df[col].str.replace(',', '.', regex=False)
                # Sayıya çevir, hata verenleri (boşluk vs.) NaN yap
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Eksik Verileri Doldur
    df[Config.INPUT_COLS] = df[Config.INPUT_COLS].fillna(0.0)
    df = df.dropna(subset=Config.TARGET_COLS).reset_index(drop=True)

    # 4. KONTROL NOKTASI (Burası hatayı gösterecek)
    print("-" * 30)
    print("🔎 VERİ KONTROLÜ:")
    print(f"Toplam Satır Sayısı: {len(df)}")
    
    # Pigmentlerin toplamı 0 olan satır var mı?
    X_temp = df[Config.INPUT_COLS].values
    zero_rows = (X_temp.sum(axis=1) == 0).sum()
    print(f"Pigment Toplamı 0 olan satır sayısı: {zero_rows}")
    if zero_rows > 0:
        print("⚠️ UYARI: Bazı satırlarda hiç pigment okunamadı! CSV formatı bozuk.")
    else:
        print("✅ Tüm satırlarda pigment verisi okundu.")
        
    # Örnek bir L değeri (Target) kontrolü
    print(f"Örnek 15L Değeri (İlk Satır): {df[Config.TARGET_COLS[0]].iloc[0]}")
    print("-" * 30)

    # 5. ROW NORMALIZATION
    X_raw = df[Config.INPUT_COLS].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 
    X = X_raw / row_sums

    y = df[Config.TARGET_COLS].values.astype(np.float32)

    # 6. Scaler
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    return X, y_scaled, y_scaler


# --- 3. MODEL MİMARİSİ (NESTED LEARNING / DEEP SETS) ---
class PigmentColorNet(nn.Module):
    def __init__(self, num_pigments, out_dim):
        super().__init__()
        
        # KATMAN 1: Embedding (Representation Learning)
        # Giriş [Batch, 60] -> Çıkış [Batch, EMBED_DIM]
        # Bias=False çünkü 0 pigmentin etkisi 0 olmalı.
        # Bu katman aslında şu işlemi yapar: Karışımdaki her pigmentin "kimlik vektörünü" 
        # oranıyla çarpıp toplar. (Weighted Sum of Embeddings)
        self.pigment_embedding = nn.Linear(num_pigments, Config.EMBED_DIM, bias=False)
        
        # KATMAN 2: Regressor (Non-Linear Physics Mapping)
        # Kimyasal özeti alıp renge dönüştüren fizik motoru.
        self.regressor = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, Config.HIDDEN_DIM),
            nn.SiLU(), # Swish aktivasyonu (ReLU'dan daha yumuşak)
            nn.Dropout(Config.DROPOUT),
            
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
            
            nn.Linear(Config.HIDDEN_DIM, out_dim)
        )
        
    def forward(self, x):
        # x: [Batch, 60] (Normalize edilmiş oranlar)
        embedding_mix = self.pigment_embedding(x) 
        output = self.regressor(embedding_mix)
        return output

# --- 4. YARDIMCI: DELTA E HESAPLAMA ---
def calculate_delta_e_mean(y_true, y_pred):
    # y_true ve y_pred shape: [Batch, 15] -> (5 açı x 3 kanal)
    # Her 3'lü grubu (L,a,b) ayırıp öklid mesafesi alacağız.
    batch_size = y_true.shape[0]
    total_de = 0
    
    # Veriyi [Batch, 5, 3] formatına çevir (5 Açı, 3 Kanal)
    y_t_reshaped = y_true.reshape(batch_size, 5, 3)
    y_p_reshaped = y_pred.reshape(batch_size, 5, 3)
    
    # Delta E = sqrt(dL^2 + da^2 + db^2)
    diff = y_t_reshaped - y_p_reshaped
    delta_e_per_angle = np.sqrt(np.sum(diff**2, axis=2)) # [Batch, 5]
    
    return np.mean(delta_e_per_angle) # Tüm açıların ve batch'in ortalaması

# --- 5. EĞİTİM DÖNGÜSÜ ---
def main():
    # Cihaz Ayarı
    device = torch.device(Config.DEVICE)
    print(f"🚀 Sistem Başlatılıyor... Cihaz: {device}")

    # Veriyi Hazırla
    X, y, y_scaler = load_and_prep_data()
    
    # Train/Val Split (%15 Validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Tensorlara Çevir
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Model Kurulumu
    model = PigmentColorNet(num_pigments=X.shape[1], out_dim=y.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    criterion = nn.HuberLoss() # Outlier'lara karşı MSE'den daha dayanıklı
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    print(f"🧠 Eğitim Başladı ({Config.EPOCHS} Epoch)...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item()
                
                # İstatistik için sakla
                all_preds.append(pred.cpu().numpy())
                all_targets.append(by.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Scheduler Adımı
        scheduler.step(avg_val_loss)
        
        # Delta E Hesaplama (Gerçek Skaladan)
        val_preds_scaled = np.vstack(all_preds)
        val_targets_scaled = np.vstack(all_targets)
        
        val_preds_real = y_scaler.inverse_transform(val_preds_scaled)
        val_targets_real = y_scaler.inverse_transform(val_targets_scaled)
        
        current_delta_e = calculate_delta_e_mean(val_targets_real, val_preds_real)

        # Loglama ve Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            checkpoint = {
                    'model_state': model.state_dict(),
                    'scaler': y_scaler,  # <--- Bunu ekledik!
                    'input_cols': Config.INPUT_COLS,
                    'target_cols': Config.TARGET_COLS
                }
            torch.save(checkpoint, "model.pt")

            best_msg = "💾 (New Best)"
        else:
            patience_counter += 1
            best_msg = ""
            
        if epoch % 50 == 0:
            print(f"Ep {epoch:04d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | ΔE: {current_delta_e:.3f} {best_msg}")
            
        if patience_counter >= Config.PATIENCE:
            print(f"⏹️ Erken durdurma tetiklendi (Epoch {epoch}).")
            break

    print("\n✅ Eğitim Tamamlandı.")
    print(f"🏆 En iyi Validation Delta E: {current_delta_e:.3f}")
    
    # --- SON TEST ---
    # Kaydedilen en iyi modeli geri yükle
    model.load_state_dict(torch.load("model.pt", weights_only=False))
    model.eval()
    
    # Örnek Bir Tahmin Göster (İlk Validation Verisi)
    sample_in = torch.tensor(X_val[:1]).to(device)
    sample_out_scaled = model(sample_in).detach().cpu().numpy()
    sample_out_real = y_scaler.inverse_transform(sample_out_scaled)
    sample_target_real = y_scaler.inverse_transform(y_val[:1])
    
    print("\n🔎 Örnek Tahmin (Açı 45 için L,a,b):")
    # 45 derece indeksleri: 6,7,8 (0'dan başlar: 15L,15a,15b, 25L,25a,25b, 45L...)
    print(f"Gerçek: {sample_target_real[0, 6:9]}")
    print(f"Tahmin: {sample_out_real[0, 6:9]}")

if __name__ == "__main__":
    main()