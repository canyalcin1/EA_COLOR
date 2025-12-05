import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import time

# --- 1. SABİT AYARLAR (Değişmeyecek Olanlar) ---
CSV_PATH = "RS400_Clean.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEARCH_EPOCHS = 300  # Arama yaparken her deneme kaç tur sürsün? (3000 çok uzun sürer, 300 ideal)
PATIENCE = 50        # Arama sırasında erken durdurma

# Hedef ve Giriş Kolonları (Senin kodundan alındı)
ANGLES = ['15', '25', '45', '75', '110']
TARGET_COLS = [f"{ang}{ch}" for ang in ANGLES for ch in ['L', 'a', 'b']]
INPUT_COLS = [
    "726C", "718Y", "783S", "772S", "755C", "744S", "717C", "980", "910", "135", 
    "828", "835", "831", "826", "836", "838", "815", "856", "892", "895", "855", 
    "851", "858", "839", "896", "862", "865", "891", "898", "845", "832", "816", 
    "818", "233", "940", "424", "356", "384", "306", "318", "331", "198", "154", 
    "290", "522", "322", "570", "410", "550", "530", "371", "482", "343", "030", 
    "960", "580", "670", "632", "110", "821", "190"
]

# --- 2. VERİYİ BİR KEZ YÜKLE (RAM'de Tutsun) ---
def load_data():
    print(f"📂 Veri önbelleğe alınıyor...")
    df = pd.read_csv(CSV_PATH, sep=None, engine='python')
    
    # Temizlik
    cols_to_fix = INPUT_COLS + TARGET_COLS
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[INPUT_COLS] = df[INPUT_COLS].fillna(0.0)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)

    # Normalizasyon ve Scale
    X_raw = df[INPUT_COLS].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 
    X = X_raw / row_sums
    y = df[TARGET_COLS].values.astype(np.float32)

    return X, y

# Veriyi Global olarak yükle (Her iterasyonda tekrar okumasın)
GLOBAL_X, GLOBAL_Y = load_data()

# --- 3. MODEL SINIFI (Parametrik) ---
class DynamicPigmentNet(nn.Module):
    def __init__(self, num_pigments, out_dim, hidden_dim, embed_dim, dropout_rate):
        super().__init__()
        self.pigment_embedding = nn.Linear(num_pigments, embed_dim, bias=False)
        
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        embedding_mix = self.pigment_embedding(x) 
        output = self.regressor(embedding_mix)
        return output

def calculate_delta_e_mean(y_true, y_pred):
    # Numpy array olarak gelir
    batch_size = y_true.shape[0]
    y_t_reshaped = y_true.reshape(batch_size, 5, 3)
    y_p_reshaped = y_pred.reshape(batch_size, 5, 3)
    diff = y_t_reshaped - y_p_reshaped
    delta_e_per_angle = np.sqrt(np.sum(diff**2, axis=2))
    return np.mean(delta_e_per_angle)

# --- 4. OBJECTIVE FUNCTION (DE'nin Optimize Edeceği Fonksiyon) ---
def objective_function(params):
    """
    params: [learning_rate, hidden_dim, embed_dim, dropout, batch_size]
    Bu fonksiyon bir eğitim simülasyonu yapar ve en iyi val_delta_e'yi döndürür.
    """
    # 1. Parametreleri Çöz (Unpack)
    lr = params[0]                # Float
    hidden_dim = int(params[1])   # Int
    embed_dim = int(params[2])    # Int
    dropout = params[3]           # Float
    batch_size = int(params[4])   # Int
    
    # 2. Veri Setini Hazırla (Split)
    # Batch Size değişeceği için Loader'ı burada kuruyoruz
    X_train, X_val, y_train, y_val = train_test_split(GLOBAL_X, GLOBAL_Y, test_size=0.15, random_state=42)
    
    # Scaler işlemini her döngüde taze yapalım (Data Leakage olmasın)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_scaled))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val_scaled))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 3. Model Kurulumu
    model = DynamicPigmentNet(
        num_pigments=GLOBAL_X.shape[1],
        out_dim=GLOBAL_Y.shape[1],
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout_rate=dropout
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()
    
    # 4. Hızlı Eğitim Döngüsü
    best_val_de = float('inf')
    
    # Hız kazanmak için scheduler yerine sabit LR kullanıyoruz bu aşamada
    # Veya basit bir decay
    
    for epoch in range(SEARCH_EPOCHS):
        # Train
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        # Val (Sadece her 10 epochta bir kontrol et - hız için)
        if epoch % 10 == 0 or epoch == SEARCH_EPOCHS - 1:
            model.eval()
            all_preds = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(DEVICE)
                    pred = model(bx)
                    all_preds.append(pred.cpu().numpy())
            
            val_preds_scaled = np.vstack(all_preds)
            val_preds_real = y_scaler.inverse_transform(val_preds_scaled)
            
            # Doğrudan Val setinin orijinal değerleriyle kıyasla
            current_de = calculate_delta_e_mean(y_val, val_preds_real)
            
            if current_de < best_val_de:
                best_val_de = current_de
                
    # Ekrana bilgi bas (İlerleme çubuğu gibi)
    print(f"🧬 Deneme: LR={lr:.5f}, Hid={hidden_dim}, Emb={embed_dim}, Drop={dropout:.2f}, Batch={batch_size} -> 📉 DeltaE: {best_val_de:.4f}")
    
    # DE minimize etmeye çalışır, o yüzden DeltaE'yi döndürüyoruz.
    return best_val_de

# --- 5. EVRİMSEL ARAMAYI BAŞLAT ---
def run_optimization():
    print(f"🚀 Hiperparametre Optimizasyonu Başlıyor... (Cihaz: {DEVICE})")
    print("⚠️ Bu işlem uzun sürebilir. Arkana yaslan ve kahve al.")
    
    # Arama Uzayı (Sınırlar)
    # Format: (min, max)
    bounds = [
        (1e-4, 1e-2),   # LR: 0.0001 ile 0.01 arası
        (32, 256),      # Hidden Dim: 32 ile 256 nöron arası
        (8, 48),        # Embed Dim: 8 ile 48 arası
        (0.0, 0.3),     # Dropout: 0 ile 0.3 arası
        (16, 128)       # Batch Size: 16 ile 128 arası
    ]
    
    # Differential Evolution Çalıştır
    # maxiter: Maksimum jenerasyon sayısı
    # popsize: Popülasyon yoğunluğu (Gen sayısı x popsize = Her nesildeki birey sayısı)
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        strategy='best1bin', # En iyi bireyden türet
        maxiter=10,          # Toplam kaç jenerasyon dönsün? (Zaman kısıtlıysa düşür)
        popsize=5,           # Popülasyon büyüklüğü (Artarsa daha iyi bulur ama yavaşlar)
        tol=0.01,            # Sonuçlar birbirine çok yaklaşırsa dur
        disp=True            # İlerlemeyi göster
    )
    
    print("\n🏆 EN İYİ HİPERPARAMETRELER BULUNDU!")
    print(f"Maliyet (Delta E): {result.fun:.4f}")
    print("-" * 30)
    print(f"LR          : {result.x[0]:.6f}")
    print(f"Hidden Dim  : {int(result.x[1])}")
    print(f"Embed Dim   : {int(result.x[2])}")
    print(f"Dropout     : {result.x[3]:.4f}")
    print(f"Batch Size  : {int(result.x[4])}")
    print("-" * 30)
    print("👉 Şimdi bu değerleri 'Config' sınıfına yazıp ana eğitimi (3000 epoch) başlatabilirsin.")

if __name__ == "__main__":
    run_optimization()