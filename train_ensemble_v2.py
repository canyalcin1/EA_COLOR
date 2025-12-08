import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os

# Chroma based weightened

# --- AYARLAR ---
class Config:
    CSV_PATH = "RS400_Clean.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optuna Altın Ayarları (V1 Mimarisi)
    EMBED_DIM = 92
    HIDDEN_DIM = 511
    DROPOUT = 0.116457
    N_LAYERS = 2
    USE_BATCHNORM = False
    
    LR = 0.001023
    EPOCHS = 2500      
    BATCH_SIZE = 32
    N_SPLITS = 5       # 5 Model (K-Fold)

    ANGLES = ['15', '25', '45', '75', '110']
    TARGET_COLS = [f"{ang}{ch}" for ang in ANGLES for ch in ['L', 'a', 'b']]
    
    # --- TEMİZ GİRİŞ LİSTESİ (SADECE PİGMENTLER) ---
    # Si, Sa, G gibi ölçümleri sildim, sadece reçete bileşenleri kaldı.
    INPUT_COLS = [
        "726C", "718Y", "783S", "772S", "755C", "744S", "717C", "980", "910", "135", 
        "828", "835", "831", "826", "836", "838", "815", "856", "892", "895", "855", 
        "851", "858", "839", "896", "862", "865", "891", "898", "845", "832", "816", 
        "818", "233", "940", "424", "356", "384", "306", "318", "331", "198", "154", 
        "290", "522", "322", "570", "410", "550", "530", "371", "482", "343", "030", 
        "960", "580", "670", "632", "110", "821", "190"
    ]

# --- MODEL (STANDART V1) ---
class PigmentColorNet(nn.Module):
    def __init__(self, num_pigments, out_dim):
        super().__init__()
        self.pigment_embedding = nn.Linear(num_pigments, Config.EMBED_DIM, bias=False)
        layers = []
        input_size = Config.EMBED_DIM
        for _ in range(Config.N_LAYERS):
            layers.append(nn.Linear(input_size, Config.HIDDEN_DIM))
            if Config.USE_BATCHNORM: layers.append(nn.BatchNorm1d(Config.HIDDEN_DIM))
            layers.append(nn.SiLU())
            if Config.DROPOUT > 0: layers.append(nn.Dropout(Config.DROPOUT))
            input_size = Config.HIDDEN_DIM
        layers.append(nn.Linear(input_size, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(self.pigment_embedding(x))

# --- AKILLI CEZA SİSTEMİ (CHROMA & FLOP WEIGHTED LOSS) ---
class SmartWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none') # Hataları tek tek al
        
    def forward(self, pred_scaled, target_scaled):
        """
        Analiz sonucuna göre:
        1. Canlı renklerde (High Chroma) hata yaparsa daha çok kız.
        2. Metalik efektlerde (High Flop) hata yaparsa biraz daha kız.
        """
        # Standart Hata
        raw_loss = self.mse(pred_scaled, target_scaled)
        
        with torch.no_grad():
            # Veriyi [Batch, 5, 3] formatına getir (L, a, b)
            t_reshaped = target_scaled.view(-1, 5, 3)
            
            # --- 1. KROMA (CANLILIK) FAKTÖRÜ ---
            # Scaled uzayda a ve b'nin büyüklüğü canlılığı gösterir
            # a (kanal 1) ve b (kanal 2)
            chroma = torch.sqrt(t_reshaped[:, :, 1]**2 + t_reshaped[:, :, 2]**2).mean(dim=1)
            
            # Kroma Ağırlığı: Gri ise 1.0, Çok canlıysa 4.0'a kadar çıksın
            # Çarpanı 3.0 yaptık çünkü analizde a/b farkı çok büyüktü.
            chroma_weight = 1.0 + (chroma * 3.0)
            
            # --- 2. FLOP (METALİK) FAKTÖRÜ ---
            # 15 derece L (Parlak) ile 110 derece L (Karanlık) arasındaki fark metalikliği gösterir.
            # L kanalı (kanal 0)
            # t_reshaped[:, 0, 0] -> 15 derece L
            # t_reshaped[:, 4, 0] -> 110 derece L
            # Scaled uzayda farklarına bakıyoruz
            flop_index = torch.abs(t_reshaped[:, 0, 0] - t_reshaped[:, 4, 0])
            
            # Flop Ağırlığı: Düz renkse 1.0, Metalikse 2.0'ye kadar çıksın
            flop_weight = 1.0 + (flop_index * 0.5)
            
            # Toplam Ağırlık (İkisinin kombinasyonu)
            total_weight = chroma_weight * flop_weight
            
            # Boyut düzeltme [Batch] -> [Batch, 1]
            total_weight = total_weight.view(-1, 1)
        
        # Cezayı uygula
        weighted_loss = raw_loss * total_weight
        return weighted_loss.mean()

def load_data():
    print(f"📂 Veri Okunuyor: {Config.CSV_PATH}")
    try:
        df = pd.read_csv(Config.CSV_PATH, sep=None, engine='python')
    except Exception as e:
        print(f"Hata: {e}")
        return None, None
        
    cols = Config.INPUT_COLS + Config.TARGET_COLS
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].str.replace(',', '.', regex=False), errors='coerce')
            
    df[Config.INPUT_COLS] = df[Config.INPUT_COLS].fillna(0.0)
    df = df.dropna(subset=Config.TARGET_COLS).reset_index(drop=True)
    
    # Tekilleştirme
    df['sig'] = df[Config.INPUT_COLS].apply(lambda r: '_'.join(r.values.round(4).astype(str)), axis=1)
    df = df.groupby('sig', as_index=False).mean(numeric_only=True).drop(columns=['sig'], errors='ignore')
    
    # V1 STRATEJİSİ: HAM AĞIRLIK (Density Yok)
    X = df[Config.INPUT_COLS].values.astype(np.float32)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1.0
    X = X/row_sums
    
    y = df[Config.TARGET_COLS].values.astype(np.float32)
    
    return X, y

def main():
    if not os.path.exists("models_final"): os.makedirs("models_final")
    X, y = load_data()
    if X is None: return
    
    kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=42)
    
    print(f"🚀 SMART WEIGHTED ENSEMBLE EĞİTİM ({Config.N_SPLITS} Model)...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- 🔴 FOLD {fold+1}/{Config.N_SPLITS} (Canlı Renk & Flop Odaklı) ---")
        
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y[train_idx])
        y_val_scaled = y_scaler.transform(y[val_idx])
        
        train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y_train_scaled))
        val_ds = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y_val_scaled))
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE*2, shuffle=False)
        
        model = PigmentColorNet(X.shape[1], y.shape[1]).to(Config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-5)
        
        # YENİ LOSS FONKSİYONU
        criterion = SmartWeightedLoss()
        
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(Config.DEVICE), by.to(Config.DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by) # Weighted Loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation (Standart MSE ile kontrol edelim ki gerçek performansı görelim)
            model.eval()
            val_loss = 0
            simple_mse = nn.MSELoss()
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(Config.DEVICE), by.to(Config.DEVICE)
                    val_loss += simple_mse(model(bx), by).item() # Burada ağırlıksız bakıyoruz
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                
                # Modeli Kaydet
                torch.save({
                    'model_state': model.state_dict(),
                    'scaler': y_scaler,
                    'input_cols': Config.INPUT_COLS,
                    'target_cols': Config.TARGET_COLS,
                    'config': {
                        'EMBED_DIM': Config.EMBED_DIM, 'HIDDEN_DIM': Config.HIDDEN_DIM,
                        'N_LAYERS': Config.N_LAYERS, 'USE_BATCHNORM': Config.USE_BATCHNORM,
                        'DROPOUT': Config.DROPOUT
                    }
                }, f"models_final/model_fold_{fold}.pt")
            else:
                patience += 1
            
            if patience >= 150:
                print(f"⏹️ Erken Durdurma (Ep {epoch}) - Best Val MSE: {best_val_loss:.5f}")
                break
                
            if epoch % 100 == 0:
                print(f"Ep {epoch} | Val MSE: {val_loss:.5f}")
        
        print(f"✅ Model Kaydedildi: models_final/model_fold_{fold}.pt")

if __name__ == "__main__":
    main()