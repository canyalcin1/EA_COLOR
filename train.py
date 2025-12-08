import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- OPTUNA'DAN GELEN ALTIN PARAMETRELER ---
# {'embed_dim': 92, 'n_layers': 2, 'hidden_dim': 511, 'dropout': 0.116, 'batchnorm': False, 'lr': 0.001, 'batch_size': 32}

class Config:
    CSV_PATH = "RS400_Clean.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Hiperparametreleri
    EMBED_DIM = 92
    HIDDEN_DIM = 511
    DROPOUT = 0.116457  # Yaklaşık %11.6
    N_LAYERS = 2        # 2 Gizli Katman
    USE_BATCHNORM = False
    
    # Eğitim Ayarları
    LR = 0.001023       # ~1e-3
    EPOCHS = 3000       # İdeal sürede dursun
    BATCH_SIZE = 32     # Optuna 32 dedi
    PATIENCE = 300      # Sabır

    # HEDEF VE GİRİŞ SÜTUNLARI
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

# --- VERİ HAZIRLIĞI ---
def load_and_prep_data():
    print(f"📂 Veri Yükleniyor: {Config.CSV_PATH}")
    df = pd.read_csv(Config.CSV_PATH, sep=None, engine='python')

    cols_to_fix = Config.INPUT_COLS + Config.TARGET_COLS
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[Config.INPUT_COLS] = df[Config.INPUT_COLS].fillna(0.0)
    df = df.dropna(subset=Config.TARGET_COLS).reset_index(drop=True)

    # Veri Tekilleştirme (Noise Reduction)
    print("🧹 Veri Tekilleştiriliyor...")
    df['signature'] = df[Config.INPUT_COLS].apply(lambda row: '_'.join(row.values.round(4).astype(str)), axis=1)
    df = df.groupby('signature', as_index=False).mean(numeric_only=True)
    if 'signature' in df.columns: df = df.drop(columns=['signature'])

    # Normalizasyon
    X_raw = df[Config.INPUT_COLS].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 
    X = X_raw / row_sums

    y = df[Config.TARGET_COLS].values.astype(np.float32)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    return X, y_scaled, y_scaler

# --- OPTUNA YAPILI MODEL ---
class PigmentColorNet(nn.Module):
    def __init__(self, num_pigments, out_dim):
        super().__init__()
        
        self.pigment_embedding = nn.Linear(num_pigments, Config.EMBED_DIM, bias=False)
        
        layers = []
        input_size = Config.EMBED_DIM
        
        # Optuna'nın istediği katman sayısı kadar döngü
        for _ in range(Config.N_LAYERS):
            layers.append(nn.Linear(input_size, Config.HIDDEN_DIM))
            if Config.USE_BATCHNORM:
                layers.append(nn.BatchNorm1d(Config.HIDDEN_DIM))
            
            layers.append(nn.SiLU())
            
            if Config.DROPOUT > 0:
                layers.append(nn.Dropout(Config.DROPOUT))
                
            input_size = Config.HIDDEN_DIM
            
        # Çıkış Katmanı
        layers.append(nn.Linear(input_size, out_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        emb = self.pigment_embedding(x)
        out = self.net(emb)
        return out

def calculate_delta_e_mean(y_true, y_pred):
    batch_size = y_true.shape[0]
    y_t_reshaped = y_true.reshape(batch_size, 5, 3)
    y_p_reshaped = y_pred.reshape(batch_size, 5, 3)
    diff = y_t_reshaped - y_p_reshaped
    return np.mean(np.sqrt(np.sum(diff**2, axis=2)))

def main():
    device = torch.device(Config.DEVICE)
    print(f"🚀 FİNAL EĞİTİM BAŞLIYOR... Cihaz: {device}")
    print(f"⚙️ Ayarlar: Layers={Config.N_LAYERS}, Hidden={Config.HIDDEN_DIM}, BN={Config.USE_BATCHNORM}, Drop={Config.DROPOUT:.4f}")

    X, y, y_scaler = load_and_prep_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = PigmentColorNet(X.shape[1], y.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    best_val_de = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
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
        
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(by.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        val_preds_real = y_scaler.inverse_transform(np.vstack(all_preds))
        val_targets_real = y_scaler.inverse_transform(np.vstack(all_targets))
        current_delta_e = calculate_delta_e_mean(val_targets_real, val_preds_real)

        if current_delta_e < best_val_de:
            best_val_de = current_delta_e
            patience_counter = 0
            checkpoint = {
                'model_state': model.state_dict(),
                'scaler': y_scaler,
                'config': {
                    'EMBED_DIM': Config.EMBED_DIM,
                    'HIDDEN_DIM': Config.HIDDEN_DIM,
                    'N_LAYERS': Config.N_LAYERS,
                    'USE_BATCHNORM': Config.USE_BATCHNORM,
                    'DROPOUT': Config.DROPOUT
                },
                'input_cols': Config.INPUT_COLS,
                'target_cols': Config.TARGET_COLS
            }
            torch.save(checkpoint, "FinalModel.pt")
            best_msg = "💾"
        else:
            patience_counter += 1
            best_msg = ""
            
        if epoch % 50 == 0:
            print(f"Ep {epoch:04d} | TrLoss: {avg_train_loss:.5f} | ValLoss: {avg_val_loss:.5f} | ΔE: {current_delta_e:.3f} {best_msg}")
            
        if patience_counter >= Config.PATIENCE:
            print("⏹️ Sabır taştı.")
            break

    print(f"\n🏆 En İyi Delta E: {best_val_de:.3f}")

if __name__ == "__main__":
    main()