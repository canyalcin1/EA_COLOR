import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- CONFIG (SABİTLER) ---
CSV_PATH = "RS400_Clean.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS_PER_TRIAL = 200  # Her deneme ne kadar sürsün? (Hızlı eleme için kısa tutuyoruz)

# --- VERİ HAZIRLIĞI (GLOBAL) ---
def get_data():
    df = pd.read_csv(CSV_PATH, sep=None, engine='python')
    
    # Sütun İsimleri (Senin Config'den alındı)
    input_cols = [
        "726C", "718Y", "783S", "772S", "755C", "744S", "717C", "980", "910", "135", 
        "828", "835", "831", "826", "836", "838", "815", "856", "892", "895", "855", 
        "851", "858", "839", "896", "862", "865", "891", "898", "845", "832", "816", 
        "818", "233", "940", "424", "356", "384", "306", "318", "331", "198", "154", 
        "290", "522", "322", "570", "410", "550", "530", "371", "482", "343", "030", 
        "960", "580", "670", "632", "110", "821", "190"
    ]
    target_cols = [f"{ang}{ch}" for ang in ['15', '25', '45', '75', '110'] for ch in ['L', 'a', 'b']]
    
    # Temizlik
    cols_to_fix = input_cols + target_cols
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[input_cols] = df[input_cols].fillna(0.0)
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    
    # Tekilleştirme (Mean Grouping)
    df['signature'] = df[input_cols].apply(lambda row: '_'.join(row.values.round(4).astype(str)), axis=1)
    df = df.groupby('signature', as_index=False).mean(numeric_only=True)
    if 'signature' in df.columns: df = df.drop(columns=['signature'])

    # Norm
    X_raw = df[input_cols].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 
    X = X_raw / row_sums
    y = df[target_cols].values.astype(np.float32)

    return X, y, input_cols, target_cols

# Veriyi bir kere yükle
X_ALL, y_ALL, IN_COLS, TGT_COLS = get_data()
y_scaler = StandardScaler()
y_scaled_ALL = y_scaler.fit_transform(y_ALL)

# --- DİNAMİK MODEL (Optuna için) ---
class DynamicNet(nn.Module):
    def __init__(self, trial, in_dim, out_dim):
        super().__init__()
        
        # 1. Hiperparametreleri Optuna Belirlesin
        embed_dim = trial.suggest_int("embed_dim", 16, 128)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
        dropout_rate = trial.suggest_float("dropout", 0.0, 0.4)
        use_batchnorm = trial.suggest_categorical("batchnorm", [True, False])

        self.pigment_embedding = nn.Linear(in_dim, embed_dim, bias=False)
        
        layers = []
        input_size = embed_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(input_size, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU()) # SiLU (Swish) genelde ReLU'dan iyidir
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_dim
            
        layers.append(nn.Linear(input_size, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pigment_embedding(x)
        return self.net(x)

# --- OPTUNA HEDEF FONKSİYONU ---
def objective(trial):
    # Train/Val Split (Her denemede aynı random_state ile sabit tutuyoruz)
    X_train, X_val, y_train, y_val = train_test_split(X_ALL, y_scaled_ALL, test_size=0.15, random_state=42)
    
    # Tensorlar
    t_X_train = torch.tensor(X_train).to(DEVICE)
    t_y_train = torch.tensor(y_train).to(DEVICE)
    t_X_val = torch.tensor(X_val).to(DEVICE)
    t_y_val = torch.tensor(y_val).to(DEVICE)
    
    # Model Kur
    model = DynamicNet(trial, X_ALL.shape[1], y_ALL.shape[1]).to(DEVICE)
    
    # Optimizer Ayarları (Learning Rate'i de Optuna seçsin)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) # Hafif bir fren her zaman iyidir
    criterion = nn.MSELoss()
    
    # Küçük Eğitim Döngüsü
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_batches = len(X_train) // batch_size
    
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        # Basit batching (Dataloader yerine manuel shuffle hız kazandırır)
        permutation = torch.randperm(t_X_train.size()[0])
        
        for i in range(0, t_X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = t_X_train[indices], t_y_train[indices]
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation ve Pruning (Kötü gidenleri erken öldür)
        model.eval()
        with torch.no_grad():
            val_pred = model(t_X_val)
            # Delta E'ye benzer bir skor kullanmak en doğrusu ama MSE de yeterli.
            # Biz MSE üzerinden gidelim, çünkü Delta E hesaplamak işlem yükü bindirir.
            val_loss = criterion(val_pred, t_y_val).item()
            
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

# --- ANA ÇALIŞTIRICI ---
if __name__ == "__main__":
    print("🧠 Optuna ile En İyi Model Aranıyor... (Bu işlem 10-20 dk sürebilir)")
    
    # Pruner: Başarısız denemeleri yarıda kesen algoritma
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50) # 50 farklı model dene
    
    print("\n" + "="*50)
    print("🏆 EN İYİ HİPERPARAMETRELER BULUNDU!")
    print("="*50)
    print("Best Params:", study.best_params)
    print("Best Val MSE:", study.best_value)
    
    # En iyi parametreleri dosyaya yaz ki unutmayalım
    with open("best_params.txt", "w") as f:
        f.write(str(study.best_params))
        
    print("✅ Parametreler 'best_params.txt' dosyasına kaydedildi.")
    print("Şimdi bu parametreleri train.py dosyasına girip EĞİTİMİ BAŞLATABİLİRSİN.")