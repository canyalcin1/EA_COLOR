import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os

class Config:
    CSV_PATH = "RS400_Clean.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optimal hyperparameters from Optuna search
    EMBED_DIM = 92
    HIDDEN_DIM = 511
    DROPOUT = 0.116457
    N_LAYERS = 2
    USE_BATCHNORM = False

    LR = 0.001023
    EPOCHS = 2000
    BATCH_SIZE = 32
    N_SPLITS = 5

    ANGLES = ['15', '25', '45', '75', '110']
    TARGET_COLS = [f"{ang}{ch}" for ang in ANGLES for ch in ['L', 'a', 'b']]

    # 62 pigment feature columns
    INPUT_COLS = [
        "726C", "718Y", "783S", "772S", "755C", "744S", "717C", "980", "910", "135", 
        "828", "835", "831", "826", "836", "838", "815", "856", "892", "895", "855", 
        "851", "858", "839", "896", "862", "865", "891", "898", "845", "832", "816", 
        "818", "233", "940", "424", "356", "384", "306", "318", "331", "198", "154", 
        "290", "522", "322", "570", "410", "550", "530", "371", "482", "343", "030", 
        "960", "580", "670", "632", "110", "821", "190"
    ]

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

def load_data():
    print(f"Loading data: {Config.CSV_PATH}")
    df = pd.read_csv(Config.CSV_PATH, sep=None, engine='python')

    # Convert comma decimals to dots and handle missing values
    cols = Config.INPUT_COLS + Config.TARGET_COLS
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].str.replace(',', '.', regex=False), errors='coerce')
    
    df[Config.INPUT_COLS] = df[Config.INPUT_COLS].fillna(0.0)
    df = df.dropna(subset=Config.TARGET_COLS).reset_index(drop=True)

    # Remove duplicate recipes by averaging
    df['sig'] = df[Config.INPUT_COLS].apply(lambda r: '_'.join(r.values.round(4).astype(str)), axis=1)
    df = df.groupby('sig', as_index=False).mean(numeric_only=True).drop(columns=['sig'], errors='ignore')
    
    X = df[Config.INPUT_COLS].values.astype(np.float32)
    # Normalize each row to sum to 1.0
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1.0
    X = X/row_sums
    y = df[Config.TARGET_COLS].values.astype(np.float32)
    
    return X, y

def main():
    if not os.path.exists("models_production"): os.makedirs("models_production")
    X, y = load_data()
    
    kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=42)
    print(f"Training {Config.N_SPLITS}-fold ensemble...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{Config.N_SPLITS} ---")
        
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y[train_idx])
        y_val_scaled = y_scaler.transform(y[val_idx])
        
        train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y_train_scaled))
        val_ds = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y_val_scaled))
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE*4, shuffle=False)  # Larger batch for faster validation

        model = PigmentColorNet(X.shape[1], y.shape[1]).to(Config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(Config.EPOCHS):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(Config.DEVICE), by.to(Config.DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:  # Validate every 10 epochs
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(Config.DEVICE), by.to(Config.DEVICE)
                        val_loss += criterion(model(bx), by).item()
                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    torch.save({
                        'model_state': model.state_dict(),
                        'scaler': y_scaler,  # Save scaler for inverse transform during inference
                        'config': {
                            'EMBED_DIM': Config.EMBED_DIM, 'HIDDEN_DIM': Config.HIDDEN_DIM,
                            'N_LAYERS': Config.N_LAYERS, 'USE_BATCHNORM': Config.USE_BATCHNORM,
                            'DROPOUT': Config.DROPOUT
                        },
                        'input_cols': Config.INPUT_COLS,
                        'target_cols': Config.TARGET_COLS
                    }, f"models_production/model_fold_{fold}.pt")
                else:
                    patience += 1
                
                if patience >= 50:  # Stop if no improvement for 500 epochs (50*10)
                    print(f"Early stopping at epoch {epoch} - Best loss: {best_val_loss:.5f}")
                    break

        print(f"Model {fold+1} completed.")

if __name__ == "__main__":
    main()