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

# Configuration for hyperparameter search
CSV_PATH = "RS400_Clean.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEARCH_EPOCHS = 300  # Reduced for faster hyperparameter search
PATIENCE = 50        # Early stopping during search

# Target and input columns
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

# Load data once and cache in memory
def load_data():
    print(f"Loading and caching data...")
    df = pd.read_csv(CSV_PATH, sep=None, engine='python')

    # Data preprocessing
    cols_to_fix = INPUT_COLS + TARGET_COLS
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[INPUT_COLS] = df[INPUT_COLS].fillna(0.0)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)

    # Row normalization
    X_raw = df[INPUT_COLS].values.astype(np.float32)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 
    X = X_raw / row_sums
    y = df[TARGET_COLS].values.astype(np.float32)

    return X, y

# Load data globally to avoid repeated I/O
GLOBAL_X, GLOBAL_Y = load_data()

# Parameterized neural network model
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
    """Calculate mean Delta E across all samples and angles"""
    batch_size = y_true.shape[0]
    y_t_reshaped = y_true.reshape(batch_size, 5, 3)  # 5 angles, 3 channels (L*a*b*)
    y_p_reshaped = y_pred.reshape(batch_size, 5, 3)
    diff = y_t_reshaped - y_p_reshaped
    delta_e_per_angle = np.sqrt(np.sum(diff**2, axis=2))
    return np.mean(delta_e_per_angle)

# Objective function for differential evolution
def objective_function(params):
    """
    params: [learning_rate, hidden_dim, embed_dim, dropout, batch_size]
    Trains a model with given hyperparameters and returns validation Delta E.
    """
    # Unpack parameters
    lr = params[0]                # Float
    hidden_dim = int(params[1])   # Int
    embed_dim = int(params[2])    # Int
    dropout = params[3]           # Float
    batch_size = int(params[4])   # Int

    # Prepare data split
    X_train, X_val, y_train, y_val = train_test_split(GLOBAL_X, GLOBAL_Y, test_size=0.15, random_state=42)

    # Fresh scaler for each trial to avoid data leakage
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_scaled))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val_scaled))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model setup
    model = DynamicPigmentNet(
        num_pigments=GLOBAL_X.shape[1],
        out_dim=GLOBAL_Y.shape[1],
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout_rate=dropout
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()  # More robust to outliers than MSE

    # Training loop
    best_val_de = float('inf')

    for epoch in range(SEARCH_EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

        # Validate every 10 epochs for speed
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

            current_de = calculate_delta_e_mean(y_val, val_preds_real)

            if current_de < best_val_de:
                best_val_de = current_de

    print(f"Trial: LR={lr:.5f}, Hidden={hidden_dim}, Embed={embed_dim}, Dropout={dropout:.2f}, Batch={batch_size} -> Delta E: {best_val_de:.4f}")

    # Return Delta E for minimization
    return best_val_de

# Run hyperparameter optimization
def run_optimization():
    print(f"Starting hyperparameter optimization (Device: {DEVICE})")

    # Search space bounds (min, max)
    bounds = [
        (1e-4, 1e-2),   # Learning rate
        (32, 256),      # Hidden dimension
        (8, 48),        # Embedding dimension
        (0.0, 0.3),     # Dropout rate
        (16, 128)       # Batch size
    ]

    # Run differential evolution
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        strategy='best1bin',  # Derive from best individual
        maxiter=10,           # Total number of generations
        popsize=5,            # Population size (higher is better but slower)
        tol=0.01,             # Stop if results converge
        disp=True             # Show progress
    )

    print("\nBest hyperparameters found:")
    print(f"Validation Delta E: {result.fun:.4f}")
    print("-" * 30)
    print(f"Learning Rate : {result.x[0]:.6f}")
    print(f"Hidden Dim    : {int(result.x[1])}")
    print(f"Embed Dim     : {int(result.x[2])}")
    print(f"Dropout       : {result.x[3]:.4f}")
    print(f"Batch Size    : {int(result.x[4])}")
    print("-" * 30)

if __name__ == "__main__":
    run_optimization()