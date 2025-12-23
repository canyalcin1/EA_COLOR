import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import time

# --- 1. SİSTEM AYARLARI ---
class SystemConfig:
    MODEL_PATH = "3DeltaEModel.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. DİNAMİK MODEL SINIFI ---
class PigmentColorNet(nn.Module):
    def __init__(self, num_pigments, out_dim, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.pigment_embedding = nn.Linear(num_pigments, embed_dim, bias=False)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        embedding_mix = self.pigment_embedding(x) 
        output = self.regressor(embedding_mix)
        return output

# --- 3. MODELİ YÜKLE ---
def load_trained_model():
    print(f"⚙️ Model Yükleniyor... ({SystemConfig.DEVICE})")
    try:
        checkpoint = torch.load(SystemConfig.MODEL_PATH, map_location=SystemConfig.DEVICE, weights_only=False)
    except FileNotFoundError:
        print("❌ HATA: Model dosyası bulunamadı! Lütfen önce 'train.py' dosyasını çalıştır.")
        exit()

    # Config'i dosyadan oku
    saved_config = checkpoint.get('config', {})
    
    # Varsayılan değerler (eğer config yoksa)
    embed_dim = saved_config.get('embed_dim', 24)
    hidden_dim = saved_config.get('hidden_dim', 128)
    dropout = saved_config.get('dropout', 0.0) 

    input_cols = checkpoint['input_cols']
    target_cols = checkpoint['target_cols']
    scaler = checkpoint['scaler']
    
    print(f"🧠 Model Yapısı: Embed={embed_dim}, Hidden={hidden_dim}, Dropout={dropout}")

    model = PigmentColorNet(
        num_pigments=len(input_cols), 
        out_dim=len(target_cols),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(SystemConfig.DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, scaler, input_cols, target_cols

# Global Nesneler (Modeli sadece 1 kere yüklüyoruz)
MODEL, SCALER, INPUT_COLS, TARGET_COLS = load_trained_model()

# --- 4. OPTİMİZASYON FONKSİYONU ---
def objective_function(recipe_candidate, target_lab_values):
    """
    args:
        recipe_candidate: DE'nin ürettiği pigment oranları
        target_lab_values: Hedef renk (argument olarak alıyoruz, global değil)
    """
    # Negatifleri sıfırla
    recipe_candidate = np.maximum(recipe_candidate, 0)
    
    # Toplamı 1 yap (Normalizasyon)
    total_sum = np.sum(recipe_candidate)
    if total_sum == 0: return 9999.0
    recipe_norm = recipe_candidate / total_sum
    
    # Tahmin (Model GPU'da çalışır)
    with torch.no_grad():
        input_tensor = torch.tensor(recipe_norm, dtype=torch.float32).unsqueeze(0).to(SystemConfig.DEVICE)
        pred_scaled = MODEL(input_tensor).cpu().numpy()
    
    pred_real = SCALER.inverse_transform(pred_scaled)
    
    # Delta E Hesapla
    # target_lab_values shape: [15] -> reshape -> [1, 5, 3]
    diff = target_lab_values.reshape(1, 5, 3) - pred_real.reshape(1, 5, 3)
    mean_delta_e = np.mean(np.sqrt(np.sum(diff**2, axis=2)))
    
    # Ceza (Sparsity Penalty): Çok fazla çeşit pigment kullanmasın
    # Örneğin: %0.1'den (binde bir) az olanları sayma
    active_pigments = np.sum(recipe_norm > 0.001) 
    sparsity_penalty = 0.0
    
    # Eğer 6'dan fazla pigment kullanıyorsa ceza yaz (Sanayi pratiği)
    if active_pigments > 6:
        sparsity_penalty = (active_pigments - 6) * 0.05
    
    return mean_delta_e + sparsity_penalty

# --- 5. REÇETE BULUCU ---
def find_recipe(target_lab):
    print(f"\n🧪 Reçete Aranıyor... Hedef: {target_lab[0:3]}...")
    
    bounds = [(0, 1)] * len(INPUT_COLS)
    
    # Differential Evolution Çalıştır
    # DÜZELTME: workers=1 yapıldı ve args parametresi eklendi.
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(target_lab,), # Hedef rengi buraya ekliyoruz
        strategy='best1bin',
        maxiter=1000,
        popsize=15, 
        tol=0.01,
        workers=1, # <--- KRİTİK DÜZELTME: Sadece 1 işlemci kullan (Çünkü GPU kullanıyoruz)
        disp=True  # İlerlemeyi göster
    )
    
    best_recipe = np.maximum(result.x, 0)
    best_recipe /= np.sum(best_recipe)
    
    print(f"✅ Bulundu! Optimize Delta E: {result.fun:.4f}")
    return best_recipe

if __name__ == "__main__":
    # TEST SENARYOSU
    df_test = pd.read_csv("RS400_Clean.csv", sep=None, engine='python')
    
    # Temizlik
    for c in TARGET_COLS + INPUT_COLS:
        if df_test[c].dtype == object:
            df_test[c] = pd.to_numeric(df_test[c].str.replace(',', '.'), errors='coerce')
    df_test = df_test.fillna(0.0)

    # Örnek: Rastgele bir satırın reçetesini bulmaya çalış
    sample_idx = 20 # Değiştirebilirsin
    print(f"📌 Örnek No: {sample_idx}")
    
    real_target = df_test[TARGET_COLS].iloc[sample_idx].values.astype(np.float32)
    real_recipe = df_test[INPUT_COLS].iloc[sample_idx].values.astype(np.float32)
    
    # Gerçek reçeteyi normalize et (kıyaslama için)
    if real_recipe.sum() > 0:
        real_recipe /= real_recipe.sum()

    # REÇETE BUL
    found_recipe = find_recipe(real_target)
    
    print("\n📊 SONUÇ KARŞILAŞTIRMA (Sadece kullanılanlar):")
    print(f"{'Pigment':<10} | {'Gerçek':<10} | {'Yapay Zeka':<10} | {'Fark'}")
    print("-" * 50)
    
    total_found_pigment = 0
    for i, col in enumerate(INPUT_COLS):
        r_val = real_recipe[i]
        f_val = found_recipe[i]
        
        # Sadece %1'in üzerindekileri veya gerçekte olanları göster
        if r_val > 0.005 or f_val > 0.005:
            print(f"{col:<10} | {r_val:.4f}     | {f_val:.4f}     | {abs(r_val-f_val):.4f}")