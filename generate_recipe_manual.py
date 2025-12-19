import pandas as pd
import numpy as np
from ga_engine import GeneticOptimizer
import torch

def main():
    # 1. Config Dosyasından Meta Verileri Çek (Herhangi bir model dosyasından)
    import glob
    model_files = glob.glob("models/model_fold_*.pt")
    if not model_files: model_files = glob.glob("models_production/model_fold_*.pt")
    
    chk = torch.load(model_files[0], map_location='cpu')
    INPUT_COLS = chk['input_cols']
    TARGET_COLS = chk['target_cols']
    
    # 2. Test Verisi
    print("🧪 Test Verisi Yükleniyor...")
    df_test = pd.read_csv("eval_dataset_clean.csv", sep=None, engine='python')
    # Temizlik
    for c in TARGET_COLS:
        if df_test[c].dtype == object: 
            df_test[c] = pd.to_numeric(df_test[c].str.replace(',', '.'), errors='coerce')
    
    # Örnek Bir Hedef Seç (Row 25)
    sample_idx = 25
    target_values = df_test[TARGET_COLS].iloc[sample_idx].values.astype(np.float32)
    print(f"🎯 Hedef Renk (Sample {sample_idx}): {target_values[:3]}...")

    # 3. GA Motorunu Başlat
    optimizer = GeneticOptimizer(
        target_lab=target_values, 
        input_cols=INPUT_COLS, 
        target_cols=TARGET_COLS
    )
    
    # 4. Çalıştır
    print("\n🧬 Genetik Algoritma Başlıyor...")
    best_recipe, best_de = optimizer.evolve(generations=150)
    
    # 5. Sonuçları Göster
    print(f"\n🏆 SONUÇ (Delta E: {best_de:.4f})")
    print("-" * 40)
    print(f"{'Pigment':<10} | {'Oran (%)'}")
    print("-" * 40)
    
    total_ratio = 0
    for i, val in enumerate(best_recipe):
        if val > 0.001: # %0.1'den küçükleri gösterme
            print(f"{INPUT_COLS[i]:<10} | {val*100:.2f}%")
            total_ratio += val
            
    print("-" * 40)
    print(f"Toplam: {total_ratio*100:.2f}%")

if __name__ == "__main__":
    main()