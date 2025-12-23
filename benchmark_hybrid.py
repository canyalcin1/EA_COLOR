import pandas as pd
import numpy as np
import torch
import time
from optimizer_engine import EvolutionaryOptimizer
import glob

def run_hybrid_benchmark():
    print("🚀 HİBRİT (Memetic) Benchmark Başlatılıyor...")
    
    # Ayarlar
    TEST_FILE = "eval_dataset_clean.csv"
    
    # Modelden sütun bilgilerini çek (Manuel giriş yapmamak için)
    model_files = glob.glob("models/model_fold_*.pt")
    if not model_files: model_files = glob.glob("models_production/model_fold_*.pt")
    chk = torch.load(model_files[0], map_location="cpu",weights_only=False)
    input_cols = chk['input_cols']
    target_cols = chk['target_cols']
    
    df = pd.read_csv(TEST_FILE)
    samples = df.sample(3) # Rastgele 3 numune
    
    results = []
    
    for idx, row in samples.iterrows():
        sample_name = str(row.get('SampleNo', f"Sample_{idx}"))
        print(f"\n🧪 Numune: {sample_name}")
        
        try:
            target_vals = row[target_cols].astype(str).str.replace(',', '.').astype(float).values
        except: continue

        # 1. ADIM: EVRİMSEL ARAMA (DE) - Kaba Ayar
        start_time = time.time()
        print("   1️⃣  Aşama: Diferansiyel Evrim (DE) çalışıyor...", end="")
        
        optimizer = EvolutionaryOptimizer(target_vals, input_cols, target_cols, device="cpu")
        optimizer.set_constraints(range(len(input_cols)), 5) # Max 5 pigment
        
        pop = optimizer.initialize_population()
        best_de_val = 999.0
        best_recipe = None
        
        # 60 Jenerasyon yeterli (Zaten ince ayar yapacağız)
        for gen in range(60):
            pop, recipe, val = optimizer.step_de(pop)
            if val < best_de_val:
                best_de_val = val
                best_recipe = recipe
        
        print(f" Bitti. DeltaE: {best_de_val:.4f}")
        
        # 2. ADIM: MEMETIC FINE TUNING - İnce Ayar
        print("   2️⃣  Aşama: Memetic (Gradient Descent) çalışıyor...", end="")
        
        # Bulunan en iyi reçeteyi alıp türevle iyileştiriyoruz
        final_recipe, final_de = optimizer.fine_tune(best_recipe, steps=200)
        
        duration = time.time() - start_time
        print(f" Bitti! 🏆 Final DeltaE: {final_de:.4f} (Süre: {duration:.2f}s)")
        
        results.append({
            "Sample": sample_name,
            "DE_DeltaE": best_de_val,
            "Hybrid_Final_DeltaE": final_de,
            "Improvement": best_de_val - final_de,
            "Time": duration
        })
        
    print("\n📊 SONUÇ ÖZETİ")
    print(pd.DataFrame(results)[["Sample", "DE_DeltaE", "Hybrid_Final_DeltaE"]])

if __name__ == "__main__":
    run_hybrid_benchmark()