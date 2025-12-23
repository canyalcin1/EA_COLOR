import pandas as pd
import numpy as np
import torch
import time
from optimizer_engine import EvolutionaryOptimizer

def run_benchmark():
    # --- AYARLAR ---
    TEST_FILE = "eval_dataset_clean.csv"
    MODELS_DIR = "models/" # Model dosyaların nerede?
    
    print("🚀 Benchmark Başlatılıyor...")
    
    # 1. Veri Setini Oku
    try:
        df = pd.read_csv(TEST_FILE)
    except Exception as e:
        print(f"Hata: {TEST_FILE} okunamadı. Dosya yolunu kontrol et. ({e})")
        return

    # 2. Optimizer'ı başlatarak sütun isimlerini öğrenelim (Dummy Start)
    # Hedef değerler şimdilik rastgele, sadece sütun adlarını çekmek için başlatıyoruz
    dummy_target = np.zeros(15) 
    # Optimizer sınıfını başlatınca modelleri yüklerken input_cols'u öğreneceğiz
    # Ancak burada senin `optimizer_engine` sınıfına küçük bir ekleme yapmadan
    # Manuel olarak modelden çekelim.
    
    model_files = glob.glob(MODELS_DIR + "model_fold_*.pt")
    if not model_files:
        print("❌ Model dosyası bulunamadı! 'train_ensemble.py' çalıştırdın mı?")
        return
        
    chk = torch.load(model_files[0], map_location="cpu", weights_only=False)
    input_cols = chk['input_cols']
    target_cols = chk['target_cols']
    
    print(f"✅ Modeller bulundu. {len(input_cols)} pigment girdisi var.")
    
    # 3. Rastgele 3 Test Numunesi Seç (Benchmark için)
    samples = df.sample(3)
    results = []
    
    strategies = ["GA", "DE", "PSO"]
    
    # --- ANA DÖNGÜ ---
    for idx, row in samples.iterrows():
        sample_name = str(row.get('SampleNo', f"Sample_{idx}"))
        print(f"\n🧪 Numune Analiz Ediliyor: {sample_name}")
        
        # Hedef değerleri al (L,a,b verileri)
        # Verideki virgülleri noktaya çevirip float yapıyoruz
        try:
            target_vals = row[target_cols].astype(str).str.replace(',', '.').astype(float).values
        except KeyError:
            print(f"Sütun hatası: {target_cols} veride bulunamadı.")
            continue
            
        for strat in strategies:
            print(f"   👉 Algoritma: {strat} çalışıyor...", end="")
            start_time = time.time()
            
            # Optimizer Kurulumu
            optimizer = EvolutionaryOptimizer(target_vals, input_cols, target_cols, device="cpu")
            # Kısıtlama: Max 5 pigment
            optimizer.set_constraints(range(len(input_cols)), 5)
            
            pop = optimizer.initialize_population()
            
            if strat == "PSO": 
                optimizer.init_pso()
            
            best_de_history = []
            final_de = 999.0
            
            # EVRİM DÖNGÜSÜ (100 Jenerasyon yeterli)
            for gen in range(100):
                if strat == "GA":
                    pop, best_rec, final_de = optimizer.step_ga(pop)
                elif strat == "DE":
                    pop, best_rec, final_de = optimizer.step_de(pop)
                elif strat == "PSO":
                    pop, best_rec, score = optimizer.step_pso(pop)
                    # PSO score dönüyor (içinde ceza puanı olabilir),
                    # rapor için saf DeltaE'yi tekrar hesaplayalım:
                    _, final_de_tensor = optimizer.calculate_fitness_batch(best_rec.unsqueeze(0))
                    final_de = final_de_tensor.item()
                
                best_de_history.append(final_de)
                
                # Erken durdurma (DeltaE < 0.3 ise çık)
                if final_de < 0.3:
                    break
            
            duration = time.time() - start_time
            print(f" Bitti. Süre: {duration:.2f}s | DeltaE: {final_de:.4f}")
            
            # Pigment Sayısı (Kaç tane aktif pigment var?)
            active_pigments = (best_rec > 0.001).sum().item()
            
            results.append({
                "Sample": sample_name,
                "Algorithm": strat,
                "Final_DeltaE": final_de,
                "Time_Sec": duration,
                "Generations": len(best_de_history),
                "Active_Pigments": active_pigments,
                # Convergence history'yi string olarak kaydet (Rapor grafiği için)
                "History": str(best_de_history)
            })
            
    # Sonuçları Kaydet
    res_df = pd.DataFrame(results)
    res_df.to_csv("benchmark_results_cpu.csv", index=False)
    print("\n✅ Benchmark tamamlandı! Sonuçlar 'benchmark_results_cpu.csv' dosyasına yazıldı.")

if __name__ == "__main__":
    import glob # Import eksik kalmasın
    run_benchmark()