import pandas as pd
import numpy as np
import torch
import time
import glob
from optimizer_engine import EvolutionaryOptimizer

def run_full_benchmark():
    # --- AYARLAR ---
    INPUT_FILE = "eval_dataset_clean.csv"
    OUTPUT_FILE = "Final_Project_Benchmark_Results.csv"
    
    print(f"🚀 TAM KAPSAMLI BENCHMARK BAŞLATILIYOR...")
    print(f"📂 Veri Seti: {INPUT_FILE}")
    
    # 1. Veri ve Model Hazırlığı
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"❌ Hata: Dosya okunamadı! ({e})")
        return

    # Modelden sütun bilgilerini çek
    model_files = glob.glob("models/model_fold_*.pt")
    if not model_files: model_files = glob.glob("models_production/model_fold_*.pt")
    chk = torch.load(model_files[0], map_location="cpu",weights_only=False)
    input_cols = chk['input_cols']
    target_cols = chk['target_cols']
    
    results = []
    
    # İstatistikler
    total_samples = len(df)
    print(f"📊 Toplam Numune Sayısı: {total_samples}")
    print("-" * 60)
    
    # --- ANA DÖNGÜ ---
    for idx, row in df.iterrows():
        sample_id = str(row.get('SampleNo', f"S_{idx}"))
        
        # Hedef değerleri parse et
        try:
            target_vals = row[target_cols].astype(str).str.replace(',', '.').astype(float).values
        except Exception as e:
            print(f"⚠️ {sample_id} atlandı: Veri formatı hatalı.")
            continue
            
        print(f"🧪 [{idx+1}/{total_samples}] {sample_id} işleniyor...", end="", flush=True)
        
        # Optimizer Başlat
        optimizer = EvolutionaryOptimizer(target_vals, input_cols, target_cols, device="cpu")
        optimizer.set_constraints(range(len(input_cols)), 8) # Max 8 pigment
        
        # --- AŞAMA 1: DE (Global Search) ---
        t0 = time.time()
        pop = optimizer.initialize_population()
        
        best_de_val = 999.0
        best_recipe = None
        
        DE_GENS = 50 # Hız için 50 jenerasyon yeterli
        for gen in range(DE_GENS):
            pop, recipe, val = optimizer.step_de(pop)
            if val < best_de_val:
                best_de_val = val
                best_recipe = recipe
                
        t1 = time.time()
        de_time = t1 - t0
        
        # --- AŞAMA 2: MEMETIC FINE-TUNE (Local Search) ---
        # Burayı manuel yapıyoruz ki adım sayısını (iterations) kaydedebilelim
        t2 = time.time()
        
        optimized_recipe = best_recipe.clone().detach().requires_grad_(True)
        ft_optimizer = torch.optim.Adam([optimized_recipe], lr=0.01)
        
        # Kullanılmayan pigmentleri maskele
        with torch.no_grad():
            zero_mask = (best_recipe <= 0.001)
            
        final_de = best_de_val
        stop_step = 0
        MAX_STEPS = 200
        
        for step in range(MAX_STEPS):
            ft_optimizer.zero_grad()
            
            # Tahmin & Loss
            preds = optimizer.predict_batch(optimized_recipe.unsqueeze(0))
            diff = preds.view(-1, 5, 3) - optimizer.target_lab.view(1, 5, 3)
            loss_de = torch.sqrt(torch.sum(diff**2, dim=2)).mean()
            
            # Sparsity Koruması
            loss_sparsity = torch.sum(torch.abs(optimized_recipe) * zero_mask) * 1000.0
            
            total_loss = loss_de + loss_sparsity
            total_loss.backward()
            ft_optimizer.step()
            
            # Constraintleri Uygula
            with torch.no_grad():
                optimized_recipe.data = torch.relu(optimized_recipe.data)
                optimized_recipe.data[zero_mask] = 0.0
                s = optimized_recipe.data.sum()
                if s > 1e-6: optimized_recipe.data /= s
                
                # En iyi değeri güncelle
                current_de = loss_de.item()
                if current_de < final_de:
                    final_de = current_de
                
                # Erken Durdurma (Delta E < 0.1 ise veya değişim durduysa)
                if current_de < 0.1:
                    stop_step = step + 1
                    break
            
            stop_step = step + 1
            
        t3 = time.time()
        hybrid_time = t3 - t2
        
        # --- METRİKLERİ KAYDET ---
        active_pigments = (optimized_recipe.detach() > 0.001).sum().item()
        improvement = best_de_val - final_de
        
        print(f" ✅ | DE: {best_de_val:.2f} -> Hibrit: {final_de:.2f} ({stop_step} adım)")
        
        results.append({
            "Sample_ID": sample_id,
            "DE_Best_DeltaE": round(best_de_val, 4),
            "Hybrid_Final_DeltaE": round(final_de, 4),
            "Improvement": round(improvement, 4),
            "Iterations_To_Converge": stop_step,
            "Active_Pigments": active_pigments,
            "DE_Time_Sec": round(de_time, 2),
            "Hybrid_Time_Sec": round(hybrid_time, 2),
            "Total_Time_Sec": round(de_time + hybrid_time, 2)
        })

    # --- RAPORLAMA ---
    res_df = pd.DataFrame(results)
    
    # Dosyaya Yaz
    res_df.to_csv(OUTPUT_FILE, index=False)
    
    # Konsola Özet Bas
    print("\n" + "="*60)
    print("🏆 PROJE SONUÇ RAPORU ÖZETİ")
    print("="*60)
    print(f"Ortalama Delta E (DE Aşaması)     : {res_df['DE_Best_DeltaE'].mean():.4f}")
    print(f"Ortalama Delta E (Hibrit Sonuç)   : {res_df['Hybrid_Final_DeltaE'].mean():.4f}")
    print(f"Ortalama İyileştirme (Improvement): {res_df['Improvement'].mean():.4f}")
    print(f"Ortalama Süre (Numune Başına)     : {res_df['Total_Time_Sec'].mean():.2f} sn")
    print(f"Başarı Oranı (Delta E < 2.0)      : {(res_df['Hybrid_Final_DeltaE'] < 2.0).mean() * 100:.1f}%")
    print("-" * 60)
    print(f"📄 Detaylı sonuçlar '{OUTPUT_FILE}' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_full_benchmark()