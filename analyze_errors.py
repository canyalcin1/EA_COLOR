import pandas as pd
import numpy as np

# --- AYARLAR ---
# Analiz edilecek dosya (V1 Ensemble Sonucu)
RESULT_CSV = "Ensemble_Sonuc_Eval.csv" 
BAD_THRESHOLD = 3.0 
ANGLES = ['15', '25', '45', '75', '110']

def main():
    print(f"🕵️ DETAYLI Hata Analizi (5 Açı): {RESULT_CSV}")
    try:
        df = pd.read_csv(RESULT_CSV, sep=None, engine='python')
    except Exception as e:
        print(f"Hata: {e}")
        return

    if "Ensemble_DeltaE" not in df.columns:
        print("❌ 'Ensemble_DeltaE' sütunu yok. predict_ensemble.py çalıştırılmalı.")
        return

    # 1. Kötü Tahminleri Ayıkla
    bad_preds = df[df["Ensemble_DeltaE"] > BAD_THRESHOLD].copy()
    good_preds = df[df["Ensemble_DeltaE"] <= BAD_THRESHOLD].copy()
    
    print(f"\n⚠️ KÖTÜ TAHMİN SAYISI: {len(bad_preds)} / {len(df)} (%{len(bad_preds)/len(df)*100:.1f})")
    
    if len(bad_preds) == 0:
        print("Kritik hata yok.")
        return

    # 2. AÇI BAZLI ANALİZ
    print("\n" + "="*60)
    print(f"{'AÇI':<6} | {'RENK':<5} | {'KÖTÜ ORT.':<12} | {'İYİ ORT.':<12} | {'FARK':<10}")
    print("="*60)
    
    for ang in ANGLES:
        for ch in ['L', 'a', 'b']:
            col_name = f"{ang}{ch}"
            if col_name in df.columns:
                avg_bad = bad_preds[col_name].mean()
                avg_good = good_preds[col_name].mean()
                diff = avg_bad - avg_good
                
                # Sadece belirgin farkları işaretleyelim
                marker = "🔴" if abs(diff) > 5.0 else ""
                
                print(f"{ang:<6} | {ch:<5} | {avg_bad:<12.1f} | {avg_good:<12.1f} | {diff:<10.1f} {marker}")
        print("-" * 60)

    # 3. YORUM VE TAVSİYE
    print("\n🧠 HIZLI YORUM:")
    
    # 110 Derece (Flop) Kontrolü
    l110_diff = bad_preds['110L'].mean() - good_preds['110L'].mean()
    if abs(l110_diff) > 10:
        print("👉 FLOP SORUNU: 110 derece (yan açı) parlaklığında ciddi sapma var.")
        print("   Bu, metalik pigmentlerin yönelimini (orientation) modelin tam çözemediğini gösterir.")
    
    # Kroma Kontrolü
    a15_diff = bad_preds['15a'].mean() - good_preds['15a'].mean()
    if a15_diff > 10:
        print("👉 KROMA SORUNU: Canlı renklerde (Yüksek 'a') hata artıyor.")
        print("   Chroma Weighted Loss eğitimi bu sorunu çözecektir.")

if __name__ == "__main__":
    main()