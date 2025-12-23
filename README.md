"# EA_COLOR" 

1. train_production (forward model eğittigimiz kod) 
2. optimize_model.py (Evrimsel ile Forward model parametre optimizasyonu denemesi)
3. tune.py (optuna library ile parametre optimizasyonu yaptıgımız kod)
4. optimizer_engine.py (Tüm Evrimsel Kodlarının Kod blokları)
5. benchmark.py (GA - DE - PSO arasında seçim yapmamızı saglayan kod)
6. benchmark_full.py (Seçilen yöntem (DE) üzerinden memetic ile finetune etme (basarısını artırmaya yönelik))
7. nsga2_engine.py (seçilecek pigment sayısını tahmin etmeye saglayan motor)
8. run_pareto.py (number of pigment seçtiren kod)
9. plot_pareto.py (kıyaslamalı number of pigment plotter)




















(renkenv310) C:\Users\TAC7\Desktop\RenkAI>python predict_ensemble.py
⚙️ 5 Model ile Ensemble Tahmin Yapılıyor...
✅ Model 1 tahmini tamam.
✅ Model 2 tahmini tamam.
✅ Model 3 tahmini tamam.
✅ Model 4 tahmini tamam.
✅ Model 5 tahmini tamam.
========================================
📊 ENSEMBLE ORTALAMA DELTA E: 2.715
========================================
✅ Sonuçlar kaydedildi: Ensemble_Sonuc.csv

(renkenv310) C:\Users\TAC7\Desktop\RenkAI>


En iyi sonuçlar train_ensemble.py'dan alınmış olup predict_ensemble.py ile değerlendirilmiştir.






(renkenv310) C:\Users\TAC7\Desktop\RenkAI>python predict_detailed_analysis.py
⚙️ DETAYLI ANALİZ BAŞLIYOR... (5 Model İncelenecek)

==================================================
MODEL           | ORTALAMA ΔE     | DURUM
==================================================
model_fold_0.pt | 3.7591          | ⚠️
model_fold_1.pt | 3.5361          | ⚠️
model_fold_2.pt | 3.9919          | ⚠️
model_fold_3.pt | 3.3296          | ⚠️
model_fold_4.pt | 3.9231          | ⚠️
--------------------------------------------------
ENSEMBLE (ORT)  | 3.0772          | 🌟
==================================================

🏆 KARŞILAŞTIRMA RAPORU:
🥇 En İyi Tekil Model: model_fold_3.pt (ΔE: 3.3296)
🤝 Ensemble Modeli   : (ΔE: 3.0772)

🛡️ SONUÇ: Ensemble, en iyi tekil modelden 0.2524 puan DAHA GÜVENLİ.
👉 Tavsiye: Ensemble yapısını kullanmaya devam et.

📄 Detaylı Rapor Kaydedildi: Detailed_Analysis_Result.csv

(renkenv310) C:\Users\TAC7\Desktop\RenkAI>python predict_detailed_analysis.py
⚙️ DETAYLI ANALİZ BAŞLIYOR... (5 Model İncelenecek)

==================================================
MODEL           | ORTALAMA ΔE     | DURUM
==================================================
model_fold_0.pt | 3.2636          | ⚠️
model_fold_1.pt | 3.3313          | ⚠️
model_fold_2.pt | 3.5280          | ⚠️
model_fold_3.pt | 2.9837          | ✅
model_fold_4.pt | 3.2142          | ⚠️
--------------------------------------------------
ENSEMBLE (ORT)  | 2.4523          | 🌟
==================================================

🏆 KARŞILAŞTIRMA RAPORU:
🥇 En İyi Tekil Model: model_fold_3.pt (ΔE: 2.9837)
🤝 Ensemble Modeli   : (ΔE: 2.4523)

🛡️ SONUÇ: Ensemble, en iyi tekil modelden 0.5314 puan DAHA GÜVENLİ.
👉 Tavsiye: Ensemble yapısını kullanmaya devam et.

📄 Detaylı Rapor Kaydedildi: Detailed_Analysis_Result_RS.csv

(renkenv310) C:\Users\TAC7\Desktop\RenkAI> 
