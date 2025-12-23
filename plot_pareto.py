import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_pareto_chart():
    # Temizlenmiş Pareto Verisi
    data = {
        'Pigment_Count': [3, 5, 8],
        'Delta_E': [3.79, 2.33, 0.77]
    }
    df = pd.DataFrame(data)

    # Grafik Ayarları
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot') # Akademik görünüm için
    
    # Çizgi ve Noktalar
    plt.plot(df['Pigment_Count'], df['Delta_E'], color='#2c3e50', linestyle='--', alpha=0.7, zorder=1)
    plt.scatter(df['Pigment_Count'], df['Delta_E'], color='#e74c3c', s=150, zorder=2, label='Pareto Optimal Solutions')

    # Etiketler
    for i, row in df.iterrows():
        plt.annotate(f"ΔE={row['Delta_E']}", 
                     (row['Pigment_Count'], row['Delta_E']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', fontsize=12, fontweight='bold')

    # Eksen İsimleri ve Başlık
    plt.title('Pareto Frontier: Cost vs. Accuracy Trade-off', fontsize=16)
    plt.xlabel('Number of Active Pigments (Cost)', fontsize=14)
    plt.ylabel('Color Difference ($\Delta E_{00}$)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Bölge Açıklamaları (Annotations)
    plt.text(3, 3.2, "Low Cost\n(Lower Accuracy)", ha='center', color='gray', fontsize=10)
    plt.text(8, 1.2, "High Fidelity\n(Higher Cost)", ha='center', color='gray', fontsize=10)

    # Kaydet
    plt.savefig('pareto_chart.png', dpi=300, bbox_inches='tight')
    print("✅ Grafik 'pareto_chart.png' olarak kaydedildi.")

if __name__ == "__main__":
    plot_pareto_chart()