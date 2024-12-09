import pandas as pd
import numpy as np

def generate_advanced_nutrition_dataset(samples=10000):
    np.random.seed(42)
    serving_per_package = np.random.randint(1.1, 15, size=samples).round(2)
    gula = np.random.uniform(0.5, 20.0, size=samples).round(2)
    total_gula = (serving_per_package * gula).round(2)
    umur = np.random.randint(18, 80, size=samples)
    berat_badan = np.random.uniform(40, 120, size=samples).round(2)
    diabetes = np.random.choice(['Ada Diabetes', 'Tidak Diabetes'], size=samples, p=[0.3, 0.7])
    kategori_gula = np.where(total_gula > 15, "Tinggi Gula", "Rendah Gula")
    rekomendasi = np.where(total_gula > 15, "Kurangi Konsumsi", "Aman Dikonsumsi")
    dataset = pd.DataFrame({
        "Serving Per Package": serving_per_package,
        "Gula (g)": gula,
        "Total Gula (g)": total_gula,
        "Umur": umur,
        "Berat Badan (kg)": berat_badan,
        "Riwayat Diabetes": diabetes,
        "Kategori Gula": kategori_gula,
        "Rekomendasi": rekomendasi
    })
    return dataset

advanced_dataset = generate_advanced_nutrition_dataset(10000)
advanced_dataset.to_csv("advanced_nutrition_data.csv", index=False)
print("File 'advanced_nutrition_data.csv' berhasil dibuat!")