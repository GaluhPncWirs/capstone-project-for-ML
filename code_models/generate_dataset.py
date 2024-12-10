import pandas as pd
import numpy as np

def generate_advanced_nutrition_dataset(samples):
    np.random.seed(42)
    serving_per_package = np.random.randint(1, 15, size=samples)
    gula = np.random.uniform(0.5, 20.0, size=samples).round(2)
    total_gula = (serving_per_package * gula).round(2)

    umur = np.random.randint(5, 80, size=samples)
    berat_badan = np.random.uniform(10, 120, size=samples).round(2)
    diabetes = np.random.choice(['Ada Diabetes', 'Tidak Diabetes'], size=samples, p=[0.3, 0.7])

    kategori_gula = np.where(
        total_gula > 25, "Kadar Gula Tinggi", 
        np.where(total_gula >= 15, "Kadar Gula Sedang", "Kadar Gula Rendah")
    )

    rekomendasi = np.where(
        total_gula > 25, "Lebih Baik Tidak Dikonsumsi", 
        np.where(total_gula >= 15, "Kurangi Konsumsi", "Aman Dikonsumsi")
    )

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

advanced_dataset = generate_advanced_nutrition_dataset(55574)
advanced_dataset.to_csv("./dataset/advanced_nutrition_data.csv", index=False)
print("File 'advanced_nutrition_data.csv' berhasil dibuat!")