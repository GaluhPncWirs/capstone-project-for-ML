import tensorflow as tf
import numpy as np
from models import label_encoder_kategori, label_encoder_rekomendasi, X

# Contoh data baru
new_data = np.array([[4, 26, 3]])  # serving_per_package, gula, total_gula

# Normalisasi fitur numerik
new_data = new_data / np.max(X, axis=0)  # Gunakan skala yang sama dengan data pelatihan

loaded_model = tf.keras.models.load_model("./model/model_baru/model_5Variabel_fix.h5")
print("Model berhasil dimuat kembali.")

# Prediksi
predictions = loaded_model.predict(new_data)

# Decode hasil prediksi
kategori_gula_pred = label_encoder_kategori.inverse_transform([np.argmax(predictions[0])])[0]
rekomendasi_pred = label_encoder_rekomendasi.inverse_transform([np.argmax(predictions[1])])[0]

print(f"Prediksi Kategori Gula: {kategori_gula_pred}")
print(f"Rekomendasi: {rekomendasi_pred}")





