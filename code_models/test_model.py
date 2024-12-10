import tensorflow as tf
import numpy as np
from models import label_encoder_kategori, label_encoder_rekomendasi, X

new_data = np.array([[4, 26, 3]])  # serving_per_package, gula, total_gula 3 variabel
# new_data = np.array([[4, 26, 3, 43, 60]])  # serving_per_package, gula, total_gula 5 variabel

new_data = new_data / np.max(X, axis=0)

loaded_model = tf.keras.models.load_model("./model/model_baru/model_5Variabel_fix.h5")
print("Model berhasil dimuat kembali.")

# Prediksi
predictions = loaded_model.predict(new_data)

# Decode hasil prediksi
kategori_gula_pred = label_encoder_kategori.inverse_transform([np.argmax(predictions[0])])[0]
rekomendasi_pred = label_encoder_rekomendasi.inverse_transform([np.argmax(predictions[1])])[0]

print(f"Prediksi Kategori Gula: {kategori_gula_pred}")
print(f"Rekomendasi: {rekomendasi_pred}")





