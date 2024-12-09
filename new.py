import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import Model, Input
import pathlib

# Membaca file CSV
df = pd.read_csv('nutrition_data_fix.csv')

# Rename columns for easier handling
df.rename(columns={
    'Serving Per Package': 'serving_per_package',
    'Gula (g)': 'gula',
    'Total Gula (g)': 'total_gula',
    # 'Umur': 'umur',
    # 'Berat Badan (kg)': 'berat badan',
    'Kategori Gula': 'kategori_gula',
    'Rekomendasi': 'rekomendasi'
}, inplace=True)

# Encode categorical columns
label_encoder_kategori = LabelEncoder()
label_encoder_rekomendasi = LabelEncoder()

df['kategori_gula'] = label_encoder_kategori.fit_transform(df['kategori_gula'])
df['rekomendasi'] = label_encoder_rekomendasi.fit_transform(df['rekomendasi'])

# Features and targets
X = df[['serving_per_package', 'gula', 'total_gula']].values
y_kategori = df['kategori_gula'].values
y_rekomendasi = df['rekomendasi'].values

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train_kategori, y_test_kategori, y_train_rekomendasi, y_test_rekomendasi = train_test_split(
    X_scaled, y_kategori, y_rekomendasi, test_size=0.2, random_state=42
)

# Define models 
def modelCnn():
    input_layer = Input(shape=(3,))

    # Shared layers
    shared = Dense(64, activation='relu')(input_layer)
    shared = Dense(32, activation='relu')(shared)

    # Output for kategori_gula
    output_kategori = Dense(len(label_encoder_kategori.classes_), activation='softmax', name='kategori_gula')(shared)

    # Output for rekomendasi
    output_rekomendasi = Dense(len(label_encoder_rekomendasi.classes_), activation='softmax', name='rekomendasi')(shared)

    # Build model
    model = Model(inputs=input_layer, outputs=[output_kategori, output_rekomendasi])

    return model

# Compile model with multiple loss functions
model = modelCnn()

model.compile(
    optimizer='adam',
    loss={
        'kategori_gula': 'sparse_categorical_crossentropy',
        'rekomendasi': 'sparse_categorical_crossentropy'
    },
    metrics={
        'kategori_gula': 'accuracy',
        'rekomendasi': 'accuracy'
    }
)

# # Train the model
# history = model.fit(
#     X_train, {'kategori_gula': y_train_kategori, 'rekomendasi': y_train_rekomendasi},
#     epochs=50,
#     batch_size=32,
#     validation_split=0.2,
#     verbose=1
# )

# # Evaluate the model
# loss, loss_kategori, loss_rekomendasi, acc_kategori, acc_rekomendasi = model.evaluate(
#     X_test, {'kategori_gula': y_test_kategori, 'rekomendasi': y_test_rekomendasi}, verbose=0
# )

# loss, acc_kategori, acc_rekomendasi


# # **Simpan Model ke Disk**
# model.save("modellll_Fixs_5Variabel.h5")
# print("Model berhasil disimpan sebagai 'model_Fixs_5Variabel.h5'.")

# export_dir = './saved_model_project/'

# model = tf.keras.models.load_model("model_Fix_5Variabel.keras")

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the converted model to a .tflite fileS
# with open("model_Fix_5Variabel.tflite", "wb") as f:
#     f.write(tflite_model)

# print("Model has been converted and saved as model.tflite")

# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# tflite_model = converter.convert()

# tflite_model_file = pathlib.Path('vegs.tflite')
# tflite_model_file.write_bytes(tflite_model)


# # Convert model from Keras
# model = tf.keras.models.load_model("model_Fix_5Variabel.keras")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open("model_Fix_5Variabel.tflite", "wb") as f:
#     f.write(tflite_model)
# print("Keras model converted and saved as model_Fix_5Variabel.tflite")

# # Save as SavedModel
# model.export('./saved_model_project/')  # Gunakan export() untuk format SavedModel

# # Convert SavedModel
# converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model_project/')
# tflite_model = converter.convert()

# tflite_model_file = pathlib.Path('vegs.tflite')
# tflite_model_file.write_bytes(tflite_model)
# print("SavedModel converted and saved as vegs.tflite")

# Pastikan LabelEncoder menggunakan label yang sama seperti saat melatih
label_encoder_diabetes = LabelEncoder()
label_encoder_diabetes.fit(["Tidak Diabetes", "Ada Diabetes"])  # Sesuaikan dengan data asli

# Contoh data baru
new_data = np.array([[1.5, 2, 18]])  # serving_per_package, gula, total_gula
diabetes_status = "Tidak Diabetes"  # Misalnya, data tambahan

# Normalisasi fitur numerik
new_data = new_data / np.max(X, axis=0)  # Gunakan skala yang sama dengan data pelatihan

# **Memuat Kembali Model (Opsional)**
loaded_model = tf.keras.models.load_model("bismillah_bisa_model.h5")
print("Model berhasil dimuat kembali.")

# Prediksi
predictions = loaded_model.predict(new_data)

# Decode hasil prediksi
kategori_gula_pred = label_encoder_kategori.inverse_transform([np.argmax(predictions[0])])[0]
rekomendasi_pred = label_encoder_rekomendasi.inverse_transform([np.argmax(predictions[1])])[0]

print(f"Prediksi Kategori Gula: {kategori_gula_pred}")
print(f"Rekomendasi: {rekomendasi_pred}")

