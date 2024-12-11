import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import Model, Input
import matplotlib.pyplot as plt

# Membaca file CSV
df = pd.read_csv('./dataset/advanced_nutrition_data.csv')

# Rename columns for easier handling
df.rename(columns={
    'Serving Per Package': 'serving_per_package',
    'Gula (g)': 'gula',
    'Total Gula (g)': 'total_gula',
    'Umur': 'umur',
    'Berat Badan (kg)': 'berat_badan',
    'Riwayat Diabetes' : 'riwayat_diabetes',
    'Kategori Gula': 'kategori_gula',
    'Rekomendasi': 'rekomendasi'
}, inplace=True)

# Encode categorical columns
label_encoder_kategori = LabelEncoder()
label_encoder_rekomendasi = LabelEncoder()
label_encoder_diabetes = LabelEncoder()  # New LabelEncoder for diabetes

df['kategori_gula'] = label_encoder_kategori.fit_transform(df['kategori_gula'])
df['rekomendasi'] = label_encoder_rekomendasi.fit_transform(df['rekomendasi'])
df['diabetes'] = label_encoder_diabetes.fit_transform(df['diabetes'])

# convert to appropriate data types (e.g., float for numeric columns)
df['gula'] = df['gula'].astype(float)
df['total_gula'] = df['total_gula'].astype(float)

# Features and targets
# X = df[['serving_per_package', 'gula', 'total_gula']].values
X = df[["serving_per_package", "gula", "total_gula", "age", "bb", "diabetes"]].values
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
    input_layer = Input(shape=(6,))
    # input_layer = Input(shape=(5,))

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


# Train the model
if __name__ == "__main__":
    history = model.fit(
        X_train, {'kategori_gula': y_train_kategori, 'rekomendasi': y_train_rekomendasi},
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate the model
    loss, loss_kategori, loss_rekomendasi, acc_kategori, acc_rekomendasi = model.evaluate(
        X_test, {'kategori_gula': y_test_kategori, 'rekomendasi': y_test_rekomendasi}, verbose=0
    )

    print(f"loss = {loss},\n acc_kategori = {acc_kategori},\n acc_rekomendasi = {acc_rekomendasi}")

    # Plot Loss
    plt.figure(figsize=(12, 5))

    # Loss for kategori_gula
    plt.subplot(1, 2, 1)
    plt.plot(history.history['kategori_gula_loss'], label='Train Loss (kategori_gula)')
    plt.plot(history.history['val_kategori_gula_loss'], label='Val Loss (kategori_gula)')
    # Loss for rekomendasi
    plt.plot(history.history['rekomendasi_loss'], label='Train Loss (rekomendasi)')
    plt.plot(history.history['val_rekomendasi_loss'], label='Val Loss (rekomendasi)')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot Accuracy
    # Accuracy for kategori_gula
    plt.subplot(1, 2, 2)
    plt.plot(history.history['kategori_gula_accuracy'], label='Train Accuracy (kategori_gula)')
    plt.plot(history.history['val_kategori_gula_accuracy'], label='Val Accuracy (kategori_gula)')
    # Accuracy for rekomendasi
    plt.plot(history.history['rekomendasi_accuracy'], label='Train Accuracy (rekomendasi)')
    plt.plot(history.history['val_rekomendasi_accuracy'], label='Val Accuracy (rekomendasi)')
    plt.title('Accuracy During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # Save the model
    model.save("./model/model_6Variabel_fix.h5")
    print("Model berhasil disimpan")

