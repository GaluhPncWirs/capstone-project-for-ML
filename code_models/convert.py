import tensorflow as tf
import json
import pathlib


export_dir = './saved_model_project/'

model = tf.keras.models.load_model("model_Fix_5Variabel.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a .tflite fileS
with open("model_Fix_5Variabel.tflite", "wb") as f:
    f.write(tflite_model)

print("Model has been converted and saved as model.tflite")

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('vegs.tflite')
tflite_model_file.write_bytes(tflite_model)


# Convert model from Keras
model = tf.keras.models.load_model("model_Fix_5Variabel.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model_Fix_5Variabel.tflite", "wb") as f:
    f.write(tflite_model)
print("Keras model converted and saved as model_Fix_5Variabel.tflite")

# Save as SavedModel
model.export('./saved_model_project/')  # Gunakan export() untuk format SavedModel

# Convert SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model_project/')
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('vegs.tflite')
tflite_model_file.write_bytes(tflite_model)
print("SavedModel converted and saved as vegs.tflite")



# # Load the model from the .h5 file
# model_path = "model_capstone.h5"
# model = tf.keras.models.load_model(model_path)

# # Convert the model's architecture to JSON
# model_json = model.to_json()

# # Save the JSON to a file
# json_path = "./data/model_architecture.json"
# with open(json_path, "w") as json_file:
#     json_file.write(model_json)

# print(f"Model architecture saved to {json_path}")
