import tensorflow as tf

model = tf.keras.models.load_model("blur_detection_with_metrics.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Ottimizzazione per mobile
tflite_model = converter.convert()

with open("blur_model_with_metrics.tflite", "wb") as f:
    f.write(tflite_model)

print("Modello convertito in TensorFlow Lite: blur_model_with_metrics.tflite")
