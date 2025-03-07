import tensorflow as tf
from SharpnessMetricsLayerClass import SharpnessMetricsLayer
try:
    model = tf.keras.models.load_model("blur_detection_with_metrics.h5", custom_objects={"SharpnessMetricsLayer": SharpnessMetricsLayer})
    print("Modello caricato con successo.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]  #Try with flex ops
    tflite_model = converter.convert()


    with open("blur_model_with_metrics.tflite", "wb") as f:
        f.write(tflite_model)

    print("Modello convertito in TensorFlow Lite: blur_model_with_metrics.tflite")

except Exception as e:
    print(f"Errore durante la conversione: {e}")