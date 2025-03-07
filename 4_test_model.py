import tensorflow as tf
import numpy as np
import cv2
import os
import time
import pandas as pd

dataset_path = "dataset"
categories = ["defocused_blurred", "motion_blurred", "sharp"]

interpreter = tf.lite.Interpreter(model_path="blur_model_with_metrics.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_index = 0  
metrics_index = 0  
classification_index = 1  

results = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_resized = np.expand_dims(img_resized.astype(np.float32), axis=0)

        start_time = time.time()
        interpreter.set_tensor(input_details[image_index]['index'], img_resized)
        interpreter.invoke()

        metrics_output = interpreter.get_tensor(output_details[metrics_index]['index'])[0]  
        tflite_prediction = interpreter.get_tensor(output_details[classification_index]['index'])[0][0]  

        tflite_time = time.time() - start_time
        predicted_label_tflite = "Nitida" if tflite_prediction > 0.5 else "Sfocata"

        if len(metrics_output) >= 5:
            laplacian, vog, tenengrad, ssim, psnr = metrics_output
        else:
            laplacian, vog, tenengrad, ssim, psnr = None, None, None, None, None

        results.append({
            "Immagine": filename,
            "Categoria": category,
            "TFLite Pred": tflite_prediction,
            "TFLite Label": predicted_label_tflite,
            "Lap": laplacian,
            "VoG": vog,
            "Tenengrad": tenengrad,
            "SSIM": ssim,
            "PSNR": psnr,
            "Tempo TFLite (s)": round(tflite_time, 4)
        })

df_results = pd.DataFrame(results)

df_results.to_csv("test_tflite_results.csv", index=False)
print("Test completato con SSIM e PSNR!")
