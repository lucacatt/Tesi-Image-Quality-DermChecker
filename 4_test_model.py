import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

dataset_path = "dataset" 
categories = ["defocused_blurred", "motion_blurred", "sharp"]

interpreter = tf.lite.Interpreter(model_path="blur_model_with_metrics.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

results = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    label = 0 if category != "sharp" else 1 

    for filename in os.listdir(folder):
        print(f"Processing {filename} in {category}...")
        img_path = os.path.join(folder, filename)

        # Carica immagine in scala di grigi
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        laplacian = laplacian_variance(img)
        blurred_img = cv2.GaussianBlur(img, (5,5), 0)
        psnr_value = psnr(img, blurred_img)
        ssim_value, _ = ssim(img, blurred_img, full=True)

        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_resized = np.expand_dims(img_resized, axis=(0, -1)) 
        img_resized = np.repeat(img_resized, 3, axis=-1)  
        
        metrics_input = np.array([[laplacian, psnr_value, ssim_value]], dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_resized.astype(np.float32))
        interpreter.set_tensor(input_details[1]['index'], metrics_input)

        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        predicted_label = "Nitida" if prediction > 0.5 else "Sfocata"

        results.append((filename, category, laplacian, psnr_value, ssim_value, prediction, predicted_label))

print("📊 RISULTATI DEL TEST")
print("Filename | Categoria | Laplacian | PSNR | SSIM | Predizione | Risultato")
for r in results:
    print(f"{r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]}")

import pandas as pd
df = pd.DataFrame(results, columns=["Filename", "Categoria", "Laplacian", "PSNR", "SSIM", "Predizione", "Risultato"])
df.to_csv("test_results.csv", index=False)

print("Risultati salvati in test_results.csv")
