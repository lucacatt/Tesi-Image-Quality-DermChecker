import os
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

dataset_path = "dataset"
categories = ["defocused_blurred", "motion_blurred", "sharp"]
model_path = "blur_model_with_metrics.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_index = 0 if tuple(input_details[0]['shape'][1:]) == (224, 224, 3) else 1
metrics_index = 1 - image_index

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

results = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    label = 0 if category != "sharp" else 1 

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        print(f"Processing {filename} in {category}...")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_resized = np.expand_dims(img_resized.astype(np.float32), axis=0)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = laplacian_variance(img_gray)
        blurred_img = cv2.GaussianBlur(img_gray, (5,5), 0)
        psnr_value = psnr(img_gray, blurred_img)
        ssim_value, _ = ssim(img_gray, blurred_img, full=True)

        metrics_input = np.array([[laplacian, psnr_value, ssim_value]], dtype=np.float32)

        interpreter.set_tensor(input_details[image_index]['index'], img_resized)
        interpreter.set_tensor(input_details[metrics_index]['index'], metrics_input)

        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        predicted_label = "Nitida" if prediction > 0.5 else "Sfocata"

        results.append((filename, category, laplacian, psnr_value, ssim_value, prediction, predicted_label))

df = pd.DataFrame(results, columns=["Filename", "Categoria", "Laplacian", "PSNR", "SSIM", "Predizione", "Risultato"])
df.to_csv("test_results.csv", index=False)
print("Risultati salvati in test_result.csv")
