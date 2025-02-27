import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


dataset_path = "dataset"

categories = ["defocused_blurred", "motion_blurred", "sharp"]

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

data = []
labels = []
metrics = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    label = 0 if category != "sharp" else 1  # 0 = sfocata (blur/motion blur), 1 = nitida

    for filename in os.listdir(folder):
        print(f"Processing {filename} in {category}...")
        img_path = os.path.join(folder, filename)

        # Carica immagine in scala di grigi
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # 🔢 Calcola le metriche
        laplacian = laplacian_variance(img)

        # Per PSNR e SSIM, confrontiamo con una versione sfocata
        blurred_img = cv2.GaussianBlur(img, (5,5), 0)
        psnr_value = psnr(img, blurred_img)
        ssim_value, _ = ssim(img, blurred_img, full=True)

        # 📦 Salviamo i dati
        data.append(img_to_array(load_img(img_path, target_size=(224,224))) / 255.0)
        labels.append(label)
        metrics.append([laplacian, psnr_value, ssim_value])


data = np.array(data)
labels = np.array(labels)
metrics = np.array(metrics)

np.save("data.npy", data)
np.save("labels.npy", labels)
np.save("metrics.npy", metrics)

print("Dataset elaborato e salvato con metriche Laplacian, PSNR e SSIM!")
