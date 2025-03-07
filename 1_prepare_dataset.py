import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

categories = ["sharp","blur_gamma", "blur"]

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def variance_of_gradient(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return (gx.var() + gy.var()) / 2

def tenengrad(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return np.sqrt(sobelx**2 + sobely**2).mean()

dataset_paths = ["C:\\Users\\crist\\Downloads\\GOPRO_Large\\train","C:\\Users\\crist\\Downloads\\GOPRO_Large\\test"]

data = []
labels = []
metrics = []

for dataset_path in dataset_paths:
    for video_folder in os.listdir(dataset_path):
        video_path = os.path.join(dataset_path, video_folder)
        if not os.path.isdir(video_path):
            continue
        
        for category in categories:
            folder = os.path.join(video_path, category)
            if not os.path.exists(folder):
                continue
            
            label = 0 if category != "sharp" else 1

            for filename in os.listdir(folder):
                print(f"Processing {filename} in {video_folder}/{category}...")
                img_path = os.path.join(folder, filename)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                laplacian = laplacian_variance(img)
                vog = variance_of_gradient(img)
                tenengrad_val = tenengrad(img)

                blurred_img = cv2.GaussianBlur(img, (5,5), 0)
                psnr_value = psnr(img, blurred_img)
                ssim_value, _ = ssim(img, blurred_img, full=True)

                data.append(img_to_array(load_img(img_path, target_size=(224,224))) / 255.0)
                labels.append(label)
                metrics.append([laplacian, vog, tenengrad_val, psnr_value, ssim_value])

final_data = np.array(data, dtype=np.float32)
final_labels = np.array(labels, dtype=np.int32)
final_metrics = np.array(metrics, dtype=np.float32)

np.save("data.npy", final_data)
np.save("labels.npy", final_labels)
np.save("metrics.npy", final_metrics)

print("Dataset elaborato e salvato con Laplacian, VoG, Tenengrad, PSNR e SSIM!")