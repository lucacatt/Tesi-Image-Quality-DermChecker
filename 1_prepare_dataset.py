import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

dataset_path = "C:\\Users\\crist\\Downloads\\GOPRO_Large\\test"

categories = ["blur", "blur_gamma", "sharp"]

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

data = []
labels = []
metrics = []

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
            blurred_img = cv2.GaussianBlur(img, (5,5), 0)
            psnr_value = psnr(img, blurred_img)
            ssim_value, _ = ssim(img, blurred_img, full=True)

            data.append(img_to_array(load_img(img_path, target_size=(224,224))) / 255.0)
            labels.append(label)
            metrics.append([laplacian, psnr_value, ssim_value])

np.save("data.npy", np.array(data))
np.save("labels.npy", np.array(labels))
np.save("metrics.npy", np.array(metrics))

print("Dataset GoPro elaborato e salvato con metriche Laplacian, PSNR e SSIM!")
