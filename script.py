import cv2
import numpy as np
import pandas as pd
import time
from skimage.metrics import structural_similarity as ssim
import os

dataset_path_blurred = os.path.join(os.path.dirname(__file__), "dataset", "blur_dataset_scaled", "defocused_blurred")
dataset_path_sharp = os.path.join(os.path.dirname(__file__), "dataset", "blur_dataset_scaled", "sharp")
image_files_blurred = [f for f in os.listdir(dataset_path_blurred) if f.endswith(('.png', '.jpg', '.jpeg'))]
results = []

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_psnr_skimage(original, degraded):
    mse = np.mean((original - degraded) ** 2)
    if mse == 0:
        return float('inf')  
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

def calculate_ssim_skimage(original, degraded):
    return ssim(original, degraded, data_range=original.max() - original.min())

def calculate_psnr_opencv(original, degraded):
    psnr_value = cv2.PSNR(original, degraded)
    return psnr_value if psnr_value > 0 else float('inf')  

def calculate_ssim_opencv(original, degraded):
    if np.all(original == original[0, 0]) or np.all(degraded == degraded[0, 0]):
        return 1.0  

    C1 = 6.5025  
    C2 = 58.5225
    original = original.astype(np.float32)
    degraded = degraded.astype(np.float32)
    mu1 = cv2.GaussianBlur(original, (5, 5), 1.5)
    mu2 = cv2.GaussianBlur(degraded, (5, 5), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(original**2, (5, 5), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(degraded**2, (5, 5), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * degraded, (5, 5), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

for blurred_file in image_files_blurred:
    if not blurred_file.endswith("_F.jpg") and not blurred_file.endswith("_F.jpeg") and not blurred_file.endswith("_F.png"):
        continue  
    sharp_file = blurred_file.replace("_F", "_S")
    
    blurred_image_path = os.path.join(dataset_path_blurred, blurred_file)
    sharp_image_path = os.path.join(dataset_path_sharp, sharp_file)

    blurred_image = cv2.imread(blurred_image_path, cv2.IMREAD_GRAYSCALE)
    sharp_image = cv2.imread(sharp_image_path, cv2.IMREAD_GRAYSCALE)

    if blurred_image is None or sharp_image is None:
        print(f"Errore immagine {sharp_file}")
        continue

    if blurred_image.shape != sharp_image.shape:
        print(f"Errore tra immagine {blurred_file} e {sharp_file}")
        continue

    start_time = time.perf_counter()
    lap_var = laplacian_variance(blurred_image)
    laplacian_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    psnr_value_skimage = calculate_psnr_skimage(sharp_image, blurred_image)
    psnr_time_skimage = max(time.perf_counter() - start_time, 1e-6)  

    start_time = time.perf_counter()
    ssim_value_skimage = calculate_ssim_skimage(sharp_image, blurred_image)
    ssim_time_skimage = max(time.perf_counter() - start_time, 1e-6)  

    start_time = time.perf_counter()
    psnr_value_opencv = calculate_psnr_opencv(sharp_image, blurred_image)
    psnr_time_opencv = max(time.perf_counter() - start_time, 1e-6)  

    start_time = time.perf_counter()
    ssim_value_opencv = calculate_ssim_opencv(sharp_image, blurred_image)
    ssim_time_opencv = max(time.perf_counter() - start_time, 1e-6)  

    results.append([
        blurred_file, sharp_file, lap_var, 
        psnr_value_skimage, ssim_value_skimage, 
        psnr_value_opencv, ssim_value_opencv, 
        laplacian_time, psnr_time_skimage, ssim_time_skimage, 
        psnr_time_opencv, ssim_time_opencv
    ])


df = pd.DataFrame(results, columns=[
    "Blurred Image", "Sharp Image", "Laplacian Variance", 
    "PSNR skimage", "SSIM skimage", "PSNR openCV", "SSIM openCV", 
    "Laplacian Time (s)", "PSNR Time (s) skimage", "SSIM Time (s) skimage", 
    "PSNR Time (s) openCV", "SSIM Time (s) openCV"
])

output_path = os.path.join(os.path.dirname(__file__), "catalogazione.xlsx")
df.to_excel(output_path, index=False, engine="openpyxl")

print(f"File Excel salvato in: {output_path}")
