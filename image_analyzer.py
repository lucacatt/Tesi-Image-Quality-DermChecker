import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import time
import os

class ImageAnalyzer:
    def __init__(self, dataset_root):
        self.dataset_path_sharp = os.path.join(dataset_root, "sharp")

    @staticmethod
    def laplacian_variance(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()
    
    @staticmethod
    def normalize_image(image):
        return image.astype(np.float32) / 255.0

    def calculate_psnr_skimage(original, degraded):
        original = ImageAnalyzer.normalize_image(original)
        degraded = ImageAnalyzer.normalize_image(degraded)
        mse = np.mean((original - degraded) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(1 / mse)

    def calculate_ssim_skimage(original, degraded):
        original = ImageAnalyzer.normalize_image(original)
        degraded = ImageAnalyzer.normalize_image(degraded)
        return ssim(original, degraded, data_range=1)

    def calculate_psnr_opencv(original, degraded):
        return cv2.PSNR(original, degraded)

    def calculate_ssim_opencv(original, degraded):
        original = ImageAnalyzer.normalize_image(original)
        degraded = ImageAnalyzer.normalize_image(degraded)
        K1 = 0.01 #costante stabilizzazione
        K2 = 0.03 #costante stabilizzazione
        C1 = (K1 * 1) ** 2
        C2 = (K2 * 1) ** 2
        
        mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(degraded, (11, 11), 1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(original**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(degraded**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(original * degraded, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def process_images(self, blurred_dataset, blurred_path, suffix, output_filename):
        results = []
        
        for blurred_file in blurred_dataset:
            if not blurred_file.endswith((".jpg", ".jpeg", ".png")):
                continue  

            sharp_file = blurred_file.replace(suffix, "_S")
            blurred_image_path = os.path.join(blurred_path, blurred_file)
            sharp_image_path = os.path.join(self.dataset_path_sharp, sharp_file)

            blurred_image = cv2.imread(blurred_image_path, cv2.IMREAD_GRAYSCALE)
            sharp_image = cv2.imread(sharp_image_path, cv2.IMREAD_GRAYSCALE)

            if blurred_image is None or sharp_image is None:
                print(f"Immagine non trovata - {sharp_file}")
                continue

            if blurred_image.shape != sharp_image.shape:
                print(f"Dimensioni non corrispondenti tra {blurred_file} e {sharp_file}")
                continue

            start_time = time.perf_counter()
            lap_var = self.laplacian_variance(blurred_image)
            laplacian_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            psnr_value = self.calculate_psnr(sharp_image, blurred_image)
            psnr_time = max(time.perf_counter() - start_time, 1e-6)

            start_time = time.perf_counter()
            ssim_value = self.calculate_ssim(sharp_image, blurred_image)
            ssim_time = max(time.perf_counter() - start_time, 1e-6)

            results.append([blurred_file, sharp_file, lap_var, psnr_value, ssim_value, laplacian_time, psnr_time, ssim_time])

        df = pd.DataFrame(results, columns=["Blurred Image", "Sharp Image", "Laplacian Variance", "PSNR", "SSIM", "Laplacian Time (s)", "PSNR Time (s)", "SSIM Time (s)"])
        output_path = os.path.join(os.path.dirname(__file__), output_filename)
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"File Excel salvato in: {output_path}")