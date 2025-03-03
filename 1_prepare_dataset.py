import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# dataset_path = "C:\\Users\\crist\\Downloads\\GOPRO_Large\\test"
# dataset_path2 = "C:\\Users\\crist\\Downloads\\GOPRO_Large\\train"
categories = ["Anaglyph","blur", "gif", "gt"]

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

dataset_paths = ["C:\\Users\\crist\\Downloads\\RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref","C:\\Users\\crist\\Downloads\\RealBlur-J_ECC_IMCORR_centroid_itensity_ref"]

if os.path.exists("data.npy"):
    print("Carico dataset esistente...")
    existing_data = np.load("data.npy")
    existing_labels = np.load("labels.npy")
    existing_metrics = np.load("metrics.npy")
    
    # Converto in liste Python per fare .append()
    data = existing_data.tolist()
    labels = existing_labels.tolist()
    metrics = existing_metrics.tolist()
else:
    print("Nessun dataset esistente trovato. Creo nuovi array...")
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
                blurred_img = cv2.GaussianBlur(img, (5,5), 0)
                psnr_value = psnr(img, blurred_img)
                ssim_value, _ = ssim(img, blurred_img, full=True)

                data.append(img_to_array(load_img(img_path, target_size=(224,224))) / 255.0)
                labels.append(label)
                metrics.append([laplacian, psnr_value, ssim_value])

final_data = np.array(data, dtype=np.float32)
final_labels = np.array(labels, dtype=np.int32)
final_metrics = np.array(metrics, dtype=np.float32)

np.save("data.npy", final_data)
np.save("labels.npy", final_labels)
np.save("metrics.npy", final_metrics)

print("Dataset elaborato e salvato con metriche Laplacian, PSNR e SSIM!")
