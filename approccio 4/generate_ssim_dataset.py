# generate_ssim_dataset.py

import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Cartelle di origine
SHARP_DIR = "sharp"
DEGRADED_DIR = "degraded"
OUTPUT_CSV = "ssim_dataset.csv"

IMG_SIZE = (224, 224)

data = []

print("📦 Inizio generazione dataset con SSIM...")

for filename in os.listdir(SHARP_DIR):
    sharp_path = os.path.join(SHARP_DIR, filename)
    degraded_path = os.path.join(DEGRADED_DIR, filename)

    if not os.path.exists(degraded_path):
        print(f"⚠️ Mancante: {degraded_path}")
        continue

    try:
        sharp_img = cv2.imread(sharp_path)
        degraded_img = cv2.imread(degraded_path)

        sharp_img = cv2.resize(sharp_img, IMG_SIZE)
        degraded_img = cv2.resize(degraded_img, IMG_SIZE)

        # Converti in scala di grigi per SSIM
        sharp_gray = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2GRAY)
        degraded_gray = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2GRAY)

        score = ssim(sharp_gray, degraded_gray)

        data.append({
            "original": sharp_path,
            "degraded": degraded_path,
            "score": score
        })

    except Exception as e:
        print(f"❌ Errore con {filename}: {e}")

# Salva CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Dataset generato e salvato in: {OUTPUT_CSV}")
