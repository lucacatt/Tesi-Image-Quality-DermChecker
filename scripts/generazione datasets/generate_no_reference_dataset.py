import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

# Config
SHARP_DIR = "sharp"
DEGRADED_DIR = "degraded"
OUTPUT_CSV = "ssim_dataset.csv"
IMG_SIZE = (384, 384)


def load_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

data = []

print("🚀 Generazione dataset no-reference...")

for filename in os.listdir(DEGRADED_DIR):
    degraded_path = os.path.join(DEGRADED_DIR, filename)
    sharp_path = os.path.join(SHARP_DIR, filename)

    if not os.path.exists(sharp_path):
        print(f"⚠️ Mancante: {sharp_path}")
        continue

    try:
        # Carica le immagini
        degraded = load_img(degraded_path)
        sharp = load_img(sharp_path)

        score = ssim(sharp, degraded)
        data.append({
            "image_path": degraded_path,
            "score": score  # Punteggio per l'immagine degradata
        })

        # Aggiungi anche l'immagine sharp al dataset con un punteggio di qualità 1.0
        data.append({
            "image_path": sharp_path,
            "score": 1.0  # Punteggio per l'immagine sharp (qualità perfetta)
        })

    except Exception as e:
        print(f"❌ Errore con {filename}: {e}")

# Salvataggio CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Dataset salvato in: {OUTPUT_CSV}")