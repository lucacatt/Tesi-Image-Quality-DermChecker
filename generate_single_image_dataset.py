import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# Config
SHARP_DIR = "/content/drive/MyDrive/approccio4/sharp"
DEGRADED_DIR = "/content/drive/MyDrive/approccio4/degraded"
MODEL_PATH = "/content/drive/MyDrive/approccio4/siamese_quality_model.keras"
OUTPUT_CSV = "/content/drive/MyDrive/approccio4/no_reference_dataset.csv"
IMG_SIZE = (384, 384)

# Load modello Siamese
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

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

        # Aggiungi l'immagine degradata al dataset con il punteggio predetto dal modello Siamese
        degraded_batch = np.expand_dims(degraded, axis=0)
        sharp_batch = np.expand_dims(sharp, axis=0)
        score = model.predict([sharp_batch, degraded_batch])[0][0]
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
