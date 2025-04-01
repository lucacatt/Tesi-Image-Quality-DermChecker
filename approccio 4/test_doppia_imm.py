from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

IMG_SIZE = (224, 224)

def load_img(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Immagine non trovata: {path}")
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

model = load_model("siamese_quality_model.keras", compile=False)

original = load_img("sharp/28_002.jpg")
degraded = load_img("degraded/28_002.jpg")

original = np.expand_dims(original, axis=0)
degraded = np.expand_dims(degraded, axis=0)

score = model.predict([original, degraded])[0][0]
print(f"📈 Sharpness score stimato: {score:.4f}")
