import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
SHARP_DIR = "L:\\Tesi Image Quality DermChecker\\Tesi-Image-Quality-DermChecker\\sharp"
DEGRADED_DIR = "L:\\Tesi Image Quality DermChecker\\Tesi-Image-Quality-DermChecker\\degraded"
MODEL_PATH = "quality_score_model.h5"

# --- 1. CARICA IMMAGINI + SCORE ---
def load_images_from_folder(folder, label_fn):
    images = []
    labels = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
            arr = tf.keras.utils.img_to_array(img) / 255.0
            images.append(arr)
            labels.append(label_fn(fname))
        except Exception as e:
            print(f"❌ Errore su {fname}: {e}")
    return np.array(images), np.array(labels)

# Per immagini nitide: score 1.0
sharp_images, sharp_labels = load_images_from_folder(SHARP_DIR, lambda _: 1.0)

# Per degradate: score stimato come 1 - degradazioni / 5
def infer_score_from_filename(fname):
    try:
        # Esempio: IMG_001_deg3.jpg → score = 1 - 3/5 = 0.4
        parts = fname.lower().split("deg")
        if len(parts) >= 2:
            degr = int(parts[1][0])  # prende primo numero dopo 'deg'
            degr = min(max(degr, 1), 5)
            return 1.0 - degr / 5.0
    except:
        pass
    return 0.0  # fallback: assume massimo degrado

degraded_images, degraded_labels = load_images_from_folder(DEGRADED_DIR, infer_score_from_filename)

# Unione
X = np.concatenate([sharp_images, degraded_images], axis=0)
y = np.concatenate([sharp_labels, degraded_labels], axis=0)

print(f"📦 Dataset totale: {len(X)} immagini")

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. MODELLO ---
base = MobileNetV2(include_top=False, input_shape=(*IMG_SIZE, 3), weights="imagenet")
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# --- 3. TRAINING ---
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# --- 4. SALVA MODELLO ---
model.save(MODEL_PATH)
print(f"✅ Modello salvato in: {MODEL_PATH}")
