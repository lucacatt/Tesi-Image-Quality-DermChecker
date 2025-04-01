# extract_embeddings.py
import os
import numpy as np
import tensorflow as tf
from train_siamese import l2_normalize_layer

# Config
IMG_SIZE = (224, 224)
BACKBONE_PATH = 'siamese_backbone_best.h5'
SHARP_DIR = 'sharp'
DEGRADED_DIR = 'degraded'

# Carica modello
model = tf.keras.models.load_model(BACKBONE_PATH, custom_objects={'l2_normalize_layer': l2_normalize_layer})

def get_embedding(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    batch = np.expand_dims(arr, axis=0)
    return model.predict(batch, verbose=0)[0]

# Estrazione
embeddings, labels = [], []

for folder, label in [(SHARP_DIR, 1), (DEGRADED_DIR, 0)]:
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            emb = get_embedding(path)
            embeddings.append(emb)
            labels.append(label)
        except Exception as e:
            print(f"Errore su {path}: {e}")

np.save("embeddings.npy", np.array(embeddings))
np.save("labels.npy", np.array(labels))
print("Embeddings e label salvati.")
