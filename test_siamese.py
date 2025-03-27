import tensorflow as tf
import numpy as np
import os

# --- Parametri ---
SAVED_BACKBONE_PATH = 'siamese_backbone_best.h5'
IMAGE_TO_TEST = 'L:\\Tesi Image Quality DermChecker\\dataset\\motion_blurred\\18_XIAOMI-REDMI-5-PLUS_M.jpg' # <-- Immagine da valutare
IMG_SIZE = (224, 224)

# --- Carica il Backbone Addestrato ---
print(f"Caricamento backbone da: {SAVED_BACKBONE_PATH}")
try:
    backbone_model = tf.keras.models.load_model(SAVED_BACKBONE_PATH)
    backbone_model.summary()
except Exception as e:
    print(f"Errore caricamento backbone: {e}")
    exit()

# --- Carica e Prepara l'Immagine di Test ---
print(f"Caricamento immagine test: {IMAGE_TO_TEST}")
try:
    img = tf.keras.utils.load_img(IMAGE_TO_TEST, target_size=IMG_SIZE, interpolation='lanczos')
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0) # Aggiungi dimensione batch
    img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
except Exception as e:
    print(f"Errore caricamento immagine: {e}")
    exit()

# --- Ottieni l'Embedding ---
print("Calcolo embedding...")
embedding = backbone_model.predict(img_tensor)[0] # [0] per rimuovere la dimensione batch
print(f"Embedding calcolato (prime 10 componenti): {embedding[:10]}")
print(f"Dimensione embedding: {embedding.shape}")

# --- Come Ottenere un Punteggio (Esempi) ---

# Opzione 1: Distanza da un Embedding "Ideale" (Richiede calcolo preventivo)
try:
    # Assumi di aver pre-calcolato e salvato l'embedding medio di immagini sharp
    ideal_sharp_embedding = np.load('ideal_sharp_embedding.npy') # Esempio
    distance = np.linalg.norm(embedding - ideal_sharp_embedding) # Distanza Euclidea
    print(f"\nDistanza dall'embedding sharp ideale: {distance:.4f}")
    # Interpretazione: distanza minore = più simile a sharp = qualità migliore (ipotetico)
except FileNotFoundError:
    print("\nFile embedding ideale non trovato. Impossibile calcolare distanza.")

# Opzione 2: Caricare un Classificatore Secondario (Addestrato sugli embedding)
# try:
#    quality_classifier = tf.keras.models.load_model('quality_classifier_on_embeddings.h5')
#    # L'input del classificatore deve matchare l'embedding_dim
#    embedding_batch_for_classifier = np.expand_dims(embedding, axis=0)
#    quality_score = quality_classifier.predict(embedding_batch_for_classifier)[0][0]
#    print(f"\nPunteggio dal classificatore secondario: {quality_score:.4f}")
# except Exception as e:
#    print(f"\nImpossibile usare classificatore secondario: {e}")

print("\nFine.")