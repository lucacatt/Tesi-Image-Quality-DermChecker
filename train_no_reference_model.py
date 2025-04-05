import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import sys

# --- Configurazione GPU ---
# Controlla e abilita l'uso della GPU, se disponibile
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    # Abilita la crescita dinamica della memoria
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"💻 Utilizzo della GPU: {physical_devices[0]}")
else:
    print("⚠ GPU non trovata, il modello utilizzerà la CPU.")

# --- Configurazioni principali ---
IMG_SIZE = (300, 300)  # Imposta la dimensione delle immagini
BATCH_SIZE = 12  # Dimensione del batch
EPOCHS = 30  # Numero di epoche
CSV_PATH = "/content/drive/MyDrive/approccio4/no_reference_dataset.csv"  # Percorso al dataset
MODEL_PATH = "/content/drive/MyDrive/approccio4/no_reference_efficientnet.keras"  # Percorso per salvare il modello

# --- Caricamento CSV ---
df = pd.read_csv(CSV_PATH)  # Carica il CSV con i dati delle immagini
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)  # Suddividi il dataset in addestramento e validazione

# Funzione per caricare e preprocessare le immagini
def load_img(path):
    img = cv2.imread("/content/drive/MyDrive/approccio4/"+ path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

# Funzione per caricare i dati dal CSV
def load_data(df):
    images = []
    scores = []

    for _, row in df.iterrows():
        img = load_img(row["image_path"])  # Carica l'immagine
        score = row["score"]  # Punteggio di sharpness
        images.append(img)
        scores.append(score)

    return np.array(images), np.array(scores)

X_train, y_train = load_data(train_df)  # Carica i dati di addestramento
X_val, y_val = load_data(val_df)  # Carica i dati di validazione

# --- Pesi dinamici per i sample ---
sample_weights = 1.0 + (1.0 - y_train) * 4.0  # Aumenta il peso dei campioni con basso punteggio di qualità

# --- Architettura EfficientNetB3 ---
base_model = EfficientNetB3(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base_model.trainable = True  # Fine-tuning attivo

inputs = layers.Input(shape=(*IMG_SIZE, 3))  # Ingresso delle immagini
x = base_model(inputs)  # Passa le immagini attraverso EfficientNetB3
x = layers.GlobalAveragePooling2D()(x)  # Media globale dei valori
x = layers.Dense(256, activation="relu")(x)  # Layer denso con ReLU
x = layers.Dropout(0.4)(x)  # Dropout per prevenire overfitting
x = layers.Dense(64, activation="relu")(x)  # Ulteriore layer denso
output = layers.Dense(1, activation="sigmoid")(x)  # Output con una singola unità per il punteggio di sharpness

model = models.Model(inputs, output)  # Crea il modello finale

model.compile(optimizer=Adam(1e-5), loss="mae", metrics=["mse", "mae"])  # Compilazione del modello
model.summary()  # Sommario del modello

# --- Training del modello ---
model.fit(
    X_train, y_train,
    sample_weight=sample_weights,  # Usa i pesi dinamici
    validation_data=(X_val, y_val),  # Dati di validazione
    epochs=EPOCHS,  # Numero di epoche
    batch_size=BATCH_SIZE,  # Dimensione del batch
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]  # Early stopping
)

# --- Salvataggio del modello ---
model.save(MODEL_PATH)  # Salva il modello
print(f"✅ Modello EfficientNetB3 salvato in: {MODEL_PATH}")