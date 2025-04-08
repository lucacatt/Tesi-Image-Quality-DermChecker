import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import sys
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

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
IMG_SIZE = (384, 384)  # Imposta la dimensione delle immagini
BATCH_SIZE = 12  # Dimensione del batch
EPOCHS = 30  # Numero di epoche
CSV_PATH = "dati_csv/ssim_dataset.csv"  # Percorso al dataset
MODEL_PATH = "modelli_keras/no_reference_efficientnet.keras"  # Percorso per salvare il modello
PREPROCESS_FUNCTION = efficientnet_preprocess_input

# --- Caricamento CSV ---
df = pd.read_csv(CSV_PATH)  # Carica il CSV con i dati delle immagini
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)  # Suddividi il dataset in addestramento e validazione

# Funzione per caricare e preprocessare le immagini
def load_img_base(path):
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"Attenzione: Impossibile leggere {path}, verrà saltata.")
            return None
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Restituisce l'immagine come float32, ma senza normalizzazione qui
        return img.astype("float32")
    except Exception as e:
        print(f"Errore durante caricamento/resize di {path}: {e}")
        return None

# Funzione per caricare i dati dal CSV
def load_data(df):
    images = []
    scores = []
    valid_indices = [] # Tiene traccia degli indici validi

    for index, row in df.iterrows():
        img = load_img_base(row["image_path"])
        if img is not None:
            images.append(img)
            scores.append(row["score"])
            valid_indices.append(index) # Salva l'indice originale

    if not images: # Se nessuna immagine è stata caricata con successo
        return np.array([]), np.array([]), []

    # Applica la funzione di preprocessing specifica del modello DOPO aver caricato
    # tutte le immagini del batch (o del set in questo caso)
    images_np = np.array(images)
    images_preprocessed = PREPROCESS_FUNCTION(images_np)

    return images_preprocessed, np.array(scores), valid_indices

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