import tensorflow as tf  # Importa TensorFlow
from tensorflow.keras import layers, models  # Importa i moduli per costruire il modello
from tensorflow.keras.applications import EfficientNetB3  # Usa EfficientNetB3 invece di MobileNetV2
from tensorflow.keras.optimizers import Adam  # Ottimizzatore Adam
import pandas as pd  # Per la gestione dei dati in formato tabellare
import numpy as np  # Per operazioni matematiche avanzate
import cv2  # Per la manipolazione delle immagini
import os  # Per interazioni con il filesystem
from sklearn.model_selection import train_test_split  # Per dividere i dati in set di addestramento e validazione

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
IMG_SIZE = (384, 384)  # Imposta la dimensione delle immagini per il modello
BATCH_SIZE = 6  # Dimensione del batch
EPOCHS = 30  # Numero di epoche
CSV_PATH = "/content/drive/MyDrive/approccio4/ssim_dataset.csv"  # Percorso al dataset
MODEL_PATH = "/content/drive/MyDrive/approccio4/siamese_quality_model.keras"  # Percorso per salvare il modello

# --- Caricamento dati ---
df = pd.read_csv(CSV_PATH)  # Carica il CSV con i dati delle immagini
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)  # Suddividi il dataset in addestramento e validazione

# Funzione per preprocessare le immagini
def preprocess(path):
    img = cv2.imread(path)  # Legge l'immagine
    img = cv2.resize(img, IMG_SIZE)  # Ridimensiona l'immagine
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte in RGB
    return img.astype("float32") / 255.0  # Normalizza i valori dei pixel

# Funzione per caricare i dati dal CSV
def load_data(df):
    originals = []
    degraded = []
    scores = []

    for _, row in df.iterrows():
        orig = preprocess(row["original"])  # Carica l'immagine originale
        deg = preprocess(row["degraded"])  # Carica l'immagine degradata
        score = row["score"]  # Punteggio di sharpness

        originals.append(orig)
        degraded.append(deg)
        scores.append(score)

    return [np.array(originals), np.array(degraded)], np.array(scores)

X_train, y_train = load_data(train_df)  # Carica i dati di addestramento
X_val, y_val = load_data(val_df)  # Carica i dati di validazione

# --- Costruzione del modello Siamese ---
# Usa EfficientNetB3 come backbone invece di MobileNetV2
base_model = EfficientNetB3(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base_model.trainable = True  # Fine-tuning attivo

# Funzione per costruire una "branch" del modello
def build_branch():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))  # Ingresso dell'immagine
    x = base_model(inputs)  # Passa attraverso EfficientNetB3
    x = layers.GlobalAveragePooling2D()(x)  # Media globale dei valori
    return models.Model(inputs, x)

# Definisce gli ingressi per le immagini originali e degradate
input_1 = layers.Input(shape=(*IMG_SIZE, 3), name="original")
input_2 = layers.Input(shape=(*IMG_SIZE, 3), name="degraded")

branch = build_branch()  # Crea la branch condivisa
feat_1 = branch(input_1)  # Estrai caratteristiche per l'immagine originale
feat_2 = branch(input_2)  # Estrai caratteristiche per l'immagine degradata

# Calcola la differenza assoluta tra le caratteristiche delle due immagini
diff_1 = layers.ReLU()(layers.Subtract()([feat_1, feat_2]))
diff_2 = layers.ReLU()(layers.Subtract()([feat_2, feat_1]))
abs_diff = layers.Add()([diff_1, diff_2])  # Somma delle differenze

# Concatenazione delle caratteristiche e passaggio a un regressore
merged = layers.Concatenate()([feat_1, feat_2, abs_diff])
x = layers.Dense(128, activation="relu")(merged)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)  # Output del modello

model = models.Model(inputs=[input_1, input_2], outputs=output)  # Modello finale

# --- Compilazione e allenamento del modello ---
model.compile(optimizer=Adam(1e-5), loss="mse", metrics=["mae"])  # Compilazione del modello
model.summary()  # Sommario del modello

model.fit(
    X_train, y_train,  # Dati di addestramento
    validation_data=(X_val, y_val),  # Dati di validazione
    epochs=EPOCHS,  # Numero di epoche
    batch_size=BATCH_SIZE,  # Dimensione del batch
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]  # Early stopping
)

# --- Salvataggio del modello ---
model.save(MODEL_PATH)  # Salva il modello
print(f"✅ Modello salvato in: {MODEL_PATH}")
