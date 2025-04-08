import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import sys
import os

MODEL_CHOICE = 'MobileNetV2'
# -----------------------------------------

# --- Configurazione GPU ---
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try: # Usa try-except per maggiore robustezza
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"💻 Utilizzo della GPU: {physical_devices[0]}")
    except RuntimeError as e:
        # Memory growth deve essere impostato all'avvio
        print(f"Errore nell'impostare memory growth: {e}")
else:
    print("⚠ GPU non trovata, il modello utilizzerà la CPU.")

# --- Configurazioni principali ---
IMG_SIZE = (384, 384)
BATCH_SIZE = 24
EPOCHS = 50
CSV_PATH = "dati_csv/ssim_dataset.csv"
MODEL_PATH = "modelli_keras/no_reference_mobilenetv2.keras"
BASE_MODEL_CLASS = MobileNetV2
PREPROCESS_FUNCTION = mobilenet_preprocess_input

print(f"--- Addestramento Modello: {MODEL_CHOICE} ---")
print(f"Percorso salvataggio: {MODEL_PATH}")

# --- Caricamento CSV ---
try:
    df = pd.read_csv(CSV_PATH)
    # Assicurati che le colonne esistano
    if 'image_path' not in df.columns or 'score' not in df.columns:
         raise ValueError("Il file CSV deve contenere le colonne 'image_path' e 'score'")
    # Gestione opzionale di percorsi non esistenti nel CSV
    df = df[df['image_path'].apply(lambda x: os.path.exists(str(x)))]
    if df.empty:
        raise ValueError("Nessun percorso immagine valido trovato nel CSV dopo la verifica dell'esistenza.")
    print(f"Caricate {len(df)} righe valide dal CSV.")
except FileNotFoundError:
    print(f"Errore: File CSV non trovato a '{CSV_PATH}'")
    sys.exit(1) # Esce dallo script
except ValueError as e:
    print(f"Errore nel caricamento CSV: {e}")
    sys.exit(1)


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Data Loading e Preprocessing ---
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
    images_preprocessed = PREPROCESS_FUNCTION(images_np) # Applica es. mobilenet_preprocess_input all'intero array

    return images_preprocessed, np.array(scores), valid_indices

print("Caricamento e preprocessing dati di addestramento...")
X_train, y_train, train_indices = load_data(train_df)
print("Caricamento e preprocessing dati di validazione...")
X_val, y_val, val_indices = load_data(val_df)

# Verifica se sono stati caricati dati
if X_train.size == 0 or X_val.size == 0:
    print("Errore: Nessun dato valido caricato per training o validazione. Verifica i percorsi nel CSV e le immagini.")
    sys.exit(1)

print(f"Dimensione X_train: {X_train.shape}, Dimensione y_train: {y_train.shape}")
print(f"Dimensione X_val: {X_val.shape}, Dimensione y_val: {y_val.shape}")

# --- Pesi dinamici per i sample ---
# Assicurati che y_train non sia vuoto prima di calcolare i pesi
if y_train.size > 0:
   # Applica la pesatura solo ai campioni effettivamente caricati
   # Bisogna recuperare gli score originali dagli indici validi per calcolare i pesi corretti
   y_train_original_scores = train_df.loc[train_indices, 'score'].values
   sample_weights = 1.0 + (1.0 - y_train_original_scores) * 4.0
else:
   sample_weights = None # Nessun peso se non ci sono dati

# --- Architettura Modello ---
# Crea il modello base scelto (MobileNetV2 o ResNet50V2)
base_model = BASE_MODEL_CLASS(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base_model.trainable = True # Fine-tuning attivo (considera di iniziare con False per le prime epoche)

inputs = layers.Input(shape=(*IMG_SIZE, 3))
# Applica il preprocessing specifico *come layer* se non fatto prima
# x = PREPROCESS_FUNCTION(inputs) # Alternativa: fare il preprocess qui
# x = base_model(x, training=False) # Passa l'input preprocessato se fatto qui
x = base_model(inputs, training=True) # Passa gli input già preprocessati
x = layers.GlobalAveragePooling2D()(x)
# Mantieni gli stessi layer Dense per confronto diretto iniziale
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation="relu")(x)
# L'output Sigmoid è appropriato se gli score sono normalizzati tra 0 e 1 (come SSIM)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, output)

# --- Compilazione ---
# Considera un learning rate leggermente diverso se necessario, ma 1e-5 è un buon punto di partenza per fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss="mae", metrics=["mse", "mae"])
model.summary()

# --- Training del modello ---
print("\n--- Inizio Training ---")
# Definisci i callback
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1), # Aumentata pazienza
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1) # Riduci LR se stalla
]

# Assicurati che X_train e sample_weights abbiano la stessa dimensione della prima dim
if sample_weights is not None and len(sample_weights) != X_train.shape[0]:
    print(f"Attenzione: Mismatch dimensioni tra X_train ({X_train.shape[0]}) e sample_weights ({len(sample_weights)}).")
    print("I pesi non verranno utilizzati.")
    sample_weights = None # Disabilita i pesi se c'è un mismatch

history = model.fit(
    X_train, y_train,
    sample_weight=sample_weights if sample_weights is not None else None,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

try:
    model.save(MODEL_PATH)
    print(f"✅ Modello {MODEL_CHOICE} salvato in: {MODEL_PATH}")
except Exception as e:
    print(f"Errore durante il salvataggio del modello: {e}")

print(f"--- Fine Addestramento Modello: {MODEL_CHOICE} ---")
