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

# --- CONFIG ---
IMG_SIZE = (300, 300)
BATCH_SIZE = 12
EPOCHS = 30
CSV_PATH = "/content/drive/MyDrive/approccio4/no_reference_dataset.csv"
MODEL_PATH = "/content/drive/MyDrive/approccio4/no_reference_efficientnet.keras"

# --- Caricamento CSV ---
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def load_img(path):
    img = cv2.imread("/content/drive/MyDrive/approccio4/"+ path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

def load_data(df):
    images = []
    scores = []

    for _, row in df.iterrows():
        img = load_img(row["image_path"])
        score = row["score"]
        images.append(img)
        scores.append(score)

    return np.array(images), np.array(scores)

X_train, y_train = load_data(train_df)
X_val, y_val = load_data(val_df)

# --- Pesi dinamici per i sample ---
sample_weights = 1.0 + (1.0 - y_train) * 4.0

# --- Architettura EfficientNetB3 ---
base_model = EfficientNetB3(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base_model.trainable = True

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, output)

model.compile(optimizer=Adam(1e-5), loss="mae", metrics=["mse", "mae"])
model.summary()

# --- Training ---
model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# --- Salvataggio ---
model.save(MODEL_PATH)
print(f"✅ Modello EfficientNetB3 salvato in: {MODEL_PATH}")
