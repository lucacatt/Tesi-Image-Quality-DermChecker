import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
CSV_PATH = "ssim_dataset.csv"
MODEL_PATH = "siamese_quality_model.keras"

# --- 1. Caricamento dati ---
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

def load_data(df):
    originals = []
    degraded = []
    scores = []

    for _, row in df.iterrows():
        orig = preprocess(row["original"])
        deg = preprocess(row["degraded"])
        score = row["score"]

        originals.append(orig)
        degraded.append(deg)
        scores.append(score)

    return [np.array(originals), np.array(degraded)], np.array(scores)

X_train, y_train = load_data(train_df)
X_val, y_val = load_data(val_df)

# --- 2. Costruzione modello Siamese ---
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base_model.trainable = True  # ✅ Fine-tuning attivo

def build_branch():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    return models.Model(inputs, x)

input_1 = layers.Input(shape=(*IMG_SIZE, 3), name="original")
input_2 = layers.Input(shape=(*IMG_SIZE, 3), name="degraded")

branch = build_branch()
feat_1 = branch(input_1)
feat_2 = branch(input_2)

# ✅ Differenza assoluta serializzabile (senza Lambda)
diff_1 = layers.ReLU()(layers.Subtract()([feat_1, feat_2]))
diff_2 = layers.ReLU()(layers.Subtract()([feat_2, feat_1]))
abs_diff = layers.Add()([diff_1, diff_2])

# --- Concatenazione e regressore ---
merged = layers.Concatenate()([feat_1, feat_2, abs_diff])
x = layers.Dense(128, activation="relu")(merged)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=[input_1, input_2], outputs=output)

# --- 3. Compilazione ---
model.compile(
    optimizer=Adam(1e-5),
    loss="mse",
    metrics=["mae"]
)
model.summary()

# --- 4. Training ---
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# --- 5. Salvataggio compatibile (nessun Lambda) ---
model.save(MODEL_PATH)
print(f"✅ Modello salvato in: {MODEL_PATH}")
