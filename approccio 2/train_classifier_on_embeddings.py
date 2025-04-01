# train_classifier_on_embeddings.py
import numpy as np
from tensorflow.keras import layers, models

# Carica dati
X = np.load("embeddings.npy")
y = np.load("labels.npy")

# Modello semplice
model = models.Sequential([
    layers.Input(shape=(128,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2)

model.save("quality_classifier_on_embeddings.h5")
print("Classificatore salvato.")
