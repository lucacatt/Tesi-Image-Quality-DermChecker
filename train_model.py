import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate
from sklearn.model_selection import train_test_split

data = np.load("data.npy")
labels = np.load("labels.npy")
metrics = np.load("metrics.npy")

data = data.astype('float32') / 255.0

X_train_img, X_val_img, X_train_met, X_val_met, y_train, y_val = train_test_split(
    data, metrics, labels, test_size=0.2, random_state=42, shuffle=True
)

image_input = Input(shape=(224, 224, 3))  # Immagini
metrics_input = Input(shape=(3,))  # Laplacian, PSNR, SSIM

x = Conv2D(32, (3,3), activation='relu')(image_input)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)

x = Concatenate()([x, metrics_input])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[image_input, metrics_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit([X_train_img, X_train_met], y_train, epochs=10, batch_size=32, validation_data=([X_val_img, X_val_met], y_val))

model.save("blur_detection_with_metrics.h5")
print("✅ Modello salvato: blur_detection_with_metrics.h5")
