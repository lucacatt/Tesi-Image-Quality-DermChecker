import numpy as np
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data = np.load("data.npy")
labels = np.load("labels.npy")
metrics = np.load("metrics.npy")

X_train_img, X_val_img, X_train_met, X_val_met, y_train, y_val = train_test_split(
    data, metrics, labels, test_size=0.2, random_state=42, shuffle=True
)

def custom_generator(X_img, X_met, y):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5)
    ])
    for img, met, label in zip(X_img, X_met, y):
        img_uint8 = (img * 255).astype(np.uint8)
        augmented = transform(image=img_uint8)
        aug_img = augmented['image'].astype(np.float32) / 255.0
        yield (aug_img, met), label  # Restituiamo un TUPLO di input

def val_generator(X_img, X_met, y):
    for img, met, label in zip(X_img, X_met, y):
        yield (img, met), label  # Restituiamo un TUPLO di input

batch_size = 32
train_steps = len(X_train_img) // batch_size
val_steps = len(X_val_img) // batch_size

output_signature = (
    (tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32), tf.TensorSpec(shape=(3,), dtype=tf.float32)),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: custom_generator(X_train_img, X_train_met, y_train),
    output_signature=output_signature
).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator(X_val_img, X_val_met, y_val),
    output_signature=output_signature
).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

image_input = Input(shape=(224, 224, 3))
metrics_input = Input(shape=(3,))
x = base_model(image_input, training=False)
x = GlobalAveragePooling2D()(x)
x = Concatenate()([x, metrics_input])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[image_input, metrics_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ckpt = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[early_stop, ckpt]
)

model.save("blur_detection_with_metrics_aug.h5")
print("Training completato!")
