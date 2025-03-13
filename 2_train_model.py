import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from SharpnessMetricsLayerClass import SharpnessMetricsLayer

data = np.load("data.npy")
labels = np.load("labels.npy")

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

def create_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(X_train, y_train)
val_dataset = create_dataset(X_val, y_val)



base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=False)
x = GlobalAveragePooling2D()(x)

sharpness_metrics = SharpnessMetricsLayer()(image_input)

x = Concatenate()([x, sharpness_metrics])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
classification_output = Dense(1, activation='sigmoid', name="classification")(x)

model = Model(inputs=image_input, outputs=[classification_output, sharpness_metrics])
model.compile(
    optimizer='adam', 
    loss=['binary_crossentropy', None],  
    metrics=[['accuracy'], None] 
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ckpt = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stop, ckpt]
)

model.save("blur_detection_with_metrics.h5")
print("Training completato con metriche interne!")