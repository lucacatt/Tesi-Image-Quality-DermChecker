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

def augment_data(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image) 
    return image, label

def create_dataset(X, y, batch_size=32, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda image, label: (tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)), label),
                              num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


train_dataset = create_dataset(X_train, y_train, augment=True)
val_dataset = create_dataset(X_val, y_val, augment=False)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = True 

image_input = Input(shape=(224, 224, 3)) 
x = base_model(image_input, training=False)
x = GlobalAveragePooling2D()(x)

sharpness_metrics = SharpnessMetricsLayer()(image_input)

x = Concatenate()([x, sharpness_metrics])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
classification_output = Dense(1, activation='sigmoid', name="classification")(x)

model = Model(inputs=image_input, outputs=[classification_output, sharpness_metrics]) 

print("Modello intensificato (architettura originale) creato.")
model.summary() 

model = tf.keras.models.load_model("blur_detection_with_metrics.h5", custom_objects={"SharpnessMetricsLayer": SharpnessMetricsLayer})
print("Modello pre-esistente caricato.")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss=['binary_crossentropy', None],
    metrics=[['accuracy'], None] 
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
ckpt = ModelCheckpoint("best_model_retrained_intensified.h5", monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
callbacks = [early_stop, ckpt, reduce_lr]


print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20, 
    callbacks=callbacks
)

model.save("blur_detection_with_metrics_retrained_intensified.h5") 
print("Training completato con modello intensificato (architettura originale) e modello RI-ALLENATO salvato!")