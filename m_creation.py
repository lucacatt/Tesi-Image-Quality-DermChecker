import tensorflow as tf
import os
import random
import numpy as np
import cv2
import gc
import tensorflow.keras.backend as K

K.clear_session()
gc.collect()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Uso della GPU forzato:", gpus[0])
    except RuntimeError as e:
        print(e)

# trasformazioni ulteriori
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
])

def apply_random_degradation(image):
    labels = []
    intensities = []
    choices = ['motion_blur', 'gaussian_blur', 'brightness', 'quality', 'contrast', 
               'colorfulness', 'noisiness', 'chromatic_aberration', 'pixelation', 'color_cast']
    
    num_degradations = random.randint(1, 5)
    chosen_degradations = random.sample(choices, num_degradations)

    degraded_image = image

    for choice in chosen_degradations:
        if choice == 'motion_blur':
            kernel_size = random.randint(20, 80)
            labels.append('motion_blur')
            intensities.append(kernel_size / 80.0)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur /= kernel_size
            image_np = degraded_image.numpy()
            image_np = cv2.filter2D(image_np, -1, kernel_motion_blur)
            degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        elif choice == 'gaussian_blur':
            blur_amount = random.randint(8, 32) * 2 + 1
            labels.append('gaussian_blur')
            intensities.append((blur_amount - 8) / (32 * 2))
            image_np = degraded_image.numpy()
            image_np = cv2.GaussianBlur(image_np, (blur_amount, blur_amount), 0)
            degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        elif choice == 'brightness':
            delta = random.uniform(-0.4, 0.4)
            labels.append('brightness')
            intensities.append((delta + 0.4) / 0.8)
            degraded_image = tf.image.adjust_brightness(degraded_image, delta=delta)

        elif choice == 'contrast':
            contrast_factor = random.uniform(0.5, 2.5)
            labels.append('contrast')
            intensities.append((contrast_factor - 0.5) / 2.0)
            degraded_image = tf.image.adjust_contrast(degraded_image, contrast_factor=contrast_factor)

        elif choice == 'pixelation':
            scale_factor = random.uniform(0.05, 0.15)
            labels.append('pixelation')
            intensities.append((scale_factor - 0.05) / 0.1)
            new_height = int(degraded_image.shape[0] * scale_factor)
            new_width = int(degraded_image.shape[1] * scale_factor)
            image_np = degraded_image.numpy()
            image_small = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image_resized = cv2.resize(image_small, (degraded_image.shape[1], degraded_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            degraded_image = tf.convert_to_tensor(image_resized, dtype=tf.float32)

    degraded_image = tf.clip_by_value(degraded_image, 0.0, 1.0)

    return degraded_image, labels, intensities

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            image = tf.keras.utils.load_img(img_path)
            image = tf.image.resize(image, (256, 256)) / 255.0
            images.append(tf.convert_to_tensor(image, dtype=tf.float32))
            filenames.append(filename)
        except Exception as e:
            print(f"Errore nel caricamento {filename}: {e}")
    return images, filenames

def create_degraded_dataset(original_images):
    dataset = []
    label_mapping = {
        'motion_blur': 0, 'gaussian_blur': 1, 'brightness': 2, 'quality': 3, 'contrast': 4,
        'colorfulness': 5, 'noisiness': 6, 'chromatic_aberration': 7, 'pixelation': 8, 'color_cast': 9
    }
    
    for image in original_images:
        degraded_image, chosen_labels, chosen_intensities = apply_random_degradation(image)
        numeric_labels = [label_mapping[label] for label in chosen_labels]
        dataset.append((image, degraded_image, numeric_labels, chosen_intensities))
    
    return dataset

dataset_path = "selection"
original_images, filenames = load_images_from_folder(dataset_path)
dataset = create_degraded_dataset(original_images)

# creazione dataset
def data_generator():
    for original, degraded, labels, intensities in dataset:
        if len(labels) > 0 and len(intensities) > 0:
            yield (original, degraded), (labels[0], intensities[0]) 

dataset_tf = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        (   #immagine originale e degradata
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
        ),
        (   # classe del filtro e intensità
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
).batch(32).prefetch(tf.data.AUTOTUNE)

# Splitting dataset
dataset_size = sum(1 for _ in dataset_tf)
train_size = int(0.8 * dataset_size)

train_dataset = dataset_tf.take(train_size)
val_dataset = dataset_tf.skip(train_size)

# MobileNetV2 con fine-tuning su due input
base_model = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3), include_top=False)
base_model.trainable = True

# Due input: immagine originale e degradata
input_original = tf.keras.Input(shape=(256, 256, 3), name="original_input")
input_degraded = tf.keras.Input(shape=(256, 256, 3), name="degraded_input")

# Feature extraction con MobileNetV2 su entrambe le immagini
features_original = base_model(input_original, training=True)
features_degraded = base_model(input_degraded, training=True)

# Concatenazione delle feature per confronto
merged_features = tf.keras.layers.Concatenate()([features_original, features_degraded])
x = tf.keras.layers.GlobalAveragePooling2D()(merged_features)

x = tf.keras.layers.Dense(1024, activation='swish')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(512, activation='swish')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(256, activation='swish')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Due output: classe del filtro e intensità
class_output = tf.keras.layers.Dense(10, activation='softmax', name='class_output')(x)
intensity_output = tf.keras.layers.Dense(1, activation='relu', name='intensity_output')(x)

# Creazione del modello con doppio input
model = tf.keras.Model(inputs=[input_original, input_degraded], outputs=[class_output, intensity_output])

# Ottimizzatore learning rate 
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

# Compilazione del modello
model.compile(
    optimizer=optimizer,
    loss={'class_output': 'sparse_categorical_crossentropy', 'intensity_output': 'mse'},
    metrics={'class_output': 'accuracy', 'intensity_output': 'mae'}
)

# Callbacks migliorati
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7
)

# Training del modello
model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[lr_schedule]
)

model.save("blur_detection_with_metrics.h5")
