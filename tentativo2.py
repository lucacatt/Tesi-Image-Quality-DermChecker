import tensorflow as tf
import os
import random
import numpy as np
import shutil
import cv2
from tensorflow.keras import regularizers

################################################################################
# 1) FUNZIONE PER APPLICARE LA DEGRADAZIONE
################################################################################
def apply_random_degradation(image):
    # Se arriva un tf.Tensor, lo convertiamo subito in numpy
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    degraded_image = image

    # Possibili degradazioni
    choices = [
        'motion_blur', 'gaussian_blur', 'brightness', 'quality',
        'contrast', 'colorfulness', 'noisiness', 'chromatic_aberration',
        'pixelation', 'color_cast'
    ]

    # Scegli da 1 a 5 degradazioni casuali da applicare
    num_degradations = random.randint(1, 5)
    chosen_degradations = random.sample(choices, num_degradations)

    for choice in chosen_degradations:
        if choice == 'motion_blur':
            print('motion blur applicato')
            kernel_size = random.randint(20, 60)  # ridotto l'intervallo
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur /= kernel_size
            degraded_image = cv2.filter2D(degraded_image, -1, kernel_motion_blur)

        elif choice == 'gaussian_blur':
            print('gaussian blur applicato')
            blur_amount = random.randint(3, 15) * 2 + 1  # blur non troppo estremo
            degraded_image = cv2.GaussianBlur(degraded_image, (blur_amount, blur_amount), 0)

        elif choice == 'brightness':
            print('brightness applicato')
            delta = random.uniform(-0.3, 0.3)
            tmp_tensor = tf.convert_to_tensor(degraded_image, dtype=tf.float32)
            tmp_tensor = tf.image.adjust_brightness(tmp_tensor, delta)
            degraded_image = tmp_tensor.numpy()

        elif choice == 'quality':
            print('quality applicato')
            tmp_tensor = tf.convert_to_tensor(degraded_image * 255, dtype=tf.uint8)
            quality = random.randint(5, 25)
            tmp_tensor = tf.io.encode_jpeg(tmp_tensor, quality=quality)
            tmp_tensor = tf.io.decode_jpeg(tmp_tensor, channels=3)
            tmp_tensor = tf.image.convert_image_dtype(tmp_tensor, tf.float32)
            degraded_image = tmp_tensor.numpy()

        elif choice == 'contrast':
            print('contrast applicato')
            contrast_factor = random.uniform(0.5, 2.5)
            tmp_tensor = tf.convert_to_tensor(degraded_image, dtype=tf.float32)
            tmp_tensor = tf.image.adjust_contrast(tmp_tensor, contrast_factor)
            degraded_image = tmp_tensor.numpy()

        elif choice == 'colorfulness':
            print('colorfulness applicato')
            saturation_factor = random.uniform(0.3, 4.0)
            tmp_tensor = tf.convert_to_tensor(degraded_image, dtype=tf.float32)
            tmp_tensor = tf.image.adjust_saturation(tmp_tensor, saturation_factor)
            degraded_image = tmp_tensor.numpy()

        elif choice == 'noisiness':
            print('noisiness applicato')
            noise_type = random.choice(['gaussian', 'salt_and_pepper', 'complex'])
            if noise_type == 'gaussian':
                noise = np.random.normal(0, random.uniform(0.05, 0.2), degraded_image.shape).astype(np.float32)
                degraded_image = degraded_image + noise
            elif noise_type == 'salt_and_pepper':
                s_vs_p = 0.5
                amount = random.uniform(0.05, 0.15)
                out = np.copy(degraded_image)
                num_salt = np.ceil(amount * degraded_image.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in degraded_image.shape]
                out[tuple(coords)] = 1.0
                num_pepper = np.ceil(amount * degraded_image.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in degraded_image.shape]
                out[tuple(coords)] = 0.0
                degraded_image = out
            elif noise_type == 'complex':
                gaussian_noise = np.random.normal(0, random.uniform(0.05, 0.2), degraded_image.shape).astype(np.float32)
                uniform_noise = np.random.uniform(-0.3, 0.3, degraded_image.shape).astype(np.float32)
                noise = gaussian_noise + uniform_noise
                degraded_image = degraded_image + noise

        elif choice == 'chromatic_aberration':
            print('chromatic_aberration applicato')
            offset = random.randint(5, 15)
            if random.random() > 0.5:
                degraded_image[:, :, 0] = np.roll(degraded_image[:, :, 0], offset, axis=1)
                degraded_image[:, :, 2] = np.roll(degraded_image[:, :, 2], -offset, axis=1)
            else:
                degraded_image[:, :, 0] = np.roll(degraded_image[:, :, 0], -offset, axis=1)
                degraded_image[:, :, 2] = np.roll(degraded_image[:, :, 2], offset, axis=1)

        elif choice == 'pixelation':
            print('pixelation applicato')
            scale_factor = random.uniform(0.05, 0.15)
            new_height = int(degraded_image.shape[0] * scale_factor)
            new_width = int(degraded_image.shape[1] * scale_factor)
            image_small = cv2.resize(degraded_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            degraded_image = cv2.resize(image_small, (degraded_image.shape[1], degraded_image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

        elif choice == 'color_cast':
            print('color_cast applicato')
            color_factor = np.array([
                1.0 + random.uniform(-0.3, 0.3),
                1.0 + random.uniform(-0.3, 0.3),
                1.0 + random.uniform(-0.3, 0.3)
            ], dtype=np.float32)
            degraded_image = degraded_image * color_factor

        # Clip finale a [0,1]
        degraded_image = np.clip(degraded_image, 0.0, 1.0)

    # Converti di nuovo in tf.Tensor
    degraded_image = tf.convert_to_tensor(degraded_image, dtype=tf.float32)
    return degraded_image

################################################################################
# 2) CREAZIONE DATASET DINAMICO
################################################################################
class ImageDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, batch_size=32, img_size=(224, 224), test_split=0.2, shuffle=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.test_split = test_split
        self.shuffle = shuffle
        self.image_paths = self.get_image_paths(root_dir)
        self.train_paths, self.test_paths = self.split_data()
        self.on_epoch_end()

    def get_image_paths(self, root_dir):
        image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(subdir, file))
        return image_paths

    def split_data(self):
        random.shuffle(self.image_paths)
        split_idx = int(len(self.image_paths) * (1 - self.test_split))
        train_paths = self.image_paths[:split_idx]
        test_paths = self.image_paths[split_idx:]
        return train_paths, test_paths

    def __getitem__(self, index):
        batch_paths = self.train_paths[index * self.batch_size: (index + 1) * self.batch_size]
        images = []
        labels = []

        for img_path in batch_paths:
            # Carica l'immagine
            image = tf.keras.utils.load_img(img_path, target_size=self.img_size)
            image = tf.keras.utils.img_to_array(image) / 255.0
            image = tf.convert_to_tensor(image, dtype=tf.float32)

            # 50% delle volte l'immagine resta nitida, altrimenti degradazione
            if random.random() > 0.5:
                degraded_image = image
                label = 1  # nitida
            else:
                degraded_image = apply_random_degradation(image)
                label = 0  # degradata

            images.append(degraded_image)
            labels.append(label)

        return np.array(images), np.array(labels)

    def __len__(self):
        return int(np.floor(len(self.train_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.train_paths)

################################################################################
# 3) FUNZIONE PER CREARE CARTELLE 'sharp' E 'degraded' (TEST SET FISSO)
################################################################################
def create_folders_and_split_dataset(image_paths):
    sharp_folder = "sharp"
    degraded_folder = "degraded"
    if not os.path.exists(sharp_folder):
        os.makedirs(sharp_folder)
    if not os.path.exists(degraded_folder):
        os.makedirs(degraded_folder)

    test_set_size = int(len(image_paths) * 0.2)
    test_images = random.sample(image_paths, test_set_size)
    train_images = [img for img in image_paths if img not in test_images]

    # Copia il 20% delle immagini originali in 'sharp'
    for img_path in test_images:
        shutil.copy(img_path, sharp_folder)

    # Applica un degrado e salva in 'degraded'
    for img_path in test_images:
        image = tf.keras.utils.load_img(img_path)
        image = tf.keras.utils.img_to_array(image) / 255.0
        degraded_image = apply_random_degradation(image)
        degraded_image = tf.image.convert_image_dtype(degraded_image, dtype=tf.uint8)
        save_path = os.path.join(degraded_folder, os.path.basename(img_path))
        tf.keras.preprocessing.image.save_img(save_path, degraded_image)

    return train_images, test_images

################################################################################
# 4) DEFINIZIONE DEL MODELLO CNN CON REGOLARIZZAZIONE E DROPOUT
################################################################################
def create_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=input_shape,
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128, activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

################################################################################
# 5) ISTANZIAZIONE DEL DATASET, SUDDIVISIONE E TRAINING
################################################################################
if __name__ == "__main__":
    root_dir = "L:\\Tesi Image Quality DermChecker\\Tesi-Image-Quality-DermChecker\\"
    
    # Crea un oggetto dataset per scoprire i percorsi
    dataset = ImageDataset(root_dir)
    # Suddivisione in train e test set fisso (cartelle 'sharp' e 'degraded')
    train_paths, test_paths = create_folders_and_split_dataset(dataset.get_image_paths(root_dir))

    # Dataset dinamico per training e test
    train_dataset = ImageDataset(root_dir, batch_size=32, img_size=(224,224))
    test_dataset = ImageDataset(root_dir, batch_size=32, img_size=(224,224), shuffle=False)

    # Crea e compila il modello
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Addestramento (aumenta pure le epoche se serve)
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=30
    )

    # Valutazione finale
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc}")

    # Salvataggio del modello
    model.save('my_model.h5')
