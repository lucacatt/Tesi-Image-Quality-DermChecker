import tensorflow as tf
import os
import random
import numpy as np
import shutil
import cv2
from tensorflow.keras import regularizers

def apply_random_degradation(image):
    # se arriva un tf.Tensor, lo converte in numpy
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    degraded_image = image

    choices = [
        'motion_blur', 'gaussian_blur', 'brightness', 'quality',
        'contrast', 'colorfulness', 'noisiness', 'chromatic_aberration',
        'pixelation', 'color_cast'
    ]

    num_degradations = random.randint(1, 5)
    chosen_degradations = random.sample(choices, num_degradations)

    for choice in chosen_degradations:
        if choice == 'motion_blur':
            print('motion blur applicato')
            kernel_size = random.randint(20, 60)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur /= kernel_size
            degraded_image = cv2.filter2D(degraded_image, -1, kernel_motion_blur)

        elif choice == 'gaussian_blur':
            print('gaussian blur applicato')
            blur_amount = random.randint(3, 15) * 2 + 1
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
            try:
                image_small = cv2.resize(degraded_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                degraded_image = cv2.resize(image_small, (degraded_image.shape[1], degraded_image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print('errore pixelation non applicato')

        elif choice == 'color_cast':
            print('color_cast applicato')
            color_factor = np.array([
                1.0 + random.uniform(-0.3, 0.3),
                1.0 + random.uniform(-0.3, 0.3),
                1.0 + random.uniform(-0.3, 0.3)
            ], dtype=np.float32)
            degraded_image = degraded_image * color_factor

        degraded_image = np.clip(degraded_image, 0.0, 1.0)

    degraded_image = tf.convert_to_tensor(degraded_image, dtype=tf.float32)
    return degraded_image, num_degradations

class ImageDataset(tf.keras.utils.Sequence):
    def __init__(self, sharp_dir, degraded_dir, batch_size=32, img_size=(224, 224), test_split=0.2, shuffle=True):
        self.sharp_dir = sharp_dir
        self.degraded_dir = degraded_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.test_split = test_split
        self.shuffle = shuffle

        # Ottieni i percorsi delle immagini dalle due cartelle
        sharp_images = self.get_image_paths(sharp_dir)
        degraded_images = self.get_image_paths(degraded_dir)

        # Abbina le immagini sharp con le rispettive immagini degraded (presupponendo che abbiano lo stesso nome)
        self.image_paths = [(sharp_img, degraded_img) for sharp_img, degraded_img in zip(sharp_images, degraded_images)]

        # Dividi i dati in training e test
        self.train_paths, self.test_paths = self.split_data()
        self.on_epoch_end()

    def get_image_paths(self, dir_path):
        image_paths = []
        for subdir, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(subdir, file))
        return image_paths

    def split_data(self):
        # Mescola le immagini in modo casuale
        random.shuffle(self.image_paths)

        # Calcola l'indice per la divisione del set di addestramento e di test
        split_idx = int(len(self.image_paths) * (1 - self.test_split))
        train_paths = self.image_paths[:split_idx]
        test_paths = self.image_paths[split_idx:]

        return train_paths, test_paths

    def __getitem__(self, index):
        batch_paths = self.train_paths[index * self.batch_size: (index + 1) * self.batch_size]
        sharp_images_batch = []
        degraded_images_batch = []
        labels = []

        for sharp_path, degraded_path in batch_paths:
            # Carica l'immagine sharp e degraded
            sharp_image = tf.keras.utils.load_img(sharp_path, target_size=self.img_size)
            sharp_image = tf.keras.utils.img_to_array(sharp_image) / 255.0
            sharp_image = tf.convert_to_tensor(sharp_image, dtype=tf.float32)

            degraded_image = tf.keras.utils.load_img(degraded_path, target_size=self.img_size)
            degraded_image = tf.keras.utils.img_to_array(degraded_image) / 255.0
            degraded_image = tf.convert_to_tensor(degraded_image, dtype=tf.float32)

            # Aggiungi le immagini alle liste separate
            sharp_images_batch.append(sharp_image)
            degraded_images_batch.append(degraded_image)
            if "sharp" in sharp_path:
                labels.append(1)  # 1 per immagini sharp
            else:
                labels.append(0)  # 0 per immagini degraded

        # Convert lists of tensors to tensors
        sharp_images_batch = tf.stack(sharp_images_batch)
        degraded_images_batch = tf.stack(degraded_images_batch)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32) # Assuming labels are integers

        return (sharp_images_batch, degraded_images_batch), labels # <--- Return a TUPLE for inputs

    def __len__(self):
        return int(np.floor(len(self.train_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            # Mescola i dati dopo ogni epoca
            random.shuffle(self.train_paths)

def create_folders_and_split_dataset(root_dir):
    # Crea le cartelle sharp e degraded se non esistono
    sharp_folder = "sharp"
    degraded_folder = "degraded"

    if not os.path.exists(sharp_folder):
        os.makedirs(sharp_folder)
    if not os.path.exists(degraded_folder):
        os.makedirs(degraded_folder)

    # Trova tutte le immagini in tutte le sottocartelle
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(subdir, file))

    # Calcola l'80% delle immagini per il training
    train_set_size = int(len(image_paths) * 0.8)
    train_images = random.sample(image_paths, train_set_size)  # Seleziona casualmente l'80% delle immagini per il training

    # Copia l'80% delle immagini in 'sharp' (queste saranno le immagini nitide)
    for img_path in train_images:
        shutil.copy(img_path, sharp_folder)

    # Applica un degrado a tutte le immagini selezionate per l'80% e salvale in 'degraded'
    for img_path in train_images:
        image = tf.keras.utils.load_img(img_path)
        image = tf.keras.utils.img_to_array(image) / 255.0
        degraded_image, num_degradations = apply_random_degradation(image)
        degraded_image = tf.image.convert_image_dtype(degraded_image, dtype=tf.uint8)
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        new_name = f"{name}{ext}"
        save_path = os.path.join(degraded_folder, new_name)
        tf.keras.preprocessing.image.save_img(save_path, degraded_image)

    return train_images

def create_model(input_shape=(224, 224, 3)):
    # Definiamo i due input separati per l'immagine originale e quella degradata
    input_sharp = tf.keras.Input(shape=input_shape, name="sharp_image")
    input_degraded = tf.keras.Input(shape=input_shape, name="degraded_image")

    # Concatenazione delle due immagini lungo l'asse dei canali (axis=-1)
    concatenated = tf.keras.layers.Concatenate(axis=-1)([input_sharp, input_degraded])

    # Prima convoluzione
    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)
    )(concatenated)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Seconda convoluzione
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Terza convoluzione
    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)

    # Strato denso
    x = tf.keras.layers.Dense(
        128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Uscita
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Crea il modello
    model = tf.keras.Model(inputs=[input_sharp, input_degraded], outputs=output)

    return model

if __name__ == "__main__":
    root_dir = "L:\\Tesi Image Quality DermChecker\\Tesi-Image-Quality-DermChecker"

    # 1. Creazione delle cartelle di test e training
    train_images = create_folders_and_split_dataset(root_dir)

    # # 2. Crea il dataset di training
    # train_dataset = ImageDataset(os.path.join(root_dir, 'sharp'), os.path.join(root_dir, 'degraded'), batch_size=32, img_size=(224, 224))

    # # 3. Crea e compila il modello
    # model = create_model()
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )

    # # 4. Addestramento (senza validation)
    # model.fit(
    #     train_dataset,
    #     epochs=30
    # )

    # # 5. Salvataggio del modello
    # model.save('modello.h5')