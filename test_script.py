import tensorflow as tf
import os
import random
import numpy as np
import shutil
import cv2

# 1) Funzione per applicare degradazioni casuali
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


# 2) Classe Dataset per caricare il test set
class ImageTestDataset(tf.keras.utils.Sequence):
    def __init__(self, image_paths, batch_size=32, img_size=(224, 224)):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.img_size = img_size

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size: (index + 1) * self.batch_size]
        images = []
        labels = []

        for img_path in batch_paths:
            image = tf.keras.utils.load_img(img_path, target_size=self.img_size)
            image = tf.keras.utils.img_to_array(image) / 255.0
            image = tf.convert_to_tensor(image, dtype=tf.float32)

            # Se l'immagine è nella cartella 'sharp', è nitida (label=1)
            # Se è nella cartella 'degraded', è degradata (label=0)
            if 'sharp' in img_path:
                label = 1
            else:
                label = 0

            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))


# 3) Carica il modello salvato
def load_model(model_path='my_model.h5'):
    model = tf.keras.models.load_model(model_path)
    return model


# 4) Funzione per caricare e testare il modello sul test set
def test_model(model, test_dataset):
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    print(f"Test accuracy: {test_acc}")
    return test_loss, test_acc


# 5) Esegui il test del modello
if __name__ == "__main__":
    # Carica le immagini di test dalle cartelle sharp e degraded
    sharp_folder = "sharp"
    degraded_folder = "degraded"

    sharp_images = [os.path.join(sharp_folder, f) for f in os.listdir(sharp_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    degraded_images = [os.path.join(degraded_folder, f) for f in os.listdir(degraded_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Unisci tutte le immagini per il test set
    test_images = sharp_images + degraded_images

    # Crea il dataset di test
    test_dataset = ImageTestDataset(test_images, batch_size=32, img_size=(224, 224))

    # Carica il modello
    model = load_model('my_model.h5')

    # Esegui il test sul modello
    test_model(model, test_dataset)
