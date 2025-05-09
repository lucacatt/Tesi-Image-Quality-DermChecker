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

if __name__ == "__main__":
    root_dir = "L:\\Tesi Image Quality DermChecker\\Tesi-Image-Quality-DermChecker"

    train_images = create_folders_and_split_dataset(root_dir)
