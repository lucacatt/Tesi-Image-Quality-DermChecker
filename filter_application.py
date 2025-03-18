import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

def apply_random_degradation(image):
    choices = ['noisiness']
    #num_degradations = random.randint(1, 3) # Applica da 1 a 3 degradazioni in sequenza
    chosen_degradations = random.sample(choices, 1)

    degraded_image = image

    for choice in chosen_degradations:
        if choice == 'motion_blur':
            kernel_size = random.randint(40, 80)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur /= kernel_size
            image_np = degraded_image.numpy()
            image_np = cv2.filter2D(image_np, -1, kernel_motion_blur)
            degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        elif choice == 'gaussian_blur':
            blur_amount = random.randint(13, 32) * 2 + 1
            image_np = degraded_image.numpy()
            image_np = cv2.GaussianBlur(image_np, (blur_amount, blur_amount), 0)
            degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        elif choice == 'brightness':
            delta = random.uniform(-1.6, 0.6) 
            degraded_image = tf.image.adjust_brightness(degraded_image, delta=delta)

        elif choice == 'quality':
            degraded_image = tf.cast(degraded_image * 255, tf.uint8)
            quality = random.randint(7, 11)
            degraded_image = tf.io.encode_jpeg(degraded_image, quality = quality)
            degraded_image = tf.io.decode_jpeg(degraded_image, channels=3)
            degraded_image = tf.image.convert_image_dtype(degraded_image, tf.float32)

        elif choice == 'contrast':
            contrast_factor = random.uniform(1.5, 2.5)
            degraded_image = tf.image.adjust_contrast(degraded_image, contrast_factor=contrast_factor)

        elif choice == 'colorfulness':
            saturation_factor = random.uniform(1.2, 5.8)
            degraded_image = tf.image.adjust_saturation(degraded_image, saturation_factor=saturation_factor)

        elif choice == 'noisiness':
            noise_type = random.choice(['gaussian'])
            if noise_type == 'gaussian':
                noise = np.random.normal(0, random.uniform(0.10, 0.25), degraded_image.shape).astype(np.float32) 
                image_np = degraded_image.numpy() + noise
                degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        #     elif noise_type == 'salt_and_pepper':
        #         s_vs_p = 0.5
        #         amount = random.uniform(0.005, 0.025) # Rumore sale e pepe ridotto
        #         out = np.copy(degraded_image.numpy())
        #         # Rumore sale
        #         num_salt = np.ceil(amount * degraded_image.numpy().size * s_vs_p)
        #         coords = [np.random.randint(0, i - 1, int(num_salt)) for i in degraded_image.shape]
        #         out[tuple(coords)] = 1.0
        #         # Rumore pepe
        #         num_pepper = np.ceil(amount* degraded_image.numpy().size * (1. - s_vs_p))
        #         coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in degraded_image.shape]
        #         out[tuple(coords)] = 0.0
        #         degraded_image = tf.convert_to_tensor(out, dtype=tf.float32)
        #     elif noise_type == 'complex': # Rumore complesso (esempio: somma di gaussiano e uniforme)
        #         gaussian_noise = np.random.normal(0, random.uniform(0.02, 0.08), degraded_image.shape).astype(np.float32)
        #         uniform_noise = np.random.uniform(-0.03, 0.03, degraded_image.shape).astype(np.float32)
        #         noise = gaussian_noise + uniform_noise
        #         image_np = degraded_image.numpy() + noise
        #         degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)


        # elif choice == 'chromatic_aberration':
        #     offset = random.randint(1, 4)
        #     image_np = degraded_image.numpy()
        #     if random.random() > 0.5:
        #         image_np[:, :, 0] = np.roll(image_np[:, :, 0], offset, axis=1)
        #         image_np[:, :, 2] = np.roll(image_np[:, :, 2], -offset, axis=1)
        #     else:
        #         image_np[:, :, 0] = np.roll(image_np[:, :, 0], -offset, axis=1)
        #         image_np[:, :, 2] = np.roll(image_np[:, :, 2], offset, axis=1)
        #     degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)
        #     degraded_image = tf.clip_by_value(degraded_image, 0.0, 1.0)

        # elif choice == 'pixelation':
        #     scale_factor = random.uniform(0.1, 0.4)
        #     new_height = int(degraded_image.shape[0] * scale_factor)
        #     new_width = int(degraded_image.shape[1] * scale_factor)
        #     image_np = degraded_image.numpy()
        #     image_small = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        #     image_resized = cv2.resize(image_small, (degraded_image.shape[1], degraded_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        #     degraded_image = tf.convert_to_tensor(image_resized, dtype=tf.float32)

        # elif choice == 'vignetting':
        #     intensity = random.uniform(0.2, 0.7) # Vignettatura più leggera
        #     radius = random.uniform(0.5, 0.9) # Raggio vignettatura più ampio
        #     image_np = degraded_image.numpy()
        #     height, width = image_np.shape[:2]
        #     x_center, y_center = width // 2, height // 2
        #     x = np.arange(width)
        #     y = np.arange(height)
        #     X, Y = np.meshgrid(x, y)
        #     distance_from_center = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        #     max_distance = np.sqrt((x_center)**2 + (y_center)**2)
        #     vignette_mask = 1 - intensity * (distance_from_center / max_distance)**2
        #     vignette_mask = np.clip(vignette_mask, 1 - intensity, 1)
        #     vignette_mask = np.tile(vignette_mask[:, :, np.newaxis], (1, 1, 3))
        #     image_np = degraded_image.numpy() * vignette_mask
        #     degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        # elif choice == 'color_cast':
        #     color_factor = np.array([1.0 + random.uniform(-0.15, 0.15),
        #                              1.0 + random.uniform(-0.15, 0.15),
        #                              1.0 + random.uniform(-0.15, 0.15)])
        #     image_np = degraded_image.numpy() * color_factor
        #     degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        # elif choice == 'lens_distortion': # Distorsione lente (barilotto o cuscino)
        #     distortion_type = random.choice(['barrel', 'pincushion'])
        #     k = random.uniform(0.1, 0.3) # Intensità distorsione
        #     image_np = degraded_image.numpy()
        #     h, w = image_np.shape[:2]
        #     distCoeff = np.array([k, 0, 0, 0]) # Solo distorsione radiale k1
        #     if distortion_type == 'pincushion':
        #         distCoeff = -distCoeff # Negativo per distorsione a cuscino

        #     # Matrice della camera (approssimativa, centrata)
        #     center = (w/2, h/2)
        #     focal_length = w  # Approssimazione
        #     cameraMatrix = np.array(
        #                      [[focal_length, 0, center[0]],
        #                      [0, focal_length, center[1]],
        #                      [0, 0, 1]], dtype = "double"
        #                      )
        #     image_distorted = cv2.undistort(image_np, cameraMatrix, distCoeff) # Undistort per creare la distorsione

        #     degraded_image = tf.convert_to_tensor(image_distorted, dtype=tf.float32)
        #     degraded_image = tf.clip_by_value(degraded_image, 0.0, 1.0) # Clip dopo la distorsione

        # elif choice == 'complex_noise': # Rumore complesso (esempio 2: rumore Perlin) - richiede libreria `perlin-numpy` (pip install perlin-numpy)
        #     try:
        #         from perlin_numpy import (generate_perlin_noise_2d, generate_fractal_noise_2d)
        #         noise_scale = random.uniform(5, 20) # Scala del rumore Perlin
        #         noise_intensity = random.uniform(0.02, 0.08) # Intensità del rumore Perlin
        #         perlin_noise = generate_fractal_noise_2d((degraded_image.shape[0], degraded_image.shape[1]), (noise_scale, noise_scale), octaves=random.randint(3, 6))
        #         perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise)) # Normalizza a [0, 1]
        #         perlin_noise = (perlin_noise - 0.5) * 2 * noise_intensity # Centra intorno a 0 e scala intensità
        #         noise = np.repeat(perlin_noise[:, :, np.newaxis], 3, axis=2) # Ripeti per 3 canali
        #         image_np = degraded_image.numpy() + noise
        #         degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)
        #     except ImportError:
        #         print("Warning: perlin-numpy not installed, complex_noise skipped.")
        #         pass # Se perlin-numpy non è installato, salta questo effetto


        degraded_image = tf.clip_by_value(degraded_image, 0.0, 1.0)

    return degraded_image

def augmentation_filter(sharp_image):
    degraded_image = apply_random_degradation(sharp_image)
    return sharp_image, degraded_image

image_path = "selection/28_002.jpg"
sharp_image = tf.keras.utils.load_img(image_path)
sharp_image = tf.keras.utils.img_to_array(sharp_image) / 255.0
sharp_image = tf.convert_to_tensor(sharp_image, dtype=tf.float32)

sharp_image, degraded_image = augmentation_filter(sharp_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Immagine Nitida")
plt.imshow(sharp_image.numpy())

plt.subplot(1, 2, 2)
plt.title("Immagine Degradata")
plt.imshow(degraded_image.numpy())
plt.show()