import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

def apply_random_degradation(image):
    choices = ['motion_blur', 'gaussian_blur', 'brightness', 'quality', 'contrast', 'colorfulness', 'noisiness', 'chromatic_aberration', 'pixelation', 'color_cast']
    num_degradations = random.randint(1, 5)
    chosen_degradations = random.sample(choices, num_degradations)

    degraded_image = image

    for choice in chosen_degradations:
        if choice == 'motion_blur':
            print('motion blur applicato')
            kernel_size = random.randint(20, 80)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur /= kernel_size
            image_np = degraded_image.numpy()
            image_np = cv2.filter2D(image_np, -1, kernel_motion_blur)
            degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        elif choice == 'gaussian_blur':
            print('gaussian blur applicato')
            blur_amount = random.randint(8, 32) * 2 + 1
            image_np = degraded_image.numpy()
            image_np = cv2.GaussianBlur(image_np, (blur_amount, blur_amount), 0)
            degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        elif choice == 'brightness':
            print('brightness applicato')
            delta = random.uniform(-0.4, 0.4) 
            degraded_image = tf.image.adjust_brightness(degraded_image, delta=delta)

        elif choice == 'quality':
            print('quality applicato')
            degraded_image = tf.cast(degraded_image * 255, tf.uint8)
            quality = random.randint(5, 15)
            degraded_image = tf.io.encode_jpeg(degraded_image, quality = quality)
            degraded_image = tf.io.decode_jpeg(degraded_image, channels=3)
            degraded_image = tf.image.convert_image_dtype(degraded_image, tf.float32)

        elif choice == 'contrast':
            print('contrast applicato')
            contrast_factor = random.uniform(0.5, 2.5)
            degraded_image = tf.image.adjust_contrast(degraded_image, contrast_factor=contrast_factor)

        elif choice == 'colorfulness':
            print('colorfulness applicato')
            saturation_factor = random.uniform(0.2, 5.8)
            degraded_image = tf.image.adjust_saturation(degraded_image, saturation_factor=saturation_factor)

        elif choice == 'noisiness':
            print('noisiness applicato')
            noise_type = random.choice(['gaussian', 'salt_and_pepper', 'complex'])
            if noise_type == 'gaussian':
                noise = np.random.normal(0, random.uniform(0.10, 0.25), degraded_image.shape).astype(np.float32) 
                image_np = degraded_image.numpy() + noise
                degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)
            elif noise_type == 'salt_and_pepper':
                s_vs_p = 0.5
                amount = random.uniform(0.09, 0.2) 
                out = np.copy(degraded_image.numpy())
                num_salt = np.ceil(amount * degraded_image.numpy().size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in degraded_image.shape]
                out[tuple(coords)] = 1.0
                num_pepper = np.ceil(amount* degraded_image.numpy().size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in degraded_image.shape]
                out[tuple(coords)] = 0.0
                degraded_image = tf.convert_to_tensor(out, dtype=tf.float32)
            elif noise_type == 'complex':
                gaussian_noise = np.random.normal(0, random.uniform(0.10, 0.25), degraded_image.shape).astype(np.float32)
                uniform_noise = np.random.uniform(-0.45, 0.45, degraded_image.shape).astype(np.float32)
                noise = gaussian_noise + uniform_noise
                image_np = degraded_image.numpy() + noise
                degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)

        elif choice == 'chromatic_aberration':
            print('chromatic_aberration applicato')
            offset = random.randint(20, 24)
            image_np = degraded_image.numpy()
            if random.random() > 0.5:
                image_np[:, :, 0] = np.roll(image_np[:, :, 0], offset, axis=1)
                image_np[:, :, 2] = np.roll(image_np[:, :, 2], -offset, axis=1)
            else:
                image_np[:, :, 0] = np.roll(image_np[:, :, 0], -offset, axis=1)
                image_np[:, :, 2] = np.roll(image_np[:, :, 2], offset, axis=1)
            degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)
            degraded_image = tf.clip_by_value(degraded_image, 0.0, 1.0)

        elif choice == 'pixelation':
            print('pixelation applicato')
            scale_factor = random.uniform(0.05, 0.15)
            new_height = int(degraded_image.shape[0] * scale_factor)
            new_width = int(degraded_image.shape[1] * scale_factor)
            image_np = degraded_image.numpy()
            image_small = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image_resized = cv2.resize(image_small, (degraded_image.shape[1], degraded_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            degraded_image = tf.convert_to_tensor(image_resized, dtype=tf.float32)

        elif choice == 'color_cast':
            print('color_cast applicato')
            color_factor = np.array([1.0 + random.uniform(-0.35, 0.35),
                                     1.0 + random.uniform(-0.35, 0.35),
                                     1.0 + random.uniform(-0.35, 0.35)])
            image_np = degraded_image.numpy() * color_factor
            degraded_image = tf.convert_to_tensor(image_np, dtype=tf.float32)
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