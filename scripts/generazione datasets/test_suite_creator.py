import os
import cv2
import numpy as np
import random
import tensorflow as tf

# Configurazione cartelle
BASE_DIR = "L:\\Tesi Image Quality DermChecker\\Tesi-Image-Quality-DermChecker\\"  # Cartella principale che contiene `sharp` e `degraded`
SHARP_DIR = os.path.join(BASE_DIR, "sharp")  # Cartella con immagini sharp (80%)
DEGRADED_DIR = os.path.join(BASE_DIR, "degraded")  # Cartella con immagini degradate
TEST_DEGRADED_DIR = os.path.join(BASE_DIR, "test_degraded")  # Nuova cartella per il test set degradato

# Crea la cartella test_degraded se non esiste
if not os.path.exists(TEST_DEGRADED_DIR):
    os.makedirs(TEST_DEGRADED_DIR)

# Funzione per caricare le immagini
def load_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (384, 384))  # Ridimensiona l'immagine
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte in RGB
    return img.astype("float32") / 255.0

# Funzione per applicare una degradazione casuale (inserisci qui la tua funzione di degradazione)
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

# Ottieni tutte le immagini nella cartella principale (ad esclusione di quelle in `sharp`)
all_images = []
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):  # Puoi aggiungere altri formati se necessario
            full_path = os.path.join(root, file)
            if full_path not in [os.path.join(SHARP_DIR, f) for f in os.listdir(SHARP_DIR)]:
                all_images.append(full_path)

# Calcola il numero di immagini da selezionare per il test set (20% delle immagini totali)
num_images_to_select = int(0.2 * len(all_images))

# Seleziona casualmente il 20% delle immagini fuori da `sharp`
selected_test_images = random.sample(all_images, num_images_to_select)

# Mescolamento 50% sharp e 50% degradato
num_images_to_leave_sharp = int(num_images_to_select / 2)  # 50% per le immagini sharp
num_images_to_degrade = num_images_to_select - num_images_to_leave_sharp  # Resto per le immagini degradate

# Seleziona casualmente il 50% delle immagini per essere lasciate come sharp
sharp_images_for_test = random.sample(selected_test_images, num_images_to_leave_sharp)

# Il resto delle immagini selezionate sarà degradato
images_to_degrade = [img for img in selected_test_images if img not in sharp_images_for_test]

# Salva le immagini sharp e degradate nella cartella test_degraded
for image_path in sharp_images_for_test:
    # Carica l'immagine sharp
    image = load_img(image_path)
    
    # Costruisci il percorso per salvare l'immagine sharp (senza modifiche)
    filename = os.path.basename(image_path)
    sharp_image_path = os.path.join(TEST_DEGRADED_DIR, filename)
    
    # Salva l'immagine sharp nella cartella di test
    image_to_save = np.clip(image * 255, 0, 255).astype(np.uint8)  # Riporta nell'intervallo 0-255
    cv2.imwrite(sharp_image_path, cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))
    print(f"Immagine sharp salvata: {sharp_image_path}")

# Salva le immagini degradate
for image_path in images_to_degrade:
    # Carica l'immagine
    image = load_img(image_path)
    
    # Applica la funzione di degradazione
    degraded_image, _ = apply_random_degradation(image)
    
    # Converti l'immagine in un formato che può essere salvato
    degraded_image = np.clip(degraded_image * 255, 0, 255).astype(np.uint8)  # Riporta nell'intervallo 0-255
    
    # Costruisci il percorso per salvare l'immagine degradata
    filename = os.path.basename(image_path)
    degraded_image_path = os.path.join(TEST_DEGRADED_DIR, filename)
    
    # Salva l'immagine degradata
    cv2.imwrite(degraded_image_path, cv2.cvtColor(degraded_image, cv2.COLOR_RGB2BGR))  # Salva in BGR
    print(f"Immagine degradata salvata: {degraded_image_path}")

print("Degradazione completata per il 0.20 delle immagini.")
