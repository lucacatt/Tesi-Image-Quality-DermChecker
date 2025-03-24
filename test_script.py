import tensorflow as tf
import os
import random
import shutil
import numpy as np
import pandas as pd

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

# 2) Funzione per creare il dataset di test (20% non usato dal train set)
def create_test_set(original_folder, sharp_folder, degraded_folder, test_folder, test_size=0.2):
    # Ottieni tutte le immagini dalla cartella originale e dalle sottocartelle
    all_images = []
    for subdir, _, files in os.walk(original_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Considera solo immagini
                all_images.append(os.path.join(subdir, file))

    # Ottieni le immagini già presenti nelle cartelle sharp e degraded
    sharp_images = [os.path.join(sharp_folder, f) for f in os.listdir(sharp_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    degraded_images = [os.path.join(degraded_folder, f) for f in os.listdir(degraded_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Rimuovi le immagini che sono già state usate per il training
    remaining_images = [img for img in all_images if img not in sharp_images and img not in degraded_images]

    # Calcola il numero di immagini da prendere per il test (20% delle immagini rimanenti)
    test_images = random.sample(remaining_images, int(len(remaining_images) * test_size))

    # Crea la cartella per il test set se non esiste
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Crea le sottocartelle sharp e degraded per il test
    test_sharp_folder = os.path.join(test_folder, 'sharp')
    test_degraded_folder = os.path.join(test_folder, 'degraded')

    if not os.path.exists(test_sharp_folder):
        os.makedirs(test_sharp_folder)

    if not os.path.exists(test_degraded_folder):
        os.makedirs(test_degraded_folder)

    # Copia le immagini di test nelle rispettive cartelle (sharp o degraded)
    for img_path in test_images:
        if 'sharp' in img_path:
            shutil.copy(img_path, test_sharp_folder)
        else:
            shutil.copy(img_path, test_degraded_folder)

    return test_images

# 3) Classe Dataset per caricare il test set
class ImageTestDataset(tf.keras.utils.Sequence):
    def __init__(self, image_paths, batch_size=32, img_size=(224, 224)):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.img_size = img_size

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size: (index + 1) * self.batch_size]
        images = []

        for img_path in batch_paths:
            image = tf.keras.utils.load_img(img_path, target_size=self.img_size)
            image = tf.keras.utils.img_to_array(image) / 255.0
            images.append(image)

        return np.array(images)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

# 4) Carica il modello salvato
def load_model(model_path='modello.h5'):
    model = tf.keras.models.load_model(model_path)
    return model

# 5) Funzione per eseguire le previsioni su ogni immagine e salvare i risultati in un CSV
def predict_and_save(model, test_dataset, output_csv='predictions.csv'):
    predictions = []
    image_paths = test_dataset.image_paths

    for i in range(len(test_dataset)):
        batch_images = test_dataset[i]
        batch_preds = model.predict(batch_images)
        for img_path, pred in zip(image_paths[i * test_dataset.batch_size: (i + 1) * test_dataset.batch_size], batch_preds):
            predictions.append([img_path, pred[0]])

    # Creazione di un DataFrame e salvataggio su CSV
    df = pd.DataFrame(predictions, columns=['Image_Path', 'Prediction'])
    df.to_csv(output_csv, index=False)
    print(f"Risultati salvati in {output_csv}")

# 6) Esegui il processo
if __name__ == "__main__":
    # Percorso alla cartella con tutte le immagini originali
    original_folder = 'images'  # Sostituisci con il percorso corretto
    sharp_folder = 'sharp'  # Cartella con immagini nitide per il training
    degraded_folder = 'degraded'  # Cartella con immagini degradate per il training
    test_folder = 'test'  # Cartella per le immagini di test create

    # Crea il test set (20% delle immagini non usate per il training)
    test_images = create_test_set(original_folder, sharp_folder, degraded_folder, test_folder)

    # Crea il dataset di test
    test_dataset = ImageTestDataset(test_images, batch_size=32, img_size=(224, 224))

    # Carica il modello
    model = load_model('modello.h5')

    # Esegui le previsioni e salva i risultati
    predict_and_save(model, test_dataset, output_csv='predictions.csv')
