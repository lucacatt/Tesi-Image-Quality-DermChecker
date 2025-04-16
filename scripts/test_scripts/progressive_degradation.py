import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt # Per visualizzare i risultati

# --- Funzione Specifica per la Pixelation ---
def apply_pixelation(image_np, downscale_factor):
    """
    Applica l'effetto pixelation a un'immagine NumPy ridimensionandola.

    Args:
        image_np (np.ndarray): L'immagine di input (formato HWC, float32, range 0-1).
        downscale_factor (float): Fattore per ridurre le dimensioni (es. 0.1 per 10%).
                                  Deve essere > 0 e <= 1. Un valore di 1 non applica pixelation.

    Returns:
        np.ndarray: L'immagine pixelata (formato HWC, float32, range 0-1).
    """
    if not (0 < downscale_factor <= 1):
        print(f"Warning: downscale_factor {downscale_factor} non valido. Deve essere tra (0, 1]. Ritorno immagine originale.")
        return image_np
    if downscale_factor == 1:
        # Nessuna pixelation se il fattore è 1
        return image_np

    original_height, original_width = image_np.shape[:2]

    # Calcola le nuove dimensioni, assicurandosi che siano almeno 1x1
    new_width = max(1, int(original_width * downscale_factor))
    new_height = max(1, int(original_height * downscale_factor))

    print(f"  Applicando Pixelation: ridimensionamento a ({new_width}x{new_height}) e poi a ({original_width}x{original_height})")

    # 1. Riduci dimensioni (downscale) - INTER_LINEAR è una buona scelta per mediare
    try:
        small_image = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
         print(f"Errore durante il downscale in pixelation: {e}. Ritorno immagine originale.")
         return image_np

    # 2. Riporta alle dimensioni originali (upscale) usando INTER_NEAREST per l'effetto pixelato
    try:
        pixelated_image = cv2.resize(small_image, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"Errore durante l'upscale in pixelation: {e}. Ritorno immagine originale.")
        # Potrebbe essere utile restituire l'immagine small_image o l'originale
        return image_np


    # Assicura che i valori rimangano nel range [0, 1]
    pixelated_image = np.clip(pixelated_image, 0.0, 1.0)
    return pixelated_image

# --- Parametri ---
IMG_SIZE = (384, 384) # Assicurati che sia la stessa dimensione usata per addestrare il modello
MODEL_PATH = "modelli_keras/no_reference_resnet50v2.keras"
IMAGE_PATH = "test_degraded/31_001.jpg" # Assicurati che il percorso sia corretto

# Definisci i livelli di pixelation (fattori di downscale)
# Un fattore più piccolo significa più pixelation (immagine intermedia più piccola)
# 1.0 significa nessuna pixelation (immagine originale)
PIXELATION_FACTORS = [1.0, 0.5, 0.25, 0.15, 0.1, 0.05] # Step 0 (originale), Step 1, ...

# --- Caricamento Modello e Immagine ---
print(f"Caricamento modello da: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Modello caricato con successo.")
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    exit()

def load_img(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Errore: Impossibile caricare l'immagine da {path}")
        return None
    # Ridimensiona all'IMG_SIZE richiesto dal modello *prima* di qualsiasi degradazione
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Converti subito in float32 [0, 1]
    return img.astype("float32") / 255.0

print(f"Caricamento immagine originale: {IMAGE_PATH}")
original_img_np = load_img(IMAGE_PATH)

if original_img_np is not None:
    results = [] # Lista per salvare i risultati (step, factor, score, image, description)

    # --- Ciclo di Pixelation Progressiva ---
    print("\n--- Inizio Test Pixelation Progressiva ---")

    for i, factor in enumerate(PIXELATION_FACTORS):
        print(f"\n--- Step {i} ---")

        current_image_np = None
        description = ""
        if factor >= 1.0: # Considera 1.0 come l'originale
            print(f"  Analisi Immagine Originale (Fattore = {factor:.2f})")
            current_image_np = original_img_np
            description = f"Originale (F={factor:.2f})"
        else:
            # Applica la pixelation all'immagine *originale*
            current_image_np = apply_pixelation(original_img_np, factor)
            description = f"Pixelation F={factor:.2f}"


        # Prepara l'immagine per il modello (aggiungi batch dimension)
        img_batch = np.expand_dims(current_image_np, axis=0)

        # Ottieni il punteggio dal modello
        score = model.predict(img_batch)[0][0]
        print(f"📊 Punteggio per {description}: {score:.4f}")

        # Salva i risultati
        results.append({
            'step': i,
            'factor': factor, # Salva il fattore usato
            'score': score,
            'image': current_image_np,
            'description': description
        })

    # --- Visualizzazione Risultati ---
    print("\n--- Visualizzazione Risultati ---")
    num_images_to_show = len(results)
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(num_images_to_show * 3, 3.5))

    if num_images_to_show == 1:
        axes = [axes] # Rendi axes iterabile anche se c'è solo un'immagine

    for i, res in enumerate(results):
        ax = axes[i]
        ax.imshow(res['image'])
        # Usa la descrizione salvata per il titolo
        title = f"Step {res['step']}\n{res['description']}\nScore: {res['score']:.3f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Pixelation Progressiva e Punteggio", y=1.03) # Titolo generale
    plt.show()

    # --- Riassunto Punteggi ---
    print("\n--- Riassunto Punteggi ---")
    print("Step | Fattore Downscale | Score")
    print("-----|-------------------|-------")
    for res in results:
        print(f"{res['step']:<4} | {res['factor']:<17.2f} | {res['score']:.4f}")

else:
    print("Impossibile procedere senza l'immagine originale.")