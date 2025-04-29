import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt # Per visualizzare i risultati

# --- Funzione Specifica per l'Aggiunta di Rumore Gaussiano ---
def add_gaussian_noise(image_np, std_dev):
    """
    Aggiunge rumore Gaussiano a un'immagine NumPy.

    Args:
        image_np (np.ndarray): L'immagine di input (formato HWC, float32, range 0-1).
        std_dev (float): La deviazione standard della distribuzione Gaussiana per il rumore.
                         Maggiore per più rumore. Deve essere >= 0. Un valore di 0 significa nessun rumore.

    Returns:
        np.ndarray: L'immagine con rumore aggiunto (formato HWC, float32, range 0-1).
    """
    if std_dev < 0:
        print(f"Warning: std_dev {std_dev} non valido. Deve essere >= 0. Ritorno immagine originale.")
        return image_np
    if std_dev == 0:
        # Nessun rumore se la deviazione standard è 0
        return image_np

    print(f"  Aggiungendo Rumore Gaussiano: std_dev={std_dev:.4f}")

    try:
        # Genera rumore Gaussiano con media 0 e deviazione standard specificata
        # La dimensione deve essere la stessa dell'immagine
        noise = np.random.normal(loc=0.0, scale=std_dev, size=image_np.shape).astype(np.float32)

        # Aggiungi il rumore all'immagine
        noisy_image = image_np + noise

        # Assicura che i valori rimangano nel range [0, 1] dopo l'aggiunta del rumore
        noisy_image = np.clip(noisy_image, 0.0, 1.0)

    except Exception as e:
         print(f"Errore durante l'aggiunta del rumore Gaussiano: {e}. Ritorno immagine originale.")
         return image_np

    return noisy_image

# --- Parametri ---
IMG_SIZE = (384, 384) # Assicurati che sia la stessa dimensione usata per addestrare il modello
MODEL_PATH = "modelli_keras/no_reference_resnet50v2.keras"
IMAGE_PATH = "sharp/32_002.jpg" # Assicurati che il percorso sia corretto

# Definisci i livelli di rumore (deviazioni standard del rumore Gaussiano)
# Un valore maggiore significa più rumore
# Un valore di 0 significa nessun rumore (immagine originale)
# Valori tipici per immagini [0,1] sono piccoli, es. tra 0.01 e 0.2
NOISE_STD_DEVS = [0.0, 0.01, 0.03, 0.07, 0.12, 0.2] # Step 0 (originale/no noise), Step 1, ...

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
    # Converti subito in float32 [0, 1] e in RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return img

print(f"Caricamento immagine originale: {IMAGE_PATH}")
original_img_np = load_img(IMAGE_PATH)

if original_img_np is not None:
    results = [] # Lista per salvare i risultati (step, parameter, score, image, description)

    # --- Ciclo di Aggiunta Rumore Progressiva ---
    print("\n--- Inizio Test Rumore Progressivo ---")

    # Modifica il ciclo per usare le deviazioni standard del rumore
    for i, noise_std in enumerate(NOISE_STD_DEVS):
        print(f"\n--- Step {i} ---")

        current_image_np = None
        description = ""
        # Considera noise_std = 0 come l'immagine originale
        if noise_std == 0:
            print(f"  Analisi Immagine Originale (Std Dev = {noise_std:.4f})")
            current_image_np = original_img_np
            description = f"Originale (Std={noise_std:.3f})"
        else:
            # Applica il rumore Gaussiano all'immagine *originale*
            current_image_np = add_gaussian_noise(original_img_np, noise_std)
            description = f"Noise Std={noise_std:.3f}"


        # Prepara l'immagine per il modello (aggiungi batch dimension)
        img_batch = np.expand_dims(current_image_np, axis=0)

        # Ottieni il punteggio dal modello
        score = model.predict(img_batch)[0][0]
        print(f"📊 Punteggio per {description}: {score:.4f}")

        # Salva i risultati
        results.append({
            'step': i,
            'parameter': noise_std, # Salva la deviazione standard usata
            'score': score,
            'image': current_image_np,
            'description': description
        })

    # --- Visualizzazione Risultati ---
    print("\n--- Visualizzazione Risultati ---")
    num_images_to_show = len(results)
    # Aumenta leggermente la dimensione della figura per accomodare titoli più lunghi se necessario
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(num_images_to_show * 3.5, 4))

    # Rendi axes iterabile anche se c'è solo un'immagine
    if num_images_to_show == 1:
        axes = [axes]

    for i, res in enumerate(results):
        ax = axes[i]
        ax.imshow(res['image'])
        # Usa la descrizione salvata per il titolo
        title = f"Step {res['step']}\n{res['description']}\nScore: {res['score']:.3f}"
        ax.set_title(title, fontsize=9) # Riduci la dimensione del font del titolo se sono molti
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Rumore Gaussiano Progressivo e Punteggio", y=1.03) # Titolo generale aggiornato
    plt.show()

    # --- Riassunto Punteggi ---
    print("\n--- Riassunto Punteggi ---")
    # Aggiorna l'intestazione e il formato di stampa
    print("Step | Std Dev | Score")
    print("-----|---------|-------")
    for res in results:
        # Usa la chiave 'parameter' che ora contiene la deviazione standard
        print(f"{res['step']:<4} | {res['parameter']:<7.4f} | {res['score']:.4f}")

else:
    print("Impossibile procedere senza l'immagine originale.")