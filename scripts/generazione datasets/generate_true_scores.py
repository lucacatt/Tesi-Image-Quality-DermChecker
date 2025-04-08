import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import time

# --- Configurazione Utente ---
# Cartella contenente TUTTE le immagini originali (potenzialmente in sottocartelle)
ORIGINAL_DATASET_ROOT = "dataset" # MODIFICA con il percorso corretto

# Cartella contenente le immagini di TEST da valutare (possono essere sharp o degraded)
TEST_IMAGES_DIR = "test_degraded" # MODIFICA se il nome è diverso

# Dimensioni a cui ridimensionare le immagini per il confronto SSIM
IMG_SIZE = (384, 384)

# Percorso del file CSV di output
OUTPUT_CSV = "dati_csv/true_scores.csv" # Nome file suggerito

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff') # Estensioni file immagine valide
# --- Fine Configurazione Utente ---

def load_and_prep_image_for_ssim(path, img_size):
    """Carica, ridimensiona, converte in grayscale e restituisce come uint8."""
    try:
        # Leggi l'immagine
        img = cv2.imread(path)
        if img is None:
            print(f"  Attenzione: Impossibile leggere l'immagine {os.path.basename(path)}. Saltata.")
            return None

        # Ridimensiona
        img_resized = cv2.resize(img, img_size)

        # Converti in grayscale (SSIM è definito su immagini a singolo canale)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # SSIM in skimage funziona bene con uint8 (range 0-255)
        # Non è necessaria normalizzazione a float se specifichiamo data_range
        return img_gray # Restituisce l'array uint8

    except Exception as e:
        print(f"  Errore durante caricamento/preparazione di {os.path.basename(path)}: {e}")
        return None

# --- Script Principale ---

# 1. Indicizza tutte le immagini originali nel dataset principale
print(f"🔎 Inizio indicizzazione immagini originali in: {ORIGINAL_DATASET_ROOT}")
start_time = time.time()
original_image_paths = {} # Dizionario: {nome_file: percorso_completo}
count_indexed = 0
duplicates_found = []

for dirpath, _, filenames in os.walk(ORIGINAL_DATASET_ROOT):
    for filename in filenames:
        if filename.lower().endswith(VALID_EXTENSIONS):
            full_path = os.path.join(dirpath, filename)
            # Controlla duplicati (opzionale ma utile)
            if filename in original_image_paths:
                duplicates_found.append(filename)
                # Decidi come gestire i duplicati: sovrascrivi (comportamento di default)
                # o mantieni il primo, o salta. Qui sovrascriviamo.
                # print(f"  Attenzione: Trovato duplicato per {filename}. Verrà usato il percorso: {full_path}")
            original_image_paths[filename] = full_path
            count_indexed += 1

end_time = time.time()
print(f"✅ Indicizzazione completata in {end_time - start_time:.2f} secondi.")
print(f"   Trovate {count_indexed} immagini originali.")
if duplicates_found:
    print(f"   Attenzione: Rilevati {len(duplicates_found)} nomi file duplicati. L'ultimo trovato per ogni nome è stato usato.")
    # print(f"   Nomi duplicati (primi 10): {list(set(duplicates_found))[:10]}")

if not original_image_paths:
    print("Errore: Nessuna immagine originale trovata nella directory specificata. Controlla ORIGINAL_DATASET_ROOT.")
    exit()

# 2. Elabora le immagini nella cartella di test
print(f"\n⚙️  Inizio elaborazione immagini di test da: {TEST_IMAGES_DIR}")
results_data = [] # Lista per contenere i risultati: {'image_path': path_test, 'ssim_score': score}
count_processed = 0
count_original_not_found = 0

if not os.path.isdir(TEST_IMAGES_DIR):
    print(f"Errore: La directory di test '{TEST_IMAGES_DIR}' non esiste.")
    exit()

test_image_files = [f for f in os.listdir(TEST_IMAGES_DIR)
                    if os.path.isfile(os.path.join(TEST_IMAGES_DIR, f)) and f.lower().endswith(VALID_EXTENSIONS)]

print(f"   Trovate {len(test_image_files)} immagini nella cartella di test.")

for test_filename in test_image_files:
    test_image_path = os.path.join(TEST_IMAGES_DIR, test_filename)
    print(f"--- Processando: {test_filename} ---")

    # Cerca l'originale corrispondente nell'indice
    if test_filename in original_image_paths:
        original_image_path = original_image_paths[test_filename]
        print(f"   Trovato originale corrispondente: {original_image_path}")

        # Carica entrambe le immagini
        print("   Caricamento immagine di test...")
        test_img = load_and_prep_image_for_ssim(test_image_path, IMG_SIZE)
        print("   Caricamento immagine originale...")
        original_img = load_and_prep_image_for_ssim(original_image_path, IMG_SIZE)

        # Calcola SSIM se entrambe le immagini sono state caricate
        if test_img is not None and original_img is not None:
            try:
                # Calcola SSIM. data_range è importante se usi uint8 (0-255)
                # Il primo argomento è l'originale, il secondo quello da confrontare
                score, diff_img = ssim(original_img, test_img, full=True, data_range=original_img.max() - original_img.min())
                # Ignoriamo diff_img qui, ci serve solo lo score
                # NB: data_range=255 è sicuro se le immagini sono uint8 standard

                print(f"   SSIM calcolato: {score:.4f}")
                results_data.append({
                    "image_path": test_image_path, # Salva il percorso dell'immagine di TEST
                    "ssim_score": score
                })
                count_processed += 1
            except Exception as e:
                print(f"   ❌ Errore durante il calcolo SSIM per {test_filename}: {e}")
        else:
            print(f"   Skipping SSIM calculation due to loading errors.")

    else:
        print(f"   ⚠️ Originale non trovato nell'indice per: {test_filename}. Impossibile calcolare SSIM.")
        count_original_not_found += 1

print(f"\n✅ Elaborazione immagini di test completata.")
print(f"   Immagini processate con successo (SSIM calcolato): {count_processed}")
print(f"   Immagini di test per cui non è stato trovato l'originale: {count_original_not_found}")

# 3. Salvataggio CSV
if results_data:
    print(f"\n💾 Salvataggio risultati in: {OUTPUT_CSV}")
    try:
        df = pd.DataFrame(results_data)
        # Ordina opzionalmente per nome file, se utile
        # df['filename'] = df['image_path'].apply(os.path.basename)
        # df = df.sort_values(by='filename').drop(columns=['filename'])
        df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f') # Formatta float per consistenza
        print(f"   File CSV salvato con successo.")
    except Exception as e:
        print(f"   ❌ Errore durante il salvataggio del file CSV: {e}")
else:
    print("\n🤷 Nessun risultato SSIM da salvare (nessuna corrispondenza trovata o errori durante l'elaborazione).")

print("\n🏁 Script terminato.")