import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Cartelle di origine per le immagini
SHARP_DIR = "/content/drive/MyDrive/approccio4/sharp"  # Cartella contenente le immagini originali
DEGRADED_DIR = "/content/drive/MyDrive/approccio4/degraded"  # Cartella contenente le immagini degradate
OUTPUT_CSV = "/content/drive/MyDrive/approccio4/ssim_dataset.csv"  # Nome del file CSV di output

IMG_SIZE = (384, 384)  # Imposta la dimensione delle immagini per il calcolo

data = []  # Lista per raccogliere i dati (immagini e punteggio SSIM)

print("📦 Inizio generazione dataset con SSIM...")

# Cicla su ogni file nella cartella delle immagini originali
for filename in os.listdir(SHARP_DIR):
    # Costruisce i percorsi per le immagini originali e degradate
    sharp_path = os.path.join(SHARP_DIR, filename)
    degraded_path = os.path.join(DEGRADED_DIR, filename)

    # Controlla se l'immagine degradata esiste
    if not os.path.exists(degraded_path):
        print(f"⚠️ Mancante: {degraded_path}")  # Avviso se l'immagine è mancante
        continue  # Passa al prossimo file

    try:
        # Carica le immagini originali e degradate
        sharp_img = cv2.imread(sharp_path)
        degraded_img = cv2.imread(degraded_path)

        # Ridimensiona le immagini alla dimensione desiderata
        sharp_img = cv2.resize(sharp_img, IMG_SIZE)
        degraded_img = cv2.resize(degraded_img, IMG_SIZE)

        # Converte le immagini in scala di grigi per calcolare SSIM
        sharp_gray = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2GRAY)
        degraded_gray = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2GRAY)

        # Calcola il punteggio SSIM tra le immagini
        score = ssim(sharp_gray, degraded_gray)

        # Aggiunge il risultato per l'immagine degradata
        data.append({
            "original": sharp_path,  # Percorso immagine originale
            "degraded": degraded_path,  # Percorso immagine degradata
            "score": score  # Punteggio SSIM calcolato
        })

        # Aggiungi anche l'immagine sharp al dataset con un punteggio di qualità ideale (1.0)
        # Le immagini sharp sono considerate perfette, quindi assegniamo loro un punteggio di 1.0
        data.append({
            "original": sharp_path,  # Percorso immagine originale
            "degraded": sharp_path,  # L'immagine originale è anche il suo "degradato" (stesso percorso)
            "score": 1.0  # Punteggio massimo per le immagini sharp
        })

    except Exception as e:
        print(f"❌ Errore con {filename}: {e}")  # Gestisce eventuali errori di lettura o calcolo

# Salva i dati raccolti in un file CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)  # Scrive i dati nel CSV senza l'indice
print(f"✅ Dataset generato e salvato in: {OUTPUT_CSV}")  # Conferma del salvataggio

