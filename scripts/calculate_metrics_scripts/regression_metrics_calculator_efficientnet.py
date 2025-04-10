import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import pandas as pd # Importa Pandas per leggere facilmente il CSV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import csv # Lo useremo per salvare le metriche

# --- Configurazione Utente ---
IMG_SIZE = (384, 384) 
MODEL_PATH = "modelli_keras/no_reference_efficientnet.keras" 
TEST_DIR = "test_degraded" 
CSV_TRUE_VALUES_PATH = "dati_csv/true_scores.csv" 
METRICS_OUTPUT_PATH = "dati_csv/efficientnet_regression_metrics.csv" 

CSV_IMAGE_PATH_COL = 'image_path'  
CSV_TRUE_VALUE_COL = 'ssim_score' 

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff') 



# --- Funzioni Ausiliarie ---
def load_and_preprocess_image(path, img_size):
    """Carica, ridimensiona, converte colore e normalizza un'immagine."""
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"Attenzione: Impossibile leggere l'immagine {os.path.basename(path)}. Saltata.")
            return None
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        return img
    except Exception as e:
        print(f"Errore durante caricamento/preprocessamento di {os.path.basename(path)}: {e}")
        return None

# --- Script Principale ---

# 1. Carica il modello
print(f"Caricamento modello da: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH, compile=False) # compile=False è spesso ok per sola inferenza
    print("Modello caricato con successo.")
except Exception as e:
    print(f"Errore irreversibile durante il caricamento del modello: {e}")
    exit() # Esce se il modello non può essere caricato

# 2. Leggi i valori veri dal CSV
print(f"Lettura valori veri da: {CSV_TRUE_VALUES_PATH}")
try:
    df_true = pd.read_csv(CSV_TRUE_VALUES_PATH)
    # Crea un dizionario per una ricerca veloce: {nome_file: valore_vero}
    # Estraiamo solo il nome base del file dal percorso nel CSV per matching più robusto
    true_values_map = pd.Series(df_true[CSV_TRUE_VALUE_COL].values,
                                index=df_true[CSV_IMAGE_PATH_COL].apply(os.path.basename)).to_dict()
    print(f"Trovati {len(true_values_map)} valori veri nel CSV.")
    if not true_values_map:
        print("Errore: Nessun valore vero trovato nel CSV o le colonne non sono corrette.")
        exit()
except FileNotFoundError:
    print(f"Errore: File CSV dei valori veri non trovato a '{CSV_TRUE_VALUES_PATH}'")
    exit()
except KeyError as e:
    print(f"Errore: Colonna '{e}' non trovata nel file CSV. Controlla i nomi delle colonne (CSV_IMAGE_PATH_COL, CSV_TRUE_VALUE_COL).")
    exit()
except Exception as e:
    print(f"Errore durante la lettura del file CSV: {e}")
    exit()

# 3. Prepara le liste per raccogliere i dati per le metriche
y_true_list = []
y_pred_list = []
processed_files = [] # Per tenere traccia dei file processati

# 4. Itera sulle immagini nella cartella di test, predici e raccogli risultati
print(f"Inizio elaborazione immagini da: {TEST_DIR}")
if not os.path.isdir(TEST_DIR):
    print(f"Errore: La directory di test '{TEST_DIR}' non esiste.")
    exit()

image_files_in_dir = [f for f in os.listdir(TEST_DIR)
                      if os.path.isfile(os.path.join(TEST_DIR, f)) and f.lower().endswith(VALID_EXTENSIONS)]

print(f"Trovate {len(image_files_in_dir)} immagini valide nella directory.")

count_matched = 0
for filename in image_files_in_dir:
    # Cerca il valore vero corrispondente usando il nome del file
    if filename in true_values_map:
        image_path = os.path.join(TEST_DIR, filename)

        # Carica e preprocessa l'immagine
        img = load_and_preprocess_image(image_path, IMG_SIZE)

        if img is not None:
            try:
                # Aggiungi dimensione batch e predici
                img_batch = np.expand_dims(img, axis=0)
                prediction = model.predict(img_batch, verbose=0)[0][0] # Assume output singolo scalare

                # Ottieni il valore vero dalla mappa
                true_value = true_values_map[filename]

                # Aggiungi alle liste
                y_true_list.append(float(true_value)) # Assicura sia float
                y_pred_list.append(float(prediction)) # Assicura sia float
                processed_files.append(filename)
                count_matched += 1

            except Exception as e:
                print(f"Errore durante la predizione per l'immagine {filename}: {e}")
    # else:
        # Opzionale: informa se un'immagine nella cartella non ha un valore vero nel CSV
        # print(f"Info: Immagine '{filename}' trovata nella cartella ma non nel file CSV dei valori veri. Saltata.")

print(f"Elaborazione completata. {count_matched} immagini sono state processate (trovate sia in {TEST_DIR} che in {CSV_TRUE_VALUES_PATH}).")

# 5. Calcola le metriche se ci sono dati sufficienti
if not y_true_list:
    print("\nErrore: Nessuna corrispondenza trovata tra le immagini nella cartella e i dati nel CSV.")
    print("Impossibile calcolare le metriche. Controlla che i nomi/percorsi nel CSV corrispondano ai nomi file nella cartella.")
else:
    print("\nCalcolo delle metriche di regressione...")
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    try:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse) # RMSE è spesso utile

        # Stampa (opzionale, dato che salveremo su CSV)
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        # 6. Salva le metriche su un file CSV
        print(f"\nSalvataggio metriche in: {METRICS_OUTPUT_PATH}")
        try:
            with open(METRICS_OUTPUT_PATH, 'w', newline='') as csvfile:
                metric_writer = csv.writer(csvfile)
                metric_writer.writerow(['Metric', 'Value']) # Header
                metric_writer.writerow(['R2', r2])
                metric_writer.writerow(['MAE', mae])
                metric_writer.writerow(['MSE', mse])
                metric_writer.writerow(['RMSE', rmse])
                metric_writer.writerow(['Num_Samples_Evaluated', len(y_true)]) # Aggiungi il numero di campioni usati
            print("Metriche salvate con successo.")
        except IOError as e:
            print(f"Errore durante il salvataggio del file CSV delle metriche: {e}")
        except Exception as e:
            print(f"Errore imprevisto durante il salvataggio delle metriche: {e}")

    except ValueError as e:
        print(f"\nErrore durante il calcolo delle metriche: {e}")
        print("Possibili cause: numero insufficiente di campioni, valori NaN o infiniti.")
    except Exception as e:
        print(f"\nErrore imprevisto durante il calcolo delle metriche: {e}")


print("\nScript terminato.")