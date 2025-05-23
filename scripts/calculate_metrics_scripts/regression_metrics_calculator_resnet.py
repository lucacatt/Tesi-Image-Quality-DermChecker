import tensorflow as tf
from tensorflow.keras.models import load_model
# Importa la funzione di preprocessing specifica per ResNetV2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import csv

MODEL_NAME = "ResNet50V2"
MODEL_PATH = "modelli_keras/no_reference_resnet50v2.keras"  
METRICS_OUTPUT_PATH = "dati_csv/resnet50v2_regression_metrics.csv" 

IMG_SIZE = (384, 384) 

TEST_DIR = "test_degraded"
CSV_TRUE_VALUES_PATH = "dati_csv/true_scores.csv"

CSV_IMAGE_PATH_COL = 'image_path'
CSV_TRUE_VALUE_COL = 'ssim_score' 

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')



# --- Funzione Ausiliaria (MODIFICATA PER RESNETV2 PREPROCESSING) ---
def load_and_preprocess_image(path, img_size):
    """Carica, ridimensiona, converte colore e applica ResNetV2 preprocess_input."""
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"Attenzione: Impossibile leggere l'immagine {os.path.basename(path)}. Saltata.")
            return None

        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Assumi che il training usasse RGB

        # Converti in float32 PRIMA di passare a preprocess_input
        img_rgb_float = img.astype("float32")

        # *** APPLICA IL PREPROCESSING SPECIFICO DI RESNETV2 ***
        img_preprocessed = resnet_preprocess_input(img_rgb_float)

        return img_preprocessed # Restituisce l'immagine preprocessata

    except Exception as e:
        print(f"Errore durante caricamento/preprocessamento di {os.path.basename(path)}: {e}")
        return None

# --- Script Principale (logica quasi identica, cambiano solo i path e il preprocessing) ---

print(f"--- Valutazione Modello: {MODEL_NAME} ---")

# 1. Carica il modello ResNet50V2
print(f"Caricamento modello da: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Modello caricato con successo.")
    # model.summary() # Puoi decommentare per vedere la struttura
except Exception as e:
    print(f"Errore irreversibile durante il caricamento del modello: {e}")
    exit()

# 2. Leggi i valori veri dal CSV (stesso codice di prima)
print(f"Lettura valori veri da: {CSV_TRUE_VALUES_PATH}")
try:
    df_true = pd.read_csv(CSV_TRUE_VALUES_PATH)
    if CSV_IMAGE_PATH_COL not in df_true.columns or CSV_TRUE_VALUE_COL not in df_true.columns:
        raise KeyError(f"Colonne '{CSV_IMAGE_PATH_COL}' o '{CSV_TRUE_VALUE_COL}' non trovate nel CSV.")

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
    print(f"Errore: {e}")
    exit()
except Exception as e:
    print(f"Errore durante la lettura del file CSV: {e}")
    exit()

# 3. Prepara le liste per raccogliere i dati (stesso codice di prima)
y_true_list = []
y_pred_list = []
processed_files = []

# 4. Itera sulle immagini, predici e raccogli risultati (stesso codice di prima,
#    ma userà la funzione load_and_preprocess_image aggiornata)
print(f"Inizio elaborazione immagini da: {TEST_DIR}")
if not os.path.isdir(TEST_DIR):
    print(f"Errore: La directory di test '{TEST_DIR}' non esiste.")
    exit()

image_files_in_dir = [f for f in os.listdir(TEST_DIR)
                      if os.path.isfile(os.path.join(TEST_DIR, f)) and f.lower().endswith(VALID_EXTENSIONS)]

print(f"Trovate {len(image_files_in_dir)} immagini valide nella directory.")

count_matched = 0
for filename in image_files_in_dir:
    if filename in true_values_map:
        image_path = os.path.join(TEST_DIR, filename)
        # *** CHIAMA LA FUNZIONE DI PREPROCESSING AGGIORNATA ***
        img = load_and_preprocess_image(image_path, IMG_SIZE)

        if img is not None:
            try:
                img_batch = np.expand_dims(img, axis=0)
                # Predizione con il modello ResNet50V2 caricato
                prediction = model.predict(img_batch, verbose=0)[0][0]

                true_value = true_values_map[filename]
                y_true_list.append(float(true_value))
                y_pred_list.append(float(prediction))
                processed_files.append(filename)
                count_matched += 1
            except Exception as e:
                print(f"Errore durante la predizione per l'immagine {filename}: {e}")

print(f"Elaborazione completata. {count_matched} immagini sono state processate.")

# 5. Calcola le metriche (stesso codice di prima)
if not y_true_list:
    print("\nErrore: Nessuna corrispondenza trovata o processata.")
    print("Impossibile calcolare le metriche.")
else:
    print("\nCalcolo delle metriche di regressione...")
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    try:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        # 6. Salva le metriche nel NUOVO file CSV (stesso codice di prima, ma usa METRICS_OUTPUT_PATH aggiornato)
        print(f"\nSalvataggio metriche per {MODEL_NAME} in: {METRICS_OUTPUT_PATH}")
        try:
            with open(METRICS_OUTPUT_PATH, 'w', newline='') as csvfile:
                metric_writer = csv.writer(csvfile)
                metric_writer.writerow(['Metric', 'Value'])
                metric_writer.writerow(['Model_Name', MODEL_NAME]) # Aggiungi il nome del modello per chiarezza
                metric_writer.writerow(['R2', r2])
                metric_writer.writerow(['MAE', mae])
                metric_writer.writerow(['MSE', mse])
                metric_writer.writerow(['RMSE', rmse])
                metric_writer.writerow(['Num_Samples_Evaluated', len(y_true)])
            print("Metriche salvate con successo.")
        except IOError as e:
            print(f"Errore durante il salvataggio del file CSV delle metriche: {e}")
        except Exception as e:
            print(f"Errore imprevisto durante il salvataggio delle metriche: {e}")

    except ValueError as e:
        print(f"\nErrore durante il calcolo delle metriche: {e}")
    except Exception as e:
        print(f"\nErrore imprevisto durante il calcolo delle metriche: {e}")


print(f"\n--- Script terminato per {MODEL_NAME} ---")