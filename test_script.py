import tensorflow as tf
import os
import numpy as np
import time

# --- Funzione load_and_create_inference_model (identica a prima) ---
def load_and_create_inference_model(original_model_path='modello.h5', input_shape=(224, 224, 3)):
    """
    Carica il modello originale a due input e crea un nuovo modello
    per l'inferenza a singolo input, simulando la concatenazione interna.
    """
    print(f"Caricamento modello originale da: {original_model_path}")
    try:
        original_model = tf.keras.models.load_model(original_model_path)
        print("Modello originale caricato con successo.")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il modello originale da {original_model_path}. Dettagli: {e}")
        return None

    print("Creazione del modello di inferenza single-input...")
    try:
        single_input = tf.keras.Input(shape=input_shape, name="inference_input")
        concatenated_input = tf.keras.layers.Concatenate(axis=-1, name="inference_concatenate")([single_input, single_input])

        first_processing_layer = None
        found_concat = False
        for layer in original_model.layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                 found_concat = True
                 continue
            if found_concat:
                 first_processing_layer = layer
                 break

        if first_processing_layer is None:
            print("ERRORE: Impossibile trovare il layer Concatenate o il layer successivo.")
            return None

        print(f"Collegamento dell'input duplicato al layer '{first_processing_layer.name}' del modello originale.")

        current_tensor = concatenated_input
        output_tensor = None
        processing_started = False
        for layer in original_model.layers:
             if layer.name == first_processing_layer.name:
                 processing_started = True
             if processing_started:
                 current_tensor = layer(current_tensor)
                 output_tensor = current_tensor

        if output_tensor is None:
             print("ERRORE: Processo di collegamento fallito.")
             return None

        inference_model = tf.keras.Model(inputs=single_input, outputs=output_tensor, name="inference_model")
        print("Modello di inferenza single-input creato con successo.")
        # inference_model.summary() # Decommenta per vedere la struttura

        return inference_model

    except Exception as e:
        print(f"ERRORE durante la creazione del modello di inferenza: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Funzione per caricare e preparare UNA singola immagine ---
def load_and_prepare_image(image_path, img_size=(224, 224)):
    """Carica un'immagine, la ridimensiona e normalizza."""
    try:
        print(f"Caricamento immagine da: {image_path}")
        img = tf.keras.utils.load_img(image_path, target_size=img_size, interpolation='lanczos')
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0 # Normalizza a [0, 1]
        # Aggiunge la dimensione del batch (il modello si aspetta un batch, anche se di 1)
        img_batch = np.expand_dims(img_array, axis=0)
        print(f"Immagine caricata e preparata con shape: {img_batch.shape}")
        return tf.convert_to_tensor(img_batch, dtype=tf.float32)
    except FileNotFoundError:
        print(f"ERRORE: Immagine non trovata: {image_path}")
        return None
    except Exception as e:
        print(f"ERRORE durante caricamento/preparazione immagine {image_path}: {e}")
        return None

# --- Blocco Principale Semplificato ---
if __name__ == "__main__":

    # --- CONFIGURAZIONE ---
    original_model_h5_path = 'modello.h5'
    # SPECIFICA QUI IL PERCORSO ESATTO DELL'IMMAGINE CHE VUOI TESTARE
    image_to_test_path = 'L:\\Tesi Image Quality DermChecker\\dataset\\motion_blurred\\18_XIAOMI-REDMI-5-PLUS_M.jpg'
    input_image_size = (224, 224) # Assicurati che coincida con l'input del modello

    print("--- Inizio Script Test Semplificato Single-Input ---")
    start_time = time.time()

    # 1. Carica e crea il modello di inferenza single-input
    print("\n--- Fase 1: Creazione Modello Inferenza ---")
    inference_model = load_and_create_inference_model(
        original_model_path=original_model_h5_path,
        input_shape=(input_image_size[0], input_image_size[1], 3)
    )

    if inference_model:
        # 2. Carica e prepara l'immagine singola
        print("\n--- Fase 2: Caricamento Immagine Test ---")
        # Assicurati che il percorso sia valido prima di procedere
        if not os.path.exists(image_to_test_path):
             print(f"ERRORE FATALE: Il percorso immagine specificato non esiste: {image_to_test_path}")
        else:
            image_tensor = load_and_prepare_image(image_to_test_path, img_size=input_image_size)

            if image_tensor is not None:
                # 3. Esegui la predizione
                print("\n--- Fase 3: Esecuzione Predizione ---")
                try:
                    prediction_start_time = time.time()
                    # Esegui predict sull'UNICO tensore immagine (che ha già la dimensione batch)
                    prediction = inference_model.predict(image_tensor, verbose=0)
                    prediction_time = time.time() - prediction_start_time

                    # Estrai il valore scalare della predizione (assumendo output Dense(1))
                    prediction_score = prediction[0][0] if prediction.ndim >= 2 else prediction[0]

                    print("\n--- RISULTATO ---")
                    print(f"Immagine Testata: {image_to_test_path}")
                    print(f"Punteggio Predetto: {prediction_score:.6f}")
                    print(f"(Tempo di predizione: {prediction_time:.4f} secondi)")

                    # Interpretazione (DA ADATTARE IN BASE AL TRAINING!)
                    # Questo è solo un ESEMPIO di interpretazione, potrebbe essere l'opposto!
                    if prediction_score < 0.5:
                        print("Interpretazione Esempio: Il modello suggerisce qualità ALTA (punteggio basso).")
                    else:
                        print("Interpretazione Esempio: Il modello suggerisce qualità BASSA (punteggio alto).")
                    print("NOTA: L'interpretazione dipende da come il modello è stato addestrato (label 0 vs 1).")


                except Exception as e:
                    print(f"ERRORE durante la predizione: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("\nImpossibile procedere con la predizione: Errore caricamento immagine.")
    else:
        print("\nImpossibile procedere: Creazione del modello di inferenza fallita.")

    end_time = time.time()
    print(f"\n--- Script Terminato --- (Tempo totale: {end_time - start_time:.2f} secondi)")