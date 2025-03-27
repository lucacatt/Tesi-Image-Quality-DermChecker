# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import regularizers
import os
import random
import numpy as np
import math
import cv2
import time
import traceback # Per stampare errori dettagliati
# Rimosso import get_custom_objects, non useremo la registrazione globale

# --- Parametri Configurabili ---
SHARP_DIR = "sharp"
DEGRADED_DIR = "degraded"
IMG_HEIGHT, IMG_WIDTH = 224, 224
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
BATCH_SIZE = 16
EPOCHS = 50
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
MARGIN = 0.5
# Nome del file per salvare SOLO il backbone addestrato (per inferenza)
SAVE_BACKBONE_PATH = 'siamese_backbone_best.h5'
# Nome temporaneo per salvare il modello siamese completo migliore durante il training
SIAMESE_BEST_PATH_TEMP = SAVE_BACKBONE_PATH.replace(".h5", "_siamese_full_temp.h5")


# --- 1. Funzione Custom per Normalizzazione (NON DECORATA) ---
# Definisci al livello superiore
def l2_normalize_layer(tensor):
  """Applica L2 Normalizzazione, gestendo potenziale divisione per zero."""
  return tf.math.l2_normalize(tensor, axis=1, epsilon=1e-6)

# --- 2. Funzione per Creare il Backbone (Embedding Model) ---
def create_embedding_model(input_shape, embedding_dim=128):
    """Crea il modello CNN base che produce l'embedding."""
    input_image = Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_image)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(embedding_dim, kernel_regularizer=regularizers.l2(1e-4))(x)

    # Usa la funzione definita esternamente nel layer Lambda
    x = layers.Lambda(
        l2_normalize_layer,
        output_shape=(embedding_dim,),
        name='embedding_output'
    )(x)

    model = Model(inputs=input_image, outputs=x, name='embedding_model')
    return model

# --- 3. Funzione Triplet Loss (NON DECORATA) ---
# Definisci al livello superiore
def triplet_loss(y_true, y_pred, margin=0.5):
    """
    Calcola la Triplet Loss.
    Assume che y_pred abbia shape (batch_size, 3, embedding_dim).
    y_true è ignorato.
    """
    anchor = y_pred[:, 0, :]
    positive = y_pred[:, 1, :]
    negative = y_pred[:, 2, :]

    pos_dist_sq = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist_sq = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    basic_loss = pos_dist_sq - neg_dist_sq + margin
    loss = tf.maximum(0.0, basic_loss)

    return tf.reduce_mean(loss)

# Wrapper per passare il margine durante la compilazione/caricamento
def get_triplet_loss_with_margin(margin=MARGIN):
    # Questa funzione interna cattura il valore di margin
    def loss_fn(y_true, y_pred):
        return triplet_loss(y_true, y_pred, margin=margin)
    # Assegna un nome riconoscibile alla funzione wrapper per il caricamento
    loss_fn.__name__ = f'triplet_loss_margin_{str(margin).replace(".", "_")}'
    return loss_fn

# --- 4. Funzione per Creare il Modello Siamese ---
def create_siamese_network(input_shape, backbone):
    """Crea il modello Siamese completo con 3 input e output per la Triplet Loss."""
    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_positive')
    input_negative = Input(shape=input_shape, name='input_negative')

    embedding_anchor = backbone(input_anchor)
    embedding_positive = backbone(input_positive)
    embedding_negative = backbone(input_negative)

    # Usa tf.stack (standard)
    output = layers.Lambda(
        lambda x: tf.stack(x, axis=1),
        output_shape=(3, backbone.output_shape[-1]),
        name='stacked_embeddings'
    )([embedding_anchor, embedding_positive, embedding_negative])

    model = Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=output,
        name='siamese_network'
    )
    return model # Compilazione nel blocco main

# --- 5. Classe Dataset per Triplette ---
class TripletDataset(tf.keras.utils.Sequence):
    def __init__(self, sharp_dir, degraded_dir, batch_size, img_size,
                 validation_split=0.2, subset='training', shuffle=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.subset = subset
        sharp_paths = self._get_image_paths(sharp_dir)
        degraded_paths = self._get_image_paths(degraded_dir)
        sharp_map = {os.path.basename(p): p for p in sharp_paths}
        self.triplet_paths = []
        print("Creazione triplette (Anchor=Sharp, Positive=Sharp, Negative=Degraded)...")
        found_pairs = 0
        for neg_path in degraded_paths:
            basename = os.path.basename(neg_path)
            if basename in sharp_map:
                anchor_pos_path = sharp_map[basename]
                self.triplet_paths.append((anchor_pos_path, anchor_pos_path, neg_path))
                found_pairs += 1
        print(f"Trovate {found_pairs} coppie sharp/degraded per creare triplette.")
        if not self.triplet_paths: raise ValueError("Nessuna tripletta creata. Controlla nomi file e cartelle.")
        num_total_samples = len(self.triplet_paths)
        num_validation_samples = int(num_total_samples * validation_split)
        num_training_samples = num_total_samples - num_validation_samples
        random.shuffle(self.triplet_paths)
        if subset == 'training':
            self.current_triplet_paths = self.triplet_paths[:num_training_samples]
            print(f"Dataset Training: {len(self.current_triplet_paths)} triplette.")
        elif subset == 'validation':
            self.current_triplet_paths = self.triplet_paths[num_training_samples:]
            print(f"Dataset Validazione: {len(self.current_triplet_paths)} triplette.")
        else: raise ValueError("subset deve essere 'training' o 'validation'")
        if not self.current_triplet_paths: print(f"ATTENZIONE: Subset '{subset}' è vuoto!")
        self.on_epoch_end()

    def _get_image_paths(self, dir_path):
        image_paths = []
        if not os.path.isdir(dir_path): return []
        for subdir, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(subdir, file))
        return image_paths

    def __len__(self):
        if not self.current_triplet_paths: return 0
        return math.ceil(len(self.current_triplet_paths) / self.batch_size)

    def __getitem__(self, index):
        if not self.current_triplet_paths: raise IndexError(f"Dataset '{self.subset}' è vuoto.")
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.current_triplet_paths))
        batch_triplet_paths = self.current_triplet_paths[start_idx:end_idx]
        batch_anchors, batch_positives, batch_negatives = [], [], []
        valid_triplets_in_batch = 0
        for anchor_p, pos_p, neg_p in batch_triplet_paths:
            try:
                anchor_img = self._load_image(anchor_p)
                pos_img = self._load_image(pos_p)
                neg_img = self._load_image(neg_p)
                batch_anchors.append(anchor_img)
                batch_positives.append(pos_img)
                batch_negatives.append(neg_img)
                valid_triplets_in_batch += 1
            except Exception as e:
                print(f"ERRORE caricando tripletta ({os.path.basename(anchor_p)}, ...): {e}. Saltata.")
                continue
        if valid_triplets_in_batch == 0:
             print(f"Attenzione: Batch {index} non ha prodotto triplette valide.")
             empty_batch = tf.zeros((0, self.img_size[0], self.img_size[1], 3), dtype=tf.float32)
             return (empty_batch, empty_batch, empty_batch), tf.zeros((0,))
        anchors_tensor = tf.convert_to_tensor(np.array(batch_anchors), dtype=tf.float32)
        positives_tensor = tf.convert_to_tensor(np.array(batch_positives), dtype=tf.float32)
        negatives_tensor = tf.convert_to_tensor(np.array(batch_negatives), dtype=tf.float32)
        dummy_labels = tf.zeros(valid_triplets_in_batch)
        return (anchors_tensor, positives_tensor, negatives_tensor), dummy_labels

    def _load_image(self, img_path):
        img = tf.keras.utils.load_img(img_path, target_size=self.img_size, interpolation='lanczos')
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        return img_array

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.current_triplet_paths)

# --- 6. Blocco Principale ---
if __name__ == "__main__":
    print("--- Inizio Training Rete Siamese (Compatibilità TF<2.7, Caricamento Esplicito) ---")
    start_time = time.time()

    # 1. Crea i Datasets
    print("\n--- Creazione Datasets ---")
    try:
        train_dataset = TripletDataset(SHARP_DIR, DEGRADED_DIR, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH),
                                     validation_split=VALIDATION_SPLIT, subset='training', shuffle=True)
        val_dataset = TripletDataset(SHARP_DIR, DEGRADED_DIR, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH),
                                   validation_split=VALIDATION_SPLIT, subset='validation', shuffle=False)
        if len(train_dataset) == 0: raise ValueError("Training dataset is empty.")
        if len(val_dataset) == 0: print("ATTENZIONE: Validation dataset is empty.") ; val_dataset = None
    except Exception as e: print(f"ERRORE Creazione Dataset: {e}"); exit()

    # 2. Crea il Backbone e il Modello Siamese
    print("\n--- Creazione Modello ---")
    embedding_backbone = create_embedding_model(INPUT_SHAPE, EMBEDDING_DIM)
    siamese_model = create_siamese_network(INPUT_SHAPE, embedding_backbone)

    print("Struttura Backbone:")
    embedding_backbone.summary(line_length=100)
    print("\nStruttura Modello Siamese:")
    siamese_model.summary(line_length=100)

    # 3. Compila il Modello Siamese
    print("\n--- Compilazione Modello ---")
    # Ottieni la funzione loss wrapper
    loss_function_wrapper = get_triplet_loss_with_margin(margin=MARGIN)
    # NON usiamo più get_custom_objects().update() qui
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_function_wrapper # Passa la funzione wrapper
    )
    print("Modello compilato.")

    # 4. Callbacks
    print("\n--- Definizione Callbacks ---")
    checkpoint_siamese = tf.keras.callbacks.ModelCheckpoint(
            filepath=SIAMESE_BEST_PATH_TEMP,
            monitor='val_loss', mode='min',
            save_best_only=True, save_weights_only=False, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)
    callbacks_list = [checkpoint_siamese, early_stopping, reduce_lr]

    # 5. Addestramento
    print("\n--- Inizio Addestramento ---")
    history = siamese_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks_list
    )

    # 6. Estrai e Salva il Backbone Migliore
    print(f"\n--- Estrazione e Salvataggio Backbone da {SIAMESE_BEST_PATH_TEMP} ---")
    try:
        # Definisci ESATTAMENTE gli oggetti custom necessari per CARICARE il siamese model
        custom_objects_for_load = {
            'l2_normalize_layer': l2_normalize_layer, # La funzione definita sopra
            # Passa il wrapper della loss usando il suo __name__ come chiave
            loss_function_wrapper.__name__: loss_function_wrapper
        }

        print(f"Caricamento modello siamese migliore da: {SIAMESE_BEST_PATH_TEMP}")
        print(f"Usando custom_objects: {list(custom_objects_for_load.keys())}") # Stampa le chiavi usate

        if not os.path.exists(SIAMESE_BEST_PATH_TEMP):
             print(f"ERRORE: File modello siamese migliore non trovato: {SIAMESE_BEST_PATH_TEMP}.")
             print("Controlla se il training ha prodotto output o se la validazione è stata eseguita.")
        else:
             # Passa il dizionario custom esplicito a load_model
             best_siamese_model = tf.keras.models.load_model(
                 SIAMESE_BEST_PATH_TEMP,
                 custom_objects=custom_objects_for_load, # Usa il dizionario definito sopra
                 compile=False # Non ricompilare, vogliamo solo estrarre layer
             )
             print("Modello siamese caricato.")

             best_backbone = best_siamese_model.get_layer('embedding_model')
             print(f"Backbone ('{best_backbone.name}') estratto.")

             # Salva SOLO il backbone.
             # Questo salvataggio ORA dovrebbe funzionare perché il modello
             # è stato caricato correttamente grazie a custom_objects.
             best_backbone.save(SAVE_BACKBONE_PATH)
             print(f"Backbone addestrato salvato con successo in: {SAVE_BACKBONE_PATH}")

             # Opzionale: Rimuovi il file temporaneo del modello siamese completo
             try:
                 os.remove(SIAMESE_BEST_PATH_TEMP)
                 print(f"File temporaneo rimosso: {SIAMESE_BEST_PATH_TEMP}")
             except OSError as e:
                 print(f"Errore rimozione file temporaneo: {e}")

    except Exception as e:
        print(f"ERRORE durante caricamento/salvataggio backbone: {e}")
        traceback.print_exc()
        print("Salvataggio del backbone fallito. Il modello siamese completo migliore (se creato) è in:", SIAMESE_BEST_PATH_TEMP)

    end_time = time.time()
    print(f"\n--- Addestramento Completato --- (Tempo totale: {(end_time - start_time)/60:.2f} minuti)")