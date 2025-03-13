import tensorflow as tf
import numpy as np
import os
import time
from SharpnessMetricsLayerClass import SharpnessMetricsLayer
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = tf.keras.models.load_model("blur_detection_with_metrics_retrained_intensified.h5", custom_objects={"SharpnessMetricsLayer": SharpnessMetricsLayer})

dataset_path = "dataset"
categories = ["defocused_blurred", "motion_blurred", "sharp"]

results = []
true_labels = []
predicted_labels = []

LAPLACIAN_TARGET_MIN = 0
LAPLACIAN_TARGET_MAX = 1000

def collect_laplacian_values():
    laplacians = []
    for category in categories:
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                img = tf.io.read_file(img_path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, (224, 224)) / 255.0
                img = tf.expand_dims(img, axis=0)

                _, sharpness_metrics = model.predict(img)
                laplacian = sharpness_metrics[0][0]
                laplacians.append(laplacian)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return min(laplacians), max(laplacians)

LAPLACIAN_MIN_NORMALIZED, LAPLACIAN_MAX_NORMALIZED = collect_laplacian_values()

def rescale_laplacian(laplacian_normalized, min_norm=LAPLACIAN_MIN_NORMALIZED, max_norm=LAPLACIAN_MAX_NORMALIZED, target_min=LAPLACIAN_TARGET_MIN, target_max=LAPLACIAN_TARGET_MAX):
    if max_norm == min_norm:
        return target_min
    rescaled_lap = ((laplacian_normalized - min_norm) / (max_norm - min_norm)) * (target_max - target_min) + target_min
    return rescaled_lap

for category in categories:
    folder = os.path.join(dataset_path, category)
    ground_truth_label = 1 if category == "sharp" else 0

    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} not found.")
        continue

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        print(f"Processing {filename} in {category}")
        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (224, 224)) / 255.0
            img = tf.expand_dims(img, axis=0)

            start_time = time.time()
            classification_output, sharpness_metrics = model.predict(img)
            test_time = time.time() - start_time

            predicted_class_prob = classification_output[0][0]
            predicted_class = 1 if predicted_class_prob > 0.5 else 0
            predicted_label_str = "Nitida" if predicted_class == 1 else "Sfocata"
            ground_truth_label_str = "Nitida" if ground_truth_label == 1 else "Sfocata"

            laplacian, vog, tenengrad, ssim, psnr = sharpness_metrics[0]

            rescaled_laplacian = rescale_laplacian(laplacian)

            results.append({
                "Immagine": filename,
                "Categoria Reale": category,
                "Etichetta Reale": ground_truth_label_str,
                "Predizione Probabilità": predicted_class_prob,
                "Classe Predetta": predicted_class,
                "Etichetta Predetta": predicted_label_str,
                "Lap (Normalizzata)": laplacian,
                "Lap (Rescaled)": rescaled_laplacian,
                "VoG": vog,
                "Tenengrad": tenengrad,
                "SSIM": ssim,
                "PSNR": psnr,
                "Tempo (s)": round(test_time, 4)
            })
            true_labels.append(ground_truth_label)
            predicted_labels.append(predicted_class)

            print(f"   Laplacian (Normalizzata): {laplacian}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

df_results = pd.DataFrame(results)
df_results.to_csv("test_results.csv", index=False)

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["Sfocata", "Nitida"]))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nTest completato e risultati salvati in test_results.csv")
