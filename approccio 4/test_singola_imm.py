from tensorflow.keras.models import load_model
import numpy as np
import cv2

IMG_SIZE = (224, 224)
model = load_model("no_reference_model_weighted.keras", compile=False)

def load_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

img = load_img("sharp/39_001.jpg")
img = np.expand_dims(img, axis=0)
score = model.predict(img)[0][0]

print(f"📈 Sharpness score (no-reference): {score:.4f}")
