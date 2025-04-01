from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

model = load_model("quality_score_model.h5", compile=False)
img = load_img("L:\\Tesi Image Quality DermChecker\\dataset\\motion_blurred\\21_HUAWEI-Y9_M.jpg", target_size=(224, 224))
arr = img_to_array(img) / 255.0
score = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0][0]
print(f"📈 Sharpness score: {score:.4f}")
