"""
Utilise le modèle entraîné pour prédire sur une image.

Usage :
  python predict.py <chemin_vers_image>

Exemple :
  python predict.py data/chest_xray/test/PNEUMONIA/person1_virus_1.jpeg
  python predict.py data/chest_xray/test/NORMAL/IM-0001-0001.jpeg
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from src.model import CNN

INPUT_SIZE    = 64
WEIGHTS_PATH  = "model_weights.npz"


def preprocess(image_path, input_size):
    img = Image.open(image_path).convert("L")
    img = img.resize((input_size, input_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[:, :, np.newaxis]        # (H, W, 1)
    return arr[np.newaxis, :, :, :]    # (1, H, W, 1) — batch de 1


def predict(image_path):
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Poids introuvables : {WEIGHTS_PATH}")
        print("Lance d'abord : python train.py")
        return

    model = CNN(input_size=INPUT_SIZE)
    model.load(WEIGHTS_PATH)

    x = preprocess(image_path, INPUT_SIZE)
    prob = model.forward(x)[0, 0]
    label = "PNEUMONIE" if prob >= 0.5 else "NORMAL"
    confidence = prob if prob >= 0.5 else 1 - prob

    print(f"\nImage      : {os.path.basename(image_path)}")
    print(f"Prédiction : {label}")
    print(f"Confiance  : {confidence * 100:.1f}%")
    print(f"(probabilité brute de pneumonie : {prob:.4f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python predict.py <chemin_image>")
        sys.exit(1)
    predict(sys.argv[1])
