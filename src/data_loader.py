"""
Chargement et prétraitement des images.

Étapes de prétraitement :
  1. Charger l'image en niveaux de gris (les poumons n'ont pas besoin de couleur)
  2. Redimensionner à input_size x input_size (le réseau attend une taille fixe)
  3. Normaliser les pixels entre 0 et 1 (diviser par 255)
     → Sans normalisation, les gradients seraient instables (valeurs trop grandes)
  4. Ajouter la dimension du canal : (H, W) → (H, W, 1)

Labels :
  - NORMAL    → 0
  - PNEUMONIA → 1
"""

import os
import numpy as np
from PIL import Image


def load_dataset(data_dir, input_size=64, max_per_class=None):
    """
    Charge les images depuis data_dir/NORMAL et data_dir/PNEUMONIA.

    Retourne :
      X : (n_samples, input_size, input_size, 1)  — images normalisées
      y : (n_samples, 1)                           — labels (0 ou 1)
    """
    images = []
    labels = []

    for label_name, label_value in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dossier introuvable : {folder}")

        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpeg", ".jpg", ".png"))]

        if max_per_class:
            files = files[:max_per_class]

        print(f"  Chargement {label_name}: {len(files)} images...")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                img = Image.open(fpath).convert("L")          # Niveaux de gris
                img = img.resize((input_size, input_size))    # Redimensionner
                arr = np.array(img, dtype=np.float32) / 255.0 # Normaliser [0, 1]
                arr = arr[:, :, np.newaxis]                    # (H, W) → (H, W, 1)
                images.append(arr)
                labels.append([label_value])
            except Exception as e:
                print(f"  Erreur sur {fname}: {e}")

    X = np.array(images, dtype=np.float32)   # (n, H, W, 1)
    y = np.array(labels, dtype=np.float32)   # (n, 1)

    # Mélanger les données (sinon le réseau voit d'abord tous les 0 puis tous les 1)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def batch_generator(X, y, batch_size):
    """
    Générateur qui découpe (X, y) en mini-batches.

    Pourquoi des mini-batches ?
    → Charger toutes les images d'un coup prendrait trop de mémoire.
    → Les mini-batches apportent du "bruit" utile qui aide à sortir
      des minima locaux.
    """
    n = len(X)
    indices = np.random.permutation(n)
    X, y = X[indices], y[indices]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[start:end], y[start:end]
