"""
Chargement et prétraitement des images.

Étapes de prétraitement :
  1. Charger l'image en niveaux de gris (les poumons n'ont pas besoin de couleur)
  2. Redimensionner à input_size x input_size (le réseau attend une taille fixe)
  3. (option) Feature engineering :
       - equalize    → égalisation d'histogramme (rehausse le contraste des
                       opacités pulmonaires, très utile sur des radios sous/sur-exposées)
       - standardize → centrage-réduction par image (moyenne 0, écart-type 1)
                       au lieu de la simple division par 255
  4. Normaliser les pixels entre 0 et 1 (diviser par 255) si standardize=False
     → Sans normalisation, les gradients seraient instables (valeurs trop grandes)
  5. Ajouter la dimension du canal : (H, W) → (H, W, 1)

Labels :
  - NORMAL    → 0
  - PNEUMONIA → 1
"""

import os
import numpy as np
from PIL import Image, ImageOps


def _preprocess(img, input_size, equalize=False, standardize=False):
    """
    Applique le pipeline de prétraitement à une image PIL (niveaux de gris)
    et retourne un tableau (input_size, input_size, 1).

    equalize / standardize : voir l'en-tête du module (feature engineering).
    """
    img = img.resize((input_size, input_size))        # Redimensionner
    if equalize:
        img = ImageOps.equalize(img)                  # Égalisation d'histogramme
    arr = np.array(img, dtype=np.float32)
    if standardize:
        arr = (arr - arr.mean()) / (arr.std() + 1e-7) # Centrage-réduction par image
    else:
        arr = arr / 255.0                             # Normalisation [0, 1]
    return arr[:, :, np.newaxis]                       # (H, W) → (H, W, 1)


def load_dataset(data_dir, input_size=64, max_per_class=None,
                 equalize=False, standardize=False):
    """
    Charge les images depuis data_dir/NORMAL et data_dir/PNEUMONIA.

    equalize / standardize : options de feature engineering (cf. _preprocess).

    Retourne :
      X : (n_samples, input_size, input_size, 1)  — images prétraitées
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
                arr = _preprocess(img, input_size, equalize, standardize)
                images.append(arr)
                labels.append([label_value])
            except Exception as e:
                print(f"  Erreur sur {fname}: {e}")

    X = np.array(images, dtype=np.float32)   # (n, H, W, 1)
    y = np.array(labels, dtype=np.float32)   # (n, 1)

    # Mélanger les données (sinon le réseau voit d'abord tous les 0 puis tous les 1)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


#  Classification de l'ORIGINE de la pneumonie (bonus 3 classes)
#  ─────────────────────────────────────────────────────────────────────────────
#  L'information virus / bactérie est encodée dans le nom des fichiers du dossier
#  PNEUMONIA : "personX_virus_Y.jpeg" ou "personX_bacteria_Y.jpeg".
#  On distingue donc 3 classes :
#    0 = NORMAL, 1 = VIRUS, 2 = BACTÉRIE
ORIGIN_CLASSES = ["NORMAL", "VIRUS", "BACTERIA"]


def _origin_label(folder_name, fname):
    """Déduit la classe d'origine (0/1/2) à partir du dossier et du nom de fichier."""
    if folder_name == "NORMAL":
        return 0
    low = fname.lower()
    if "virus" in low:
        return 1
    if "bacteria" in low:
        return 2
    return None  # nom inattendu → ignoré


def load_dataset_origin(data_dir, input_size=64, max_per_class=None,
                        equalize=False, standardize=False):
    """
    Charge les images en 3 classes : NORMAL / VIRUS / BACTÉRIE.

    equalize / standardize : options de feature engineering (cf. _preprocess).

    Retourne :
      X : (n_samples, input_size, input_size, 1)  — images prétraitées
      y : (n_samples, 3)                           — labels one-hot
    """
    images = []
    labels = []
    counts = {0: 0, 1: 0, 2: 0}

    for folder_name in ("NORMAL", "PNEUMONIA"):
        folder = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dossier introuvable : {folder}")

        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpeg", ".jpg", ".png"))]

        for fname in files:
            label = _origin_label(folder_name, fname)
            if label is None:
                continue
            if max_per_class and counts[label] >= max_per_class:
                continue
            fpath = os.path.join(folder, fname)
            try:
                img = Image.open(fpath).convert("L")
                arr = _preprocess(img, input_size, equalize, standardize)
                images.append(arr)
                labels.append(label)
                counts[label] += 1
            except Exception as e:
                print(f"  Erreur sur {fname}: {e}")

    print(f"  Origine → NORMAL: {counts[0]}, VIRUS: {counts[1]}, BACTÉRIE: {counts[2]}")

    X = np.array(images, dtype=np.float32)
    y_int = np.array(labels, dtype=np.int64)
    # one-hot (n, 3)
    y = np.zeros((len(y_int), 3), dtype=np.float32)
    y[np.arange(len(y_int)), y_int] = 1.0

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
