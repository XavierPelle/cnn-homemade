"""
Script d'entraînement principal.

Usage :
  python train.py

Paramètres modifiables :
  INPUT_SIZE    : taille des images en entrée (plus grand = plus précis mais plus lent)
  MAX_PER_CLASS : nombre max d'images par classe (None = toutes)
  BATCH_SIZE    : taille des mini-batches
  EPOCHS        : nombre de passages sur les données
  LR            : learning rate
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.model import CNN
from src.losses import BinaryCrossEntropy
from src.optimizer import SGDMomentum
from src.trainer import Trainer
from src.data_loader import load_dataset

# ── Hyperparamètres ────────────────────────────────────────────────────────────
INPUT_SIZE    = 64
MAX_PER_CLASS = None   # Toutes les images
BATCH_SIZE    = 16
EPOCHS        = 20
LR            = 0.01
MOMENTUM      = 0.9

DATA_DIR      = os.path.join(os.path.dirname(__file__), "data", "chest_xray")

# ── Chargement des données ─────────────────────────────────────────────────────
print("=== Chargement des données ===")
print("Train :")
X_train, y_train = load_dataset(
    os.path.join(DATA_DIR, "train"),
    input_size=INPUT_SIZE,
    max_per_class=MAX_PER_CLASS,
)

print("Validation :")
X_val, y_val = load_dataset(
    os.path.join(DATA_DIR, "val"),
    input_size=INPUT_SIZE,
    max_per_class=None,
)

n_normal    = int(np.sum(y_train == 0))
n_pneumonie = int(np.sum(y_train == 1))
n_total     = len(y_train)

print(f"\nTrain : {X_train.shape}  Labels : {y_train.shape}")
print(f"Val   : {X_val.shape}  Labels : {y_val.shape}")
print(f"Distribution train → NORMAL: {n_normal}, PNEUMONIE: {n_pneumonie}")

# ── Pondération des classes ────────────────────────────────────────────────────
# Rééquilibre l'influence de chaque classe dans la loss.
# La classe minoritaire (NORMAL) reçoit un poids plus élevé.
pos_weight = n_total / (2 * n_pneumonie)  # poids PNEUMONIE
neg_weight = n_total / (2 * n_normal)     # poids NORMAL
print(f"Poids → PNEUMONIE: {pos_weight:.3f}, NORMAL: {neg_weight:.3f}")

# ── Création du modèle ─────────────────────────────────────────────────────────
print("\n=== Architecture du CNN ===")
print("Conv8 → ReLU → Pool → Conv16 → ReLU → Pool → Flatten → Dense128 → ReLU → Dense1 → Sigmoid")

model     = CNN(input_size=INPUT_SIZE)
loss_fn   = BinaryCrossEntropy(pos_weight=pos_weight, neg_weight=neg_weight)
optimizer = SGDMomentum(lr=LR, momentum=MOMENTUM)
trainer   = Trainer(model, loss_fn, optimizer)

# ── Entraînement ───────────────────────────────────────────────────────────────
print(f"\n=== Entraînement ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}) ===")
history = trainer.train(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    X_val=X_val,
    y_val=y_val,
)

# ── Évaluation finale sur le test set ─────────────────────────────────────────
print("\n=== Évaluation finale (test set) ===")
X_test, y_test = load_dataset(
    os.path.join(DATA_DIR, "test"),
    input_size=INPUT_SIZE,
    max_per_class=None,
)
test_loss, test_acc = trainer.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f"Test loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")

# ── Sauvegarde ─────────────────────────────────────────────────────────────────
model.save("model_weights")
