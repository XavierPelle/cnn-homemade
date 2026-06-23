"""
Entraînement du modèle d'ORIGINE de la pneumonie (bonus 3 classes).

Classes : 0 = NORMAL, 1 = VIRUS, 2 = BACTÉRIE

Le modèle est un CNN à sortie Softmax (3 neurones) entraîné avec une
cross-entropy catégorielle pondérée par l'inverse de la fréquence des classes
(le dataset est déséquilibré : BACTÉRIE > VIRUS ≈ NORMAL).

Les poids sont sauvegardés dans origin_weights.npz pour être rechargés
directement dans le notebook (pas besoin de ré-entraîner).

Usage :
  python train_origin.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.model import CNN
from src.losses import CategoricalCrossEntropy
from src.optimizer import SGDMomentum
from src.trainer import Trainer
from src.data_loader import load_dataset_origin, ORIGIN_CLASSES

# ── Hyperparamètres ──────────────────────────────────────────────────────────
INPUT_SIZE    = 48
MAX_PER_CLASS = 1000   # équilibre les 3 classes et garde un temps raisonnable
BATCH_SIZE    = 32
EPOCHS        = 12
LR            = 0.01
MOMENTUM      = 0.9
DENSE_UNITS   = 64     # plus petit que 128 → moins de surapprentissage
VAL_FRAC      = 0.2    # part du train réservée à la validation (early stopping)
SEED          = 42

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "chest_xray")


def main():
    np.random.seed(SEED)

    print("=== Chargement (origine, 3 classes) ===")
    X_all, y_all = load_dataset_origin(
        os.path.join(DATA_DIR, "train"), input_size=INPUT_SIZE, max_per_class=MAX_PER_CLASS,
    )

    # Split train/validation STRATIFIÉ (le test n'est jamais touché ici : il est
    # réservé à l'évaluation finale honnête dans le notebook).
    labels = np.argmax(y_all, axis=1)
    val_idx, tr_idx = [], []
    rng = np.random.RandomState(SEED)
    for c in range(3):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * VAL_FRAC)
        val_idx.extend(idx[:cut]); tr_idx.extend(idx[cut:])
    rng.shuffle(tr_idx); rng.shuffle(val_idx)
    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_val,   y_val   = X_all[val_idx], y_all[val_idx]

    print(f"\nTrain : {X_train.shape}  Validation : {X_val.shape}")

    # Pondération par l'inverse de la fréquence : w_c = n_total / (n_classes * n_c)
    counts = y_train.sum(axis=0)            # (3,)
    n_total = counts.sum()
    class_weights = n_total / (len(counts) * counts)
    print("Comptes train :", dict(zip(ORIGIN_CLASSES, counts.astype(int))))
    print("Poids classes :", dict(zip(ORIGIN_CLASSES, np.round(class_weights, 3))))

    model     = CNN(input_size=INPUT_SIZE, n_classes=3, dense_units=DENSE_UNITS)
    loss_fn   = CategoricalCrossEntropy(class_weights=class_weights)
    optimizer = SGDMomentum(lr=LR, momentum=MOMENTUM)
    trainer   = Trainer(model, loss_fn, optimizer)

    print(f"\n=== Entraînement ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}) ===")
    print("On sauvegarde le MEILLEUR modèle selon l'accuracy de validation (early stopping).")

    import io, contextlib
    best_val_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        # une epoch d'entraînement (on masque le log interne du Trainer)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train(X_train, y_train, epochs=1, batch_size=BATCH_SIZE)
        val_loss, val_acc = trainer.evaluate(X_val, y_val, batch_size=BATCH_SIZE)
        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save("origin_weights")   # écrase avec le meilleur à ce stade
            flag = "  ← meilleur (sauvegardé)"
        print(f"Epoch {epoch:2d}/{EPOCHS}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{flag}")

    print(f"\nMeilleure accuracy de validation : {best_val_acc:.4f}")
    print("Le fichier origin_weights.npz contient le meilleur modèle.")


if __name__ == "__main__":
    main()
