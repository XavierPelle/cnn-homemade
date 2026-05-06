"""
Boucle d'entraînement.

À chaque epoch :
  1. On découpe les données en mini-batches
  2. Pour chaque batch :
       a. Forward  : prédire
       b. Loss     : mesurer l'erreur
       c. Backward : calculer les gradients
       d. Update   : corriger les poids
  3. On mesure la loss et l'accuracy sur le batch
  4. Optionnel : on évalue sur les données de validation
"""

import numpy as np
from .data_loader import batch_generator


class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, X_train, y_train, epochs, batch_size, X_val=None, y_val=None):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            for X_batch, y_batch in batch_generator(X_train, y_train, batch_size):
                # ── Forward ──────────────────────────────────────────────
                y_pred = self.model.forward(X_batch)

                # ── Loss ─────────────────────────────────────────────────
                loss = self.loss_fn(y_pred, y_batch)
                epoch_losses.append(loss)

                # ── Backward ─────────────────────────────────────────────
                grad = self.loss_fn.gradient(y_pred, y_batch)
                self.model.backward(grad)

                # ── Update des poids ──────────────────────────────────────
                self.optimizer.update(self.model.layers)

                # ── Métriques du batch ────────────────────────────────────
                preds = (y_pred >= 0.5).astype(int)
                epoch_correct += np.sum(preds == y_batch)
                epoch_total += len(y_batch)

            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / epoch_total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            log = f"Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  acc={train_acc:.4f}"

            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                log += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"

            print(log)

        return history

    def evaluate(self, X, y, batch_size=32):
        """Évalue le modèle sur un jeu de données (sans mise à jour des poids)."""
        losses = []
        correct = 0
        total = 0

        for X_batch, y_batch in batch_generator(X, y, batch_size):
            y_pred = self.model.forward(X_batch)
            losses.append(self.loss_fn(y_pred, y_batch))
            preds = (y_pred >= 0.5).astype(int)
            correct += np.sum(preds == y_batch)
            total += len(y_batch)

        return np.mean(losses), correct / total
