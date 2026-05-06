"""
Fonction de perte : Binary Cross-Entropy (BCE) avec pondération des classes.

Formule pondérée :
  loss = -mean( w1 * y * log(ŷ)  +  w0 * (1-y) * log(1-ŷ) )

  - w1 (pos_weight) : poids de la classe PNEUMONIE (y=1)
  - w0 (neg_weight) : poids de la classe NORMAL    (y=0)

Pourquoi pondérer ?
  Le dataset a 1341 NORMAL vs 3875 PNEUMONIE (ratio ~1:3).
  Sans pondération, le réseau minimise sa loss en disant toujours
  "PNEUMONIE" — il a raison 75% du temps sans rien apprendre.
  Les poids rééquilibrent l'influence de chaque classe :
    pos_weight = n_total / (2 * n_pneumonie)
    neg_weight = n_total / (2 * n_normal)

Correction de bug :
  La version précédente retournait (ŷ - y) dans gradient(), qui est
  le gradient COMBINÉ Sigmoid+BCE. Comme Sigmoid.backward() est une
  couche à part qui multiplie par ŷ*(1-ŷ), le gradient était appliqué
  deux fois. On retourne maintenant dL/dŷ (vrai gradient par rapport
  à la sortie Sigmoid), et Sigmoid.backward() fait correctement
  dL/dz = dL/dŷ * ŷ*(1-ŷ) = (ŷ - y) / batch_size.
"""

import numpy as np


class BinaryCrossEntropy:
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        """
        pos_weight : poids classe positive (PNEUMONIE, y=1)
        neg_weight : poids classe négative (NORMAL, y=0)
        Par défaut 1.0 = pas de pondération.
        """
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def __call__(self, y_pred, y_true):
        """
        y_pred : (batch, 1) — probabilités prédites
        y_true : (batch, 1) — labels réels (0 ou 1)
        """
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            self.pos_weight * y_true * np.log(y_pred) +
            self.neg_weight * (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def gradient(self, y_pred, y_true):
        """
        Retourne dL/dŷ — gradient par rapport à la SORTIE de Sigmoid.
        Sigmoid.backward() va ensuite multiplier par ŷ*(1-ŷ),
        ce qui donne le gradient pré-activation correct.
        """
        eps = 1e-8
        y_pred_c = np.clip(y_pred, eps, 1 - eps)
        batch_size = y_pred.shape[0]
        return -(
            self.pos_weight * y_true / y_pred_c -
            self.neg_weight * (1 - y_true) / (1 - y_pred_c)
        ) / batch_size
