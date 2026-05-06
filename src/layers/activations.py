"""
Fonctions d'activation.

Une activation introduit de la NON-LINÉARITÉ dans le réseau.
Sans ça, empiler des couches ne servirait à rien : la composition
de fonctions linéaires reste linéaire, et on ne pourrait pas
apprendre des patterns complexes comme les formes dans les poumons.

Chaque activation est aussi une couche (elle a un forward et un backward).
"""

import numpy as np
from .base import Layer


class ReLU(Layer):
    """
    ReLU : Rectified Linear Unit
      forward  : f(x) = max(0, x)
      backward : f'(x) = 1 si x > 0, sinon 0

    Intuition : on "éteint" les neurones négatifs. Ça force le réseau
    à n'activer que les features pertinentes.
    """

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        # Le gradient ne passe que là où l'entrée était positive
        return grad_output * (self.input > 0)


class Sigmoid(Layer):
    """
    Sigmoid : f(x) = 1 / (1 + e^(-x))
      backward : f'(x) = f(x) * (1 - f(x))

    Utilisé en sortie pour obtenir une probabilité entre 0 et 1.
    0 → NORMAL, 1 → PNEUMONIA
    """

    def forward(self, x):
        # Clip pour éviter overflow dans exp
        x_clipped = np.clip(x, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.output

    def backward(self, grad_output):
        s = self.output
        return grad_output * s * (1 - s)
