"""
Couche Max Pooling.

But : réduire la taille spatiale des feature maps tout en gardant
les features les plus importantes (le maximum de chaque région).

Exemple avec pool_size=2 :
  [[1, 3, 2, 4],      →   [[3, 4],
   [5, 6, 7, 8],           [6, 8]]
   ...]

Avantages :
  1. Réduit le nombre de paramètres (moins de calculs)
  2. Apporte une invariance à la translation (si une feature
     se déplace légèrement, on la détecte quand même)

Backpropagation :
  Le gradient ne passe QUE par la position du maximum.
  Les autres positions reçoivent un gradient de 0.
  (On ne peut pas "apprendre" d'une valeur qu'on a ignorée.)
"""

import numpy as np
from .base import Layer


class MaxPool2D(Layer):
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, x):
        """
        x : (batch, H, W, C)
        output : (batch, H//pool_size, W//pool_size, C)
        """
        self.input = x
        batch, H, W, C = x.shape
        p = self.pool_size
        H_out = H // p
        W_out = W // p

        output = np.zeros((batch, H_out, W_out, C))

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*p:(i+1)*p, j*p:(j+1)*p, :]
                output[:, i, j, :] = np.max(patch, axis=(1, 2))

        return output

    def backward(self, grad_output):
        """
        On redistribue chaque gradient uniquement vers la position du max.
        """
        x = self.input
        batch, H, W, C = x.shape
        p = self.pool_size
        H_out = grad_output.shape[1]
        W_out = grad_output.shape[2]

        grad_input = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*p:(i+1)*p, j*p:(j+1)*p, :]  # (batch, p, p, C)
                # Masque : 1 là où se trouve le max, 0 ailleurs
                max_val = np.max(patch, axis=(1, 2), keepdims=True)  # (batch, 1, 1, C)
                mask = (patch == max_val)  # (batch, p, p, C)

                # En cas d'ex-aequo, on divise le gradient (comportement standard)
                mask = mask / np.sum(mask, axis=(1, 2), keepdims=True)

                g = grad_output[:, i, j, :]  # (batch, C)
                grad_input[:, i*p:(i+1)*p, j*p:(j+1)*p, :] += mask * g[:, np.newaxis, np.newaxis, :]

        return grad_input
