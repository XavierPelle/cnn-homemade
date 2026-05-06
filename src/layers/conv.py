"""
Couche de Convolution 2D.

CONCEPT CLÉ :
Un filtre (kernel) de taille (kH, kW) "glisse" sur l'image.
À chaque position (i, j), on calcule le produit scalaire entre
le patch de l'image et le filtre → un seul nombre.
En faisant ça pour toute l'image, on obtient une "feature map".

Avec n_filters filtres, on obtient n_filters feature maps,
chacune détectant un type de feature différent
(bords horizontaux, verticaux, textures, etc.).

Dimensions :
  - input  : (batch, H, W, C)           C = canaux (1 pour grayscale)
  - output : (batch, H_out, W_out, n_filters)
  - H_out  = H - kH + 1  (sans padding)
  - W_out  = W - kW + 1

Backpropagation :
  - grad_filters : corrélation entre l'input et grad_output
  - grad_input   : "transposed convolution" (corrélation entre grad_output et filtres retournés)
  - grad_bias    : somme de grad_output sur les dims spatiales
"""

import numpy as np
from .base import Layer


class Conv2D(Layer):
    def __init__(self, n_filters, kernel_size, n_channels=1):
        """
        n_filters   : nombre de filtres (= profondeur de la sortie)
        kernel_size : taille du filtre carré (ex: 3 → filtre 3x3)
        n_channels  : profondeur de l'entrée (1 pour grayscale)
        """
        kH = kW = kernel_size
        # He initialization adaptée aux convolutions
        fan_in = kH * kW * n_channels
        self.filters = np.random.randn(n_filters, kH, kW, n_channels) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros(n_filters)

        self.grad_filters = None
        self.grad_bias = None
        self.kernel_size = kernel_size
        self.n_filters = n_filters

    def forward(self, x):
        """
        x : (batch_size, H, W, C)
        """
        self.input = x
        batch, H, W, C = x.shape
        kH = kW = self.kernel_size
        H_out = H - kH + 1
        W_out = W - kW + 1

        output = np.zeros((batch, H_out, W_out, self.n_filters))

        # Pour chaque filtre, on fait glisser une fenêtre sur l'image
        for f in range(self.n_filters):
            for i in range(H_out):
                for j in range(W_out):
                    # Patch de l'image à cette position : (batch, kH, kW, C)
                    patch = x[:, i:i+kH, j:j+kW, :]
                    # Produit scalaire avec le filtre f : somme sur kH, kW, C
                    output[:, i, j, f] = np.sum(patch * self.filters[f], axis=(1, 2, 3))
            output[:, :, :, f] += self.bias[f]

        return output

    def backward(self, grad_output):
        """
        grad_output : (batch, H_out, W_out, n_filters)
        Retourne grad_input : (batch, H, W, C)
        """
        x = self.input
        batch, H, W, C = x.shape
        kH = kW = self.kernel_size
        H_out, W_out = grad_output.shape[1], grad_output.shape[2]

        self.grad_filters = np.zeros_like(self.filters)
        self.grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(x)

        for f in range(self.n_filters):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[:, i:i+kH, j:j+kW, :]
                    # grad_output[:, i, j, f] : (batch,) → gradient scalaire pour ce pixel de sortie
                    g = grad_output[:, i, j, f]  # (batch,)

                    # Gradient du filtre : patch pondéré par g
                    # (batch, kH, kW, C) * (batch, 1, 1, 1) → somme sur batch
                    self.grad_filters[f] += np.sum(patch * g[:, np.newaxis, np.newaxis, np.newaxis], axis=0)

                    # Gradient de l'input : filtre "redistribué" sur le patch
                    grad_input[:, i:i+kH, j:j+kW, :] += self.filters[f] * g[:, np.newaxis, np.newaxis, np.newaxis]

            # Gradient du biais : somme sur batch et positions spatiales
            self.grad_bias[f] = np.sum(grad_output[:, :, :, f])

        return grad_input

    def get_params(self):
        return [self.filters, self.bias]

    def get_grads(self):
        return [self.grad_filters, self.grad_bias]
