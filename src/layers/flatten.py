"""
Couche Flatten.

Fait la transition entre les couches convolutionnelles (données en 3D)
et les couches denses (données en 1D).

Exemple :
  (batch, 14, 14, 16)  →  (batch, 14*14*16)  =  (batch, 3136)

Le backward n'a qu'à "re-former" le gradient dans la forme d'origine.
"""

import numpy as np
from .base import Layer


class Flatten(Layer):
    def forward(self, x):
        self.input_shape = x.shape
        batch = x.shape[0]
        return x.reshape(batch, -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
