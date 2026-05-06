"""
Optimizer : SGD avec Momentum.

Stochastic Gradient Descent (SGD) :
  w = w - lr * grad_w

Problème du SGD pur : les gradients sont bruités (on calcule sur des mini-batches),
donc le réseau "zigzague" vers le minimum au lieu d'aller droit.

Solution : le Momentum.
On accumule une "vitesse" v qui est une moyenne mobile des gradients passés.
  v = momentum * v - lr * grad_w
  w = w + v

Intuition : si le gradient pointe souvent dans la même direction → on accélère.
Si ça oscille → les directions s'annulent → on ralentit automatiquement.

Hyperparamètres :
  - lr       : learning rate (ex: 0.01) — amplitude des corrections
  - momentum : inertie (ex: 0.9)       — 0 = SGD pur, proche de 1 = beaucoup d'inertie
"""

import numpy as np


class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def update(self, layers):
        """
        Met à jour les poids de toutes les couches.
        layers : liste des couches du modèle
        """
        for i, layer in enumerate(layers):
            params = layer.get_params()
            grads = layer.get_grads()

            for j, (param, grad) in enumerate(zip(params, grads)):
                if grad is None:
                    continue

                key = (i, j)
                if key not in self.velocities:
                    self.velocities[key] = np.zeros_like(param)

                v = self.velocities[key]
                v = self.momentum * v - self.lr * grad
                self.velocities[key] = v
                param += v
