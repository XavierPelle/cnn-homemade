"""
Couche Dense (Fully Connected).

Chaque neurone de cette couche est connecté à TOUS les neurones
de la couche précédente. C'est le réseau de neurones "classique".

Formule : output = input @ W + b
  - input  : (batch_size, n_input)
  - W      : (n_input, n_output)  ← les poids appris
  - b      : (1, n_output)        ← les biais
  - output : (batch_size, n_output)

Backpropagation (règle de la chaîne) :
  - grad_input   = grad_output @ W.T        → ce qu'on renvoie à la couche précédente
  - grad_weights = input.T @ grad_output    → comment corriger W
  - grad_bias    = sum(grad_output, axis=0) → comment corriger b
"""

import numpy as np
from .base import Layer


class Dense(Layer):
    def __init__(self, n_input, n_output):
        """
        Initialisation des poids avec "He initialization" :
        W ~ N(0, sqrt(2 / n_input))

        Pourquoi ? Si on initialise W trop grand → les activations explosent.
        Trop petit → le gradient disparaît (vanishing gradient).
        He init est calibré pour ReLU.
        """
        self.weights = np.random.randn(n_input, n_output) * np.sqrt(2.0 / n_input)
        self.bias = np.zeros((1, n_output))

        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output):
        self.grad_weights = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.weights.T

    def get_params(self):
        return [self.weights, self.bias]

    def get_grads(self):
        return [self.grad_weights, self.grad_bias]
