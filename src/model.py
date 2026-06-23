"""
Modèle CNN séquentiel.

Architecture :
  Conv(8 filtres, 3x3) → ReLU → MaxPool(2x2)
  Conv(16 filtres, 3x3) → ReLU → MaxPool(2x2)
  Flatten
  Dense(128) → ReLU
  Dense(1) → Sigmoid

Pourquoi cette architecture ?
  - Les premières couches Conv+Pool détectent des features de bas niveau
    (bords, textures) et réduisent la taille spatiale.
  - Les couches suivantes combinent ces features pour détecter des patterns
    de plus haut niveau (formes dans les poumons).
  - La partie Dense finale fait la classification à partir de ces features.

Pour une image 64x64 en entrée :
  après Conv1 : 62x62x8
  après Pool1 : 31x31x8
  après Conv2 : 29x29x16
  après Pool2 : 14x14x16
  après Flatten : 3136
  après Dense1 : 128
  après Dense2 : 1  (probabilité de pneumonie)
"""

import numpy as np
from .layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, Sigmoid, Softmax


class CNN:
    def __init__(self, input_size=64, n_classes=1, n_filters1=8, n_filters2=16, dense_units=128):
        """
        input_size  : côté des images carrées en entrée
        n_classes   : 1 → binaire (Sigmoid, NORMAL vs PNEUMONIE)
                      ≥3 → multi-classes (Softmax, ex : NORMAL/VIRUS/BACTÉRIE)
        n_filters1  : nombre de filtres de la 1re couche conv
        n_filters2  : nombre de filtres de la 2e couche conv
        dense_units : neurones de la couche dense cachée
        Les hyperparamètres d'architecture sont exposés pour le tuning.
        """
        # Calcul dynamique de la taille après conv+pool
        s = input_size
        s = (s - 2)      # après Conv 3x3
        s = s // 2       # après MaxPool 2x2
        s = (s - 2)      # après Conv 3x3
        s = s // 2       # après MaxPool 2x2
        flatten_size = s * s * n_filters2

        self.input_size = input_size
        self.n_classes  = n_classes

        # Couche de sortie : Sigmoid (1 neurone) en binaire, Softmax sinon.
        if n_classes == 1:
            output_layer = [Dense(dense_units, 1), Sigmoid()]
        else:
            output_layer = [Dense(dense_units, n_classes), Softmax()]

        self.layers = [
            Conv2D(n_filters=n_filters1, kernel_size=3, n_channels=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Conv2D(n_filters=n_filters2, kernel_size=3, n_channels=n_filters1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(flatten_size, dense_units),
            ReLU(),
            *output_layer,
        ]

    def forward(self, x):
        """Passe les données à travers toutes les couches."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_with_activations(self, x):
        """
        Comme forward(), mais capture les activations intermédiaires.
        Utilisé pour la visualisation en temps réel.

        Retourne un dict avec :
          after_relu1   : feature maps après Conv1+ReLU  (batch, 62, 62, 8)
          after_relu2   : feature maps après Conv2+ReLU  (batch, 29, 29, 16)
          after_dense1  : activations Dense1+ReLU        (batch, 128)
          output        : probabilité finale Sigmoid      (batch, 1)

        Indices des couches :
          0=Conv1, 1=ReLU, 2=Pool1, 3=Conv2, 4=ReLU, 5=Pool2,
          6=Flatten, 7=Dense1, 8=ReLU, 9=Dense2, 10=Sigmoid
        """
        activations = {}
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i == 1:   activations['after_relu1']  = x
            if i == 4:   activations['after_relu2']  = x
            if i == 8:   activations['after_dense1'] = x
        activations['output'] = x
        return activations

    def backward(self, grad):
        """Remonter le gradient en sens inverse."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, x):
        """
        Binaire   : 1 (PNEUMONIE) ou 0 (NORMAL) selon le seuil 0.5.
        Multi-classes : indice de la classe la plus probable (argmax).
        """
        probs = self.forward(x)
        if self.n_classes == 1:
            return (probs >= 0.5).astype(int)
        return np.argmax(probs, axis=1)

    def save(self, path):
        """Sauvegarde les poids du modèle."""
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            for j, p in enumerate(layer_params):
                params[f"layer_{i}_param_{j}"] = p
        np.savez(path, **params)
        print(f"Modèle sauvegardé dans {path}.npz")

    def load(self, path):
        """Charge les poids depuis un fichier."""
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            for j, p in enumerate(layer_params):
                key = f"layer_{i}_param_{j}"
                if key in data:
                    p[:] = data[key]
        print(f"Modèle chargé depuis {path}")
