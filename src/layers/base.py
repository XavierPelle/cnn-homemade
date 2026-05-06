"""
Classe de base pour toutes les couches du CNN.

Chaque couche doit implémenter :
  - forward(x)  : passe les données "vers l'avant" (calcule la sortie)
  - backward(grad) : passe le gradient "vers l'arrière" (calcule comment le réseau doit corriger ses poids)

C'est le principe fondamental de la backpropagation :
le réseau fait une prédiction (forward), mesure son erreur,
puis corrige chaque couche en remontant du dernier au premier (backward).
"""


class Layer:
    def forward(self, x):
        """
        Calcule la sortie de la couche à partir de l'entrée x.
        Doit stocker x dans self.input pour pouvoir l'utiliser dans backward.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Reçoit le gradient de la loss par rapport à la sortie de cette couche,
        et retourne le gradient par rapport à son entrée (pour la couche précédente).
        Met aussi à jour les gradients des poids internes (self.grad_weights, etc.)
        """
        raise NotImplementedError

    def get_params(self):
        """Retourne les paramètres entraînables (poids, biais) sous forme de liste."""
        return []

    def get_grads(self):
        """Retourne les gradients correspondants aux paramètres."""
        return []
