"""
Génère la SYNTHÈSE PDF du projet ZOIDBERG2.0 (synthesis.pdf).

Le document récapitule, sur quelques pages :
  - le contexte et la démarche,
  - les résultats du modèle binaire (NORMAL vs PNEUMONIE) sur le test set,
  - les résultats du modèle 3 classes (origine : NORMAL / VIRUS / BACTÉRIE),
  - les figures clés (matrices de confusion, courbes ROC).

On utilise matplotlib (PdfPages) — aucune dépendance LaTeX/HTML nécessaire.
Les chiffres sont calculés en rechargeant les modèles pré-entraînés, donc
reproductibles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.model import CNN
from src.data_loader import load_dataset, load_dataset_origin, ORIGIN_CLASSES
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             roc_auc_score)

DATA = os.path.join("data", "chest_xray")


def predict_proba(model, X, bs=32):
    return np.vstack([model.forward(X[i:i+bs]) for i in range(0, len(X), bs)])


def compute_binary():
    X, y = load_dataset(os.path.join(DATA, "test"), input_size=64)
    m = CNN(input_size=64)
    m.load("model_weights.npz")
    proba = predict_proba(m, X)
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred),
        "rec": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, proba),
        "cm": confusion_matrix(y, pred),
        "roc": (fpr, tpr),
        "n": len(y),
    }


def compute_origin():
    X, y = load_dataset_origin(os.path.join(DATA, "test"), input_size=48)
    m = CNN(input_size=48, n_classes=3, dense_units=64)
    m.load("origin_weights.npz")
    proba = predict_proba(m, X)
    pred = np.argmax(proba, axis=1)
    true = np.argmax(y, axis=1)
    rocs = {}
    for c, name in enumerate(ORIGIN_CLASSES):
        fpr, tpr, _ = roc_curve((true == c).astype(int), proba[:, c])
        rocs[name] = (fpr, tpr, auc(fpr, tpr))
    return {
        "acc": accuracy_score(true, pred),
        "auc_macro": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "cm": confusion_matrix(true, pred),
        "rocs": rocs,
        "per_class": {
            name: {
                "prec": precision_score(true, pred, labels=[c], average="micro"),
                "rec": recall_score(true, pred, labels=[c], average="micro"),
            } for c, name in enumerate(ORIGIN_CLASSES)
        },
        "n": len(true),
    }


def draw_confusion(ax, cm, labels, title, cmap):
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.set_title(title, fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=10)


def main():
    print("Calcul des métriques (binaire)…")
    b = compute_binary()
    print("Calcul des métriques (origine 3 classes)…")
    o = compute_origin()

    with PdfPages("synthesis.pdf") as pdf:
        # ── Page 1 : titre + contexte + résultats clés ──────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.5, 0.94, "ZOIDBERG2.0 — Synthèse", ha="center", fontsize=20, weight="bold")
        fig.text(0.5, 0.905, "Détection de pneumonie sur radios thoraciques (CNN « from scratch »)",
                 ha="center", fontsize=11, style="italic")

        intro = (
            "Objectif : aider au diagnostic de la pneumonie à partir de radiographies thoraciques,\n"
            "à l'aide d'un réseau de neurones convolutif entièrement implémenté en NumPy\n"
            "(forward + rétropropagation, sans librairie de deep learning).\n\n"
            "Démarche :\n"
            "  • Exploration des données et gestion du déséquilibre (loss pondérée).\n"
            "  • Procédure train / validation / test rigoureuse.\n"
            "  • Comparaison simple train-test split vs validation croisée (k-fold).\n"
            "  • Tuning des hyperparamètres sur un set de validation.\n"
            "  • PCA pour l'exploration et la réduction de dimension.\n"
            "  • Feature engineering : égalisation d'histogramme / standardisation (gain d'AUC).\n"
            "  • Métriques adaptées au médical : recall élevé et ROC-AUC.\n"
            "  • Bonus : prédiction de l'origine (NORMAL / VIRUS / BACTÉRIE)."
        )
        fig.text(0.08, 0.70, intro, ha="left", va="top", fontsize=10.5)

        # Tableau résultats binaire
        fig.text(0.08, 0.50, "Modèle binaire — NORMAL vs PNEUMONIE  (test : %d images)" % b["n"],
                 fontsize=12, weight="bold")
        rows = [
            ["Accuracy", f"{b['acc']:.3f}"],
            ["Precision", f"{b['prec']:.3f}"],
            ["Recall (sensibilité)", f"{b['rec']:.3f}"],
            ["F1-score", f"{b['f1']:.3f}"],
            ["ROC-AUC", f"{b['auc']:.3f}"],
        ]
        ax_t = fig.add_axes([0.08, 0.30, 0.40, 0.17]); ax_t.axis("off")
        t = ax_t.table(cellText=rows, colLabels=["Métrique", "Valeur"],
                       cellLoc="left", loc="center")
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)

        fig.text(0.54, 0.50, "Origine — 3 classes  (test : %d images)" % o["n"],
                 fontsize=12, weight="bold")
        rows2 = [
            ["Accuracy", f"{o['acc']:.3f}"],
            ["AUC macro (OvR)", f"{o['auc_macro']:.3f}"],
        ]
        for name in ORIGIN_CLASSES:
            pc = o["per_class"][name]
            rows2.append([f"Recall {name}", f"{pc['rec']:.3f}"])
        ax_t2 = fig.add_axes([0.54, 0.30, 0.40, 0.17]); ax_t2.axis("off")
        t2 = ax_t2.table(cellText=rows2, colLabels=["Métrique", "Valeur"],
                         cellLoc="left", loc="center")
        t2.auto_set_font_size(False); t2.set_fontsize(10); t2.scale(1, 1.5)

        note = (
            "Lecture : sur la détection binaire, le modèle privilégie un recall élevé — il rate très\n"
            "peu de pneumonies, au prix de quelques fausses alertes, ce qui est le bon compromis en\n"
            "contexte médical. L'AUC (indépendante du seuil et du déséquilibre) confirme un réel\n"
            "pouvoir discriminant. La tâche à 3 classes est plus difficile : la distinction\n"
            "virus/bactérie reste partiellement confondue, ce qui est attendu radiologiquement."
        )
        fig.text(0.08, 0.24, note, ha="left", va="top", fontsize=9.5)
        fig.text(0.5, 0.04, "Détail complet, code et figures : zoidberg.ipynb / zoidberg.html",
                 ha="center", fontsize=9, style="italic", color="gray")
        pdf.savefig(fig); plt.close(fig)

        # ── Page 2 : figures binaire ────────────────────────────────────────
        fig, ax = plt.subplots(1, 2, figsize=(11.69, 8.27))  # A4 paysage
        fig.suptitle("Modèle binaire — diagnostic des résultats", fontsize=15, weight="bold")
        draw_confusion(ax[0], b["cm"], ["NORMAL", "PNEUMONIE"],
                       "Matrice de confusion", "Blues")
        fpr, tpr = b["roc"]
        ax[1].plot(fpr, tpr, color="#D7544C", lw=2, label=f"ROC (AUC = {b['auc']:.3f})")
        ax[1].plot([0, 1], [0, 1], ls="--", color="gray", label="hasard")
        ax[1].set_xlabel("Taux de faux positifs"); ax[1].set_ylabel("Taux de vrais positifs")
        ax[1].set_title("Courbe ROC"); ax[1].legend(loc="lower right"); ax[1].grid(alpha=.3)
        pdf.savefig(fig); plt.close(fig)

        # ── Page 3 : figures origine ────────────────────────────────────────
        fig, ax = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("Origine de la pneumonie (3 classes) — diagnostic", fontsize=15, weight="bold")
        draw_confusion(ax[0], o["cm"], ORIGIN_CLASSES, "Matrice de confusion", "Purples")
        for name, (fpr, tpr, a) in o["rocs"].items():
            ax[1].plot(fpr, tpr, lw=2, label=f"{name} (AUC={a:.2f})")
        ax[1].plot([0, 1], [0, 1], ls="--", color="gray")
        ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
        ax[1].set_title(f"ROC one-vs-rest (AUC macro = {o['auc_macro']:.3f})")
        ax[1].legend(loc="lower right"); ax[1].grid(alpha=.3)
        pdf.savefig(fig); plt.close(fig)

    print("synthesis.pdf généré.")


if __name__ == "__main__":
    main()
