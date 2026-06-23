"""
Serveur Flask + SocketIO pour la visualisation en temps réel du CNN.
"""

import sys, os, threading, base64, io, json, random
import numpy as np
from PIL import Image
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import cm as _mcm
    _cmap = _mcm.get_cmap('inferno')
    def arr_to_b64_color(arr):
        a = arr.astype(np.float32)
        mn, mx = a.min(), a.max()
        if mx > mn:
            a = (a - mn) / (mx - mn)
        rgb = (_cmap(a)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode='RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
except ImportError:
    arr_to_b64_color = None

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import CNN
from src.losses import BinaryCrossEntropy
from src.optimizer import SGDMomentum
from src.data_loader import load_dataset, batch_generator

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_stop_flag    = threading.Event()
_training_lock = threading.Lock()


# ── Utilitaires ───────────────────────────────────────────────────────────────

def arr_to_b64(arr):
    """Array 2D numpy → PNG base64."""
    a = arr.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx > mn:
        a = (a - mn) / (mx - mn)
    img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_weight_payload(model):
    """
    Collecte les poids et biais de toutes les couches pour la visualisation.

    Couches du modèle :
      0 = Conv2D(8, 3, 1)    → filters (8,3,3,1), bias (8,)
      3 = Conv2D(16, 3, 8)   → filters (16,3,3,8), bias (16,)
      7 = Dense(3136, 128)   → weights (3136,128), bias (1,128)
      9 = Dense(128, 1)      → weights (128,1), bias (1,1)
    """
    conv1  = model.layers[0]
    conv2  = model.layers[3]
    dense1 = model.layers[7]
    dense2 = model.layers[9]

    # Conv1 : 8 filtres (3,3,1) → liste de matrices 3×3 (valeurs brutes pour JS)
    conv1_filters = conv1.filters[:, :, :, 0].tolist()   # (8,3,3)

    # Conv2 : 16 filtres (3,3,8) → moyenne sur les canaux d'entrée → (16,3,3)
    conv2_filters = np.mean(conv2.filters, axis=3).tolist()

    # Dense2 : poids de connexion Dense1→sortie, un par neurone Dense1 (128 valeurs)
    d2_weights = dense2.weights.flatten().tolist()

    return {
        "conv1_filters": conv1_filters,            # (8, 3, 3)
        "conv1_bias":    conv1.bias.tolist(),       # (8,)
        "conv2_filters": conv2_filters,            # (16, 3, 3)
        "conv2_bias":    conv2.bias.tolist(),       # (16,)
        "dense1_bias":   dense1.bias.flatten().tolist(),   # (128,)
        "dense2_weights":d2_weights,               # (128,)
        "dense2_bias":   float(dense2.bias.flatten()[0]),  # scalaire
    }


# ── Boucle d'entraînement ─────────────────────────────────────────────────────

def training_loop(config):
    try:
        _run_training(config)
    except Exception as e:
        import traceback
        socketio.emit("error", {"message": str(e) + "\n" + traceback.format_exc()})


def _run_training(config):
    epochs        = config.get("epochs", 10)
    max_per_class = config.get("max_per_class", 300)
    batch_size    = config.get("batch_size", 16)
    lr            = config.get("lr", 0.01)
    emit_every    = config.get("emit_every", 3)

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chest_xray")

    socketio.emit("status", {"message": "Chargement des images…"})

    X_train, y_train = load_dataset(
        os.path.join(DATA_DIR, "train"), input_size=64,
        max_per_class=max_per_class if max_per_class > 0 else None,
    )
    X_val, y_val = load_dataset(os.path.join(DATA_DIR, "val"), input_size=64)

    n_total    = len(y_train)
    n_pos      = int(np.sum(y_train == 1))
    n_neg      = int(np.sum(y_train == 0))
    pos_weight = n_total / (2 * n_pos)
    neg_weight = n_total / (2 * n_neg)

    model     = CNN(input_size=64)
    loss_fn   = BinaryCrossEntropy(pos_weight=pos_weight, neg_weight=neg_weight)
    optimizer = SGDMomentum(lr=lr, momentum=0.9)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    socketio.emit("training_start", {
        "epochs": epochs, "batch_size": batch_size,
        "n_train": n_total, "n_normal": n_neg, "n_pneumonie": n_pos,
        "pos_weight": round(pos_weight, 3), "neg_weight": round(neg_weight, 3),
    })

    # Envoyer les poids initiaux (aléatoires) dès le début
    socketio.emit("weights_update", get_weight_payload(model))

    for epoch in range(1, epochs + 1):
        if _stop_flag.is_set():
            break

        epoch_losses, epoch_correct, epoch_total = [], 0, 0
        batches = list(batch_generator(X_train, y_train, batch_size))

        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            if _stop_flag.is_set():
                break

            acts   = model.forward_with_activations(X_batch)
            y_pred = acts["output"]

            loss = loss_fn(y_pred, y_batch)
            epoch_losses.append(loss)

            model.backward(loss_fn.gradient(y_pred, y_batch))
            optimizer.update(model.layers)

            preds = (y_pred >= 0.5).astype(int)
            epoch_correct += int(np.sum(preds == y_batch))
            epoch_total   += len(y_batch)

            if batch_idx % emit_every == 0:
                fm1 = acts["after_relu1"][0]   # (62, 62, 8)
                fm2 = acts["after_relu2"][0]   # (29, 29, 16)
                d1  = acts["after_dense1"][0]  # (128,)
                prob = float(y_pred[0, 0])
                d1_norm = (d1 / (np.max(np.abs(d1)) + 1e-8)).tolist()

                socketio.emit("batch_update", {
                    "epoch": epoch, "total_epochs": epochs,
                    "batch": batch_idx + 1, "total_batches": len(batches),
                    "loss":        round(float(loss), 5),
                    "acc":         round(epoch_correct / max(epoch_total, 1), 4),
                    "input_image": arr_to_b64(X_batch[0, :, :, 0]),
                    "true_label":  int(y_batch[0, 0]),
                    "prediction":  round(prob, 4),
                    "conv1_maps":  [arr_to_b64(fm1[:, :, i]) for i in range(8)],
                    "conv2_maps":  [arr_to_b64(fm2[:, :, i]) for i in range(8)],
                    "dense_acts":  d1_norm,   # 128 valeurs normalisées [-1,1]
                })

        # ── Validation ────────────────────────────────────────────────────────
        val_losses, val_correct, val_total = [], 0, 0
        for Xb, yb in batch_generator(X_val, y_val, batch_size):
            yp = model.forward(Xb)
            val_losses.append(float(loss_fn(yp, yb)))
            val_correct += int(np.sum((yp >= 0.5).astype(int) == yb))
            val_total   += len(yb)

        tl = round(float(np.mean(epoch_losses)), 5)
        ta = round(epoch_correct / epoch_total, 4)
        vl = round(float(np.mean(val_losses)), 5)
        va = round(val_correct / val_total, 4)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        # Envoyer poids mis à jour à chaque fin d'epoch
        socketio.emit("weights_update", get_weight_payload(model))
        socketio.emit("epoch_end", {
            "epoch": epoch, "train_loss": tl, "train_acc": ta,
            "val_loss": vl, "val_acc": va, "history": history,
        })

    socketio.emit("training_done", {"message": "Entraînement terminé !"})


# ── Routes & événements SocketIO ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/xray/<label>")
def serve_xray(label):
    """Sert une vraie radio JPEG pour les slides du guide."""
    base = os.path.dirname(os.path.dirname(__file__))
    if label == "normal":
        path = os.path.join(base, "data", "chest_xray", "train", "NORMAL",
                            "NORMAL2-IM-0557-0001.jpeg")
    else:
        path = os.path.join(base, "data", "chest_xray", "train", "PNEUMONIA",
                            "person413_virus_844.jpeg")
    with open(path, "rb") as f:
        data = f.read()
    return data, 200, {"Content-Type": "image/jpeg", "Cache-Control": "max-age=3600"}


# ── Inférence ─────────────────────────────────────────────────────────────────

_inference_model = None
_origin_model    = None

ORIGIN_LABELS = ["NORMAL", "VIRUS", "BACTÉRIE"]   # classes 0/1/2 du modèle d'origine


def _base_dir():
    return os.path.dirname(os.path.dirname(__file__))


def _load_inference_model():
    """Modèle binaire 64×64 (NORMAL vs PNEUMONIE)."""
    global _inference_model
    if _inference_model is not None:
        return _inference_model
    weights = os.path.join(_base_dir(), "model_weights.npz")
    if not os.path.exists(weights):
        return None
    m = CNN(input_size=64)
    m.load(weights)
    _inference_model = m
    return m


def _load_origin_model():
    """Modèle 3 classes 48×48 (NORMAL / VIRUS / BACTÉRIE). Optionnel."""
    global _origin_model
    if _origin_model is not None:
        return _origin_model
    weights = os.path.join(_base_dir(), "origin_weights.npz")
    if not os.path.exists(weights):
        return None
    m = CNN(input_size=48, n_classes=3, dense_units=64)
    m.load(weights)
    _origin_model = m
    return m


def _build_inference_payload(img_gray, true_label=None):
    """
    Exécute les deux modèles sur une image PIL en niveaux de gris et construit
    le payload complet pour l'interface.

    - Modèle binaire (64×64) : feature maps + probabilité de pneumonie + contributions.
    - Modèle d'origine (48×48) : probabilités des 3 classes (si le modèle existe).

    true_label : 0/1 si connu (images du test set), None pour un upload.
    Retourne (payload, error_message).
    """
    model = _load_inference_model()
    if model is None:
        return None, "Modèle binaire non trouvé — lance 'python train.py' d'abord."

    # ── Modèle binaire ───────────────────────────────────────────────────────
    arr64 = np.array(img_gray.resize((64, 64)), dtype=np.float32) / 255.0
    x64   = arr64[:, :, np.newaxis][np.newaxis]            # (1, 64, 64, 1)

    acts = model.forward_with_activations(x64)
    pred = float(acts["output"][0, 0])

    fm1 = acts["after_relu1"][0]
    fm2 = acts["after_relu2"][0]
    d1  = acts["after_dense1"][0]

    logit = float(np.log(max(pred, 1e-7) / max(1 - pred, 1e-7)))
    colorize = arr_to_b64_color if arr_to_b64_color is not None else arr_to_b64

    d2_weights    = model.layers[9].weights.flatten()       # (128,)
    contributions = (d1 * d2_weights).tolist()              # contribution de chaque neurone au logit

    payload = {
        "prediction":    round(pred, 4),
        "logit":         round(logit, 3),
        "input_image":   arr_to_b64(arr64),
        "conv1_maps":    [colorize(fm1[:, :, i]) for i in range(8)],
        "conv2_maps":    [colorize(fm2[:, :, i]) for i in range(8)],
        "contributions": contributions,
        "true_label":    true_label,
    }

    # ── Modèle d'origine (3 classes) ─────────────────────────────────────────
    origin = _load_origin_model()
    if origin is not None:
        arr48 = np.array(img_gray.resize((48, 48)), dtype=np.float32) / 255.0
        x48   = arr48[:, :, np.newaxis][np.newaxis]
        probs = origin.forward(x48)[0]                      # (3,)
        payload["origin_probs"]  = [round(float(p), 4) for p in probs]
        payload["origin_labels"] = ORIGIN_LABELS

    return payload, None


@app.route("/test-images")
def test_images():
    """Retourne 12 images du test set (6 NORMAL + 6 PNEUMONIE) en miniature base64."""
    base = os.path.dirname(os.path.dirname(__file__))
    result = []
    for label_name, label_val in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        folder = os.path.join(base, "data", "chest_xray", "test", label_name)
        files  = [f for f in os.listdir(folder) if f.lower().endswith((".jpeg", ".jpg"))]
        random.shuffle(files)
        for fname in files[:20]:
            path = os.path.join(folder, fname)
            img  = Image.open(path).convert("L")
            thumb = img.copy()
            thumb.thumbnail((96, 96))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=75)
            result.append({
                "path":       f"{label_name}/{fname}",
                "label":      label_val,
                "label_name": label_name,
                "thumb":      base64.b64encode(buf.getvalue()).decode(),
            })
    random.shuffle(result)
    return json.dumps(result)


@socketio.on("run_inference")
def handle_inference(data):
    """Exécute l'inférence sur une image du test set et renvoie toutes les activations."""
    img_path = os.path.join(_base_dir(), "data", "chest_xray", "test", data["path"])
    try:
        img = Image.open(img_path).convert("L")
    except Exception as e:
        socketio.emit("inference_error", {"message": f"Image illisible : {e}"})
        return

    payload, err = _build_inference_payload(img, true_label=data.get("label"))
    if err:
        socketio.emit("inference_error", {"message": err})
        return
    socketio.emit("inference_result", payload)


@app.route("/infer-upload", methods=["POST"])
def infer_upload():
    """
    Inférence sur une radiographie envoyée par l'utilisateur (drag & drop).
    Retourne le même payload que l'événement SocketIO inference_result, en JSON.
    Le vrai label est inconnu → true_label = None.
    """
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "Aucune image reçue."}), 400
    try:
        img = Image.open(file.stream).convert("L")
    except Exception as e:
        return jsonify({"error": f"Fichier image invalide : {e}"}), 400

    payload, err = _build_inference_payload(img, true_label=None)
    if err:
        return jsonify({"error": err}), 500
    return jsonify(payload)


@socketio.on("start_training")
def on_start(config):
    with _training_lock:
        _stop_flag.clear()
        t = threading.Thread(target=training_loop, args=(config,), daemon=True)
        t.start()


@socketio.on("stop_training")
def on_stop():
    _stop_flag.set()
    socketio.emit("status", {"message": "Arrêt demandé…"})


if __name__ == "__main__":
    print("Serveur démarré → http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
