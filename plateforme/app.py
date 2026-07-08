"""
FaceVision — Backend Flask v2.0
Pipeline reproduit depuis notebook_corrige_avec_analyse_scientifique.ipynb :
  Split 50/50 stratifie par identite -> FaceNet 512D (InceptionResnetV1, VGGFace2)
  -> |e1-e2| in R^512 -> StandardScaler -> SVM RBF + Calibration Platt
  -> P(genuine) -> Decision ternaire (ACCEPTE / INCERTAIN / REJETE)
  via marge fixe +/-0.15 autour du seuil EER.

Pas de variables comportementales (plus de tentatives/echecs utilisateur) :
le systeme est purement visuel, base sur |e1-e2| in R^512.
"""
import os, json, random
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, send_from_directory

BASE       = os.path.dirname(os.path.abspath(__file__))
PLATFORM   = os.path.join(BASE, "platform")
MODELS_DIR = os.path.join(BASE, "models")
DATA_DIR   = os.path.join(BASE, "data")

app = Flask(__name__, static_folder=PLATFORM)

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

# ── Chargement modeles + metadonnees ──────────────────────────────────────
def load_pkl(name):
    p = os.path.join(MODELS_DIR, name)
    return joblib.load(p) if os.path.exists(p) else None

def load_json(name):
    p = os.path.join(MODELS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else {}

MODELS = {
    "orl":  {"svm": load_pkl("orl_svm_calibrated.pkl"),  "scaler": load_pkl("orl_scaler.pkl"),
             "meta": load_json("orl_meta.json"),  "seuil": load_json("orl_seuil_meta.json")},
    "yale": {"svm": load_pkl("yale_svm_calibrated.pkl"), "scaler": load_pkl("yale_scaler.pkl"),
             "meta": load_json("yale_meta.json"), "seuil": load_json("yale_seuil_meta.json")},
}

MARGIN = 0.15  # Tlow = thresh - MARGIN, Thigh = thresh + MARGIN

# ── FaceNet / MTCNN (optionnel — necessite torch + facenet-pytorch) ────────
_FACENET = {"mtcnn": None, "resnet": None, "available": None}

def get_facenet():
    if _FACENET["available"] is not None:
        return _FACENET["mtcnn"], _FACENET["resnet"]
    try:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn  = MTCNN(image_size=160, margin=20, post_process=True, device=device)
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        _FACENET.update(mtcnn=mtcnn, resnet=resnet, available=True)
        print("[FACENET] torch + facenet-pytorch charges -> embeddings live actives")
    except Exception as e:
        _FACENET.update(mtcnn=None, resnet=None, available=False)
        print(f"[FACENET] indisponible ({e}) -> upload d'image desactive pour le live extract, "
              f"utilisez la selection parmi les images du dataset")
    return _FACENET["mtcnn"], _FACENET["resnet"]

def embed_image_from_bgr(bgr_img):
    """Detecte/aligne le visage (MTCNN) et calcule l'embedding FaceNet 512D normalise L2."""
    mtcnn, resnet = get_facenet()
    if mtcnn is None or resnet is None:
        return None, "FaceNet indisponible sur ce serveur (torch / facenet-pytorch non installes)."
    import torch
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    if face is None:
        return None, "Aucun visage detecte dans l'image (fallback image entiere non applique cote API live)."
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0)).cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-9), None

# ── Index des sujets / images (depuis data/<DATASET>_MTCNN/test) ──────────
def build_index(dataset):
    """Construit l'index {person_id: [chemins images]} a partir du test set MTCNN."""
    folder = "ORL_MTCNN" if dataset == "orl" else "YALE_MTCNN"
    root = os.path.join(DATA_DIR, folder, "test")
    idx = {}
    if not os.path.isdir(root):
        return idx
    for p in sorted(os.listdir(root)):
        pp = os.path.join(root, p)
        if os.path.isdir(pp):
            imgs = sorted(os.listdir(pp))
            idx[p] = [f"/data/{folder}/test/{p}/{f}" for f in imgs]
    return idx

def img_url_to_path(url):
    """Convertit une URL /data/... renvoyee au frontend en chemin disque."""
    rel = url.lstrip("/")
    return os.path.join(BASE, rel)

def load_embedding_for_image(dataset, image_url):
    """Charge l'embedding .npy correspondant a une image du dataset (deja extrait, normalise L2)."""
    folder = "ORL_EMBEDDINGS" if dataset == "orl" else "YALE_EMBEDDINGS"
    rel = image_url.lstrip("/")
    parts = rel.split("/")  # data / ORL_MTCNN / test / person / file.png
    person, fname = parts[-2], parts[-1]
    stem = os.path.splitext(fname)[0]
    npy_path = os.path.join(DATA_DIR, folder, "test", person, stem + ".npy")
    if not os.path.exists(npy_path):
        return None
    emb = np.load(npy_path)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-9)

def cosine_sim(e1, e2):
    return float(np.dot(e1, e2))  # deja normalises L2

def build_feature_vector(e1, e2):
    """X = |e1 - e2| in R^512 — feature engineering du pipeline."""
    return np.abs(e1 - e2).astype(np.float32)

def run_decision(dataset, fv):
    """Applique StandardScaler + SVM calibre -> P(genuine), puis decision ternaire."""
    m = MODELS.get(dataset)
    if not m or m["svm"] is None or m["scaler"] is None:
        return None
    X = m["scaler"].transform(fv.reshape(1, -1))
    proba = float(m["svm"].predict_proba(X)[0, 1])
    # Seuil EER du SVM (echelle proba [0,1]) — utilise pour la zone ACCEPTE/INCERTAIN/REJETE.
    thresh = float(m["meta"].get("eer_threshold", 0.5))
    # Seuil EER du systeme a seuil fixe (echelle cosine [-1,1]) — utilise UNIQUEMENT
    # pour comparer a cos_score, ne doit jamais etre confondu avec 'thresh' ci-dessus.
    cos_thresh = float(m["seuil"].get("eer_threshold", 0.5))
    tlow, thigh = thresh - MARGIN, thresh + MARGIN
    tlow, thigh = max(0.05, tlow), min(0.95, thigh)
    if proba >= thigh:
        zone = "ACCEPTE"
    elif proba < tlow:
        zone = "REJETE"
    else:
        zone = "INCERTAIN"
    return dict(proba=proba, thresh=thresh, cos_thresh=cos_thresh, tlow=tlow, thigh=thigh, zone=zone)

def top_contributing_dims(fv, n=6):
    """Renvoie les n dimensions de |e1-e2| avec la plus grande valeur (les plus differentes)."""
    idx = np.argsort(fv)[::-1][:n]
    return [{"dim": f"e{int(i)}", "val": f"{float(fv[i]):.3f}"} for i in idx]

# ── Static / pages ─────────────────────────────────────────────────────────
@app.route("/")
def root():
    return send_from_directory(PLATFORM, "index.html")

@app.route("/<path:path>")
def static_files(path):
    full = os.path.join(PLATFORM, path)
    if os.path.exists(full):
        return send_from_directory(PLATFORM, path)
    data_full = os.path.join(BASE, path)
    if path.startswith("data/") and os.path.exists(data_full):
        return send_from_directory(BASE, path)
    return send_from_directory(PLATFORM, "index.html")

@app.route("/data/<path:path>")
def data_files(path):
    return send_from_directory(DATA_DIR, path)

# ── API: dataset listing (subjects + thumbnails) ──────────────────────────
@app.route("/api/dataset/<dataset>")
def api_dataset(dataset):
    idx = build_index(dataset)
    persons = [{"id": pid, "images": imgs, "n_images": len(imgs)} for pid, imgs in idx.items()]
    n_total = sum(len(v) for v in idx.values())
    stats = dict(
        subjects=len(idx),
        images_total=n_total,
        images_per_subject=round(n_total / len(idx), 1) if idx else 0,
    )
    return jsonify(dict(persons=persons, stats=stats))

def best_match_across_datasets(e_probe):
    """Identification 1:N — compare l'embedding sonde a UNE SEULE image de reference
    par identite enrolee (la 1ere du dossier test), comme un enrolement biometrique
    a template unique. Comparer a TOUTES les images de chaque sujet puis garder le
    meilleur score sur-estime artificiellement la confiance (on choisirait toujours
    la photo la plus favorable parmi 5-10 par sujet), ce qui fait disparaitre la
    zone INCERTAIN observee dans le notebook (evaluation paire par paire, sans
    selection de la meilleure image). Un seul template par identite reproduit
    fidelement le comportement du notebook."""
    best = None
    for dataset in ("orl", "yale"):
        idx = build_index(dataset)
        for person, images in idx.items():
            if not images:
                continue
            img_url = images[0]  # template unique par identite (pas de selection du meilleur match)
            e_ref = load_embedding_for_image(dataset, img_url)
            if e_ref is None:
                continue
            sim = cosine_sim(e_probe, e_ref)
            if best is None or sim > best["sim"]:
                best = dict(sim=sim, dataset=dataset, person=person, image=img_url, emb=e_ref)
    return best

# ── API: verification (paire reference vs sonde) ──────────────────────────
@app.route("/api/verify", methods=["POST"])
def api_verify():
    dataset = None
    ref_subject = ref_image = probe_image = None
    probe_file = None

    if request.content_type and "multipart/form-data" in request.content_type:
        dataset     = request.form.get("dataset")
        ref_subject = request.form.get("ref_subject")
        ref_image   = request.form.get("ref_image")
        probe_file  = request.files.get("probe_file")
    else:
        data = request.get_json(force=True)
        dataset     = data.get("dataset")
        ref_subject = data.get("ref_subject")
        ref_image   = data.get("ref_image")
        probe_image = data.get("probe_image")

    # ── Mode identification 1:N (page Test simplifiee : une seule image importee,
    #    aucun dataset/sujet de reference choisi -> on cherche la meilleure correspondance) ──
    if not dataset and probe_file is not None:
        buf = np.frombuffer(probe_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify(error="image invalide"), 400
        e_probe, err = embed_image_from_bgr(bgr)
        if e_probe is None:
            return jsonify(error=err), 400

        match = best_match_across_datasets(e_probe)
        if match is None:
            return jsonify(error="aucune identite enrolee trouvee (data/ vide)"), 500

        dataset = match["dataset"]
        fv = build_feature_vector(match["emb"], e_probe)
        dec = run_decision(dataset, fv)
        if dec is None:
            return jsonify(error="modele indisponible — lancez train_from_notebook.py"), 500

        meta = MODELS[dataset]["meta"]
        return jsonify(dict(
            svm_proba=dec["proba"], cos_score=match["sim"],
            eer_threshold=dec["thresh"], cos_threshold=dec["cos_thresh"],
            tlow=dec["tlow"], thigh=dec["thigh"], zone=dec["zone"],
            dataset=dataset,
            auc=f'{meta.get("auc", 0):.4f}' if meta.get("auc") else "—",
            eer=f'{meta.get("eer", 0)*100:.2f}%' if meta.get("eer") else "—",
            expected="Inconnu (image importee)",
            ref_img=match["image"],
            probe_img="",
            ref_name=f'{match["person"]} ({dataset.upper()}) — meilleure correspondance',
            probe_name="Image importee",
            top_dims=top_contributing_dims(fv),
        ))

    if dataset not in ("orl", "yale"):
        return jsonify(error="dataset invalide"), 400

    e_ref = load_embedding_for_image(dataset, ref_image)
    if e_ref is None:
        return jsonify(error="embedding reference introuvable"), 400

    probe_name = None
    if probe_file is not None:
        buf = np.frombuffer(probe_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify(error="image invalide"), 400
        e_probe, err = embed_image_from_bgr(bgr)
        if e_probe is None:
            return jsonify(error=err), 400
        probe_image = None
        probe_name = "Image importee"
    else:
        e_probe = load_embedding_for_image(dataset, probe_image)
        if e_probe is None:
            return jsonify(error="embedding sonde introuvable"), 400
        probe_person = probe_image.split("/")[-2]
        probe_name = f"{probe_person}" + (" (meme sujet)" if probe_person == ref_subject else " (autre sujet)")

    fv = build_feature_vector(e_ref, e_probe)
    dec = run_decision(dataset, fv)
    if dec is None:
        return jsonify(error="modele indisponible — lancez train_from_notebook.py"), 500

    cos = cosine_sim(e_ref, e_probe)
    meta = MODELS[dataset]["meta"]
    expected = "Genuine" if (probe_image and probe_image.split("/")[-2] == ref_subject) else "Impostor"

    return jsonify(dict(
        svm_proba=dec["proba"], cos_score=cos,
        eer_threshold=dec["thresh"], cos_threshold=dec["cos_thresh"],
        tlow=dec["tlow"], thigh=dec["thigh"], zone=dec["zone"],
        dataset=dataset,
        auc=f'{meta.get("auc", 0):.4f}' if meta.get("auc") else "—",
        eer=f'{meta.get("eer", 0)*100:.2f}%' if meta.get("eer") else "—",
        expected=expected,
        ref_img=f"/data/{'ORL_MTCNN' if dataset=='orl' else 'YALE_MTCNN'}/test/{ref_subject}/{os.path.basename(ref_image)}",
        probe_img=probe_image or "",
        ref_name=ref_subject,
        probe_name=probe_name,
        top_dims=top_contributing_dims(fv),
    ))



# ── API: free pair verification (image A vs image B, no enrollment needed) ──
@app.route("/api/verify_pair", methods=["POST"])
def api_verify_pair():
    """Compare deux images uploadées directement (A ↔ B) sans passer par une base enrôlée.
    Pipeline : MTCNN + FaceNet 512D pour chaque image → |eA - eB| → StandardScaler → SVM RBF calibré.

    Paramètres multipart:
      - image_a, image_b : fichiers images à comparer
      - dataset (optionnel) : "orl" ou "yale" — force l'utilisation d'un modèle spécifique.
        Si non fourni, les deux modèles sont testés et on retourne celui dont la probabilité
        est la PLUS PROCHE du seuil EER (pour ne pas masquer les cas INCERTAIN).
    """
    if not request.content_type or "multipart/form-data" not in request.content_type:
        return jsonify(error="Content-Type must be multipart/form-data"), 400

    file_a = request.files.get("image_a")
    file_b = request.files.get("image_b")
    forced_dataset = request.form.get("dataset")  # orl ou yale, optionnel

    if not file_a or not file_b:
        return jsonify(error="Deux images requises (image_a et image_b)"), 400

    # --- Extract embedding A ---
    buf_a = np.frombuffer(file_a.read(), dtype=np.uint8)
    bgr_a = cv2.imdecode(buf_a, cv2.IMREAD_COLOR)
    if bgr_a is None:
        return jsonify(error="image A invalide"), 400
    e_a, err = embed_image_from_bgr(bgr_a)
    if e_a is None:
        return jsonify(error=f"extraction A échouée: {err}"), 400

    # --- Extract embedding B ---
    buf_b = np.frombuffer(file_b.read(), dtype=np.uint8)
    bgr_b = cv2.imdecode(buf_b, cv2.IMREAD_COLOR)
    if bgr_b is None:
        return jsonify(error="image B invalide"), 400
    e_b, err = embed_image_from_bgr(bgr_b)
    if e_b is None:
        return jsonify(error=f"extraction B échouée: {err}"), 400

    # --- Build feature vector ---
    fv = build_feature_vector(e_a, e_b)

    def run_model(dataset):
        """Exécute le SVM pour un dataset donné, retourne (proba, zone, meta, seuil)."""
        m = MODELS.get(dataset)
        if not m or m["svm"] is None or m["scaler"] is None:
            return None
        X = m["scaler"].transform(fv.reshape(1, -1))
        proba = float(m["svm"].predict_proba(X)[0, 1])
        thresh = float(m["meta"].get("eer_threshold", 0.5))
        cos_thresh = float(m["seuil"].get("eer_threshold", 0.5))
        tlow, thigh = thresh - MARGIN, thresh + MARGIN
        tlow, thigh = max(0.05, tlow), min(0.95, thigh)
        if proba >= thigh:
            zone = "ACCEPTE"
        elif proba < tlow:
            zone = "REJETE"
        else:
            zone = "INCERTAIN"
        return dict(
            proba=proba, thresh=thresh, cos_thresh=cos_thresh,
            tlow=tlow, thigh=thigh, zone=zone, meta=m["meta"],
            dataset=dataset
        )

    # --- Choix du modèle ---
    if forced_dataset in ("orl", "yale"):
        # L'utilisateur a forcé un dataset spécifique
        result = run_model(forced_dataset)
        if result is None:
            return jsonify(error=f"modèle {forced_dataset} indisponible"), 500
    else:
        # Pas de dataset forcé : tester les deux et choisir celui dont la proba
        # est la PLUS PROCHE du seuil (pour ne pas masquer les INCERTAIN)
        results = []
        for ds in ("orl", "yale"):
            r = run_model(ds)
            if r is not None:
                results.append(r)

        if not results:
            return jsonify(error="aucun modèle disponible — lancez train_from_notebook.py"), 500

        # Choisir le résultat dont |proba - thresh| est le plus petit
        # (celui qui est le plus proche du seuil = le plus "incertain")
        result = min(results, key=lambda r: abs(r["proba"] - r["thresh"]))

    cos = cosine_sim(e_a, e_b)

    return jsonify(dict(
        svm_proba=result["proba"],
        cos_score=cos,
        eer_threshold=result["thresh"],
        cos_threshold=result["cos_thresh"],
        tlow=result["tlow"],
        thigh=result["thigh"],
        zone=result["zone"],
        dataset=result["dataset"],
        auc=f'{result["meta"].get("auc", 0):.4f}' if result["meta"].get("auc") else "—",
        eer=f'{result["meta"].get("eer", 0)*100:.2f}%' if result["meta"].get("eer") else "—",
        expected="Paire libre (test manuel)",
        ref_img="",
        probe_img="",
        ref_name="Image B",
        probe_name="Image A",
        top_dims=top_contributing_dims(fv),
    ))


@app.route("/api/facenet_status")
def api_facenet_status():
    mtcnn, resnet = get_facenet()
    return jsonify(available=(mtcnn is not None and resnet is not None))


if __name__ == "__main__":
    print("FaceVision v2.0 — http://localhost:5000")
    app.run(debug=False, use_reloader=False, port=5000)
