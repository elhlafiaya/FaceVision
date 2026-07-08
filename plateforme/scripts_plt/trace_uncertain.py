"""
trace_incertain.py
===================
Reproduit EXACTEMENT le mode "identification" de app_final.py :
  - 1 image sonde (probe) a la fois
  - comparee au template UNIQUE de chaque identite (images[0] du dossier test)
  - meilleur match (cosine) choisi -> SVM calibre du dataset correspondant -> decision

Contrairement a train_from_notebook.py (qui teste TOUTES les paires C(n,2) par
sujet pour calculer les metriques), ce script rejoue le protocole reel de la
plateforme : chaque image de test (sauf les templates eux-memes) est traitee
comme si elle avait ete uploadee dans la page Test, et on regarde dans quelle
zone (ACCEPTE / INCERTAIN / REJETE) elle tombe.

Objectif : obtenir la liste EXACTE des images/identites qui donnent INCERTAIN
avec la methode "template unique" utilisee par la plateforme, pour pouvoir les
re-uploader et verifier que le resultat affiche est identique.

Usage :
    python trace_incertain.py
    (a lancer depuis le dossier racine du projet, la ou se trouve data/ et models/)
"""
import os, json
import numpy as np
import joblib

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
DATA_DIR   = os.path.join(BASE, "data")
MARGIN     = 0.15

# ── Chargement modeles (identique a app_final.py) ──────────────────────────
def load_pkl(name):
    p = os.path.join(MODELS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Modele manquant : {p}")
    return joblib.load(p)

def load_json(name):
    p = os.path.join(MODELS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Meta manquante : {p}")
    return json.load(open(p))

MODELS = {
    "orl":  {"svm": load_pkl("orl_svm_calibrated.pkl"),  "scaler": load_pkl("orl_scaler.pkl"),
             "meta": load_json("orl_meta.json")},
    "yale": {"svm": load_pkl("yale_svm_calibrated.pkl"), "scaler": load_pkl("yale_scaler.pkl"),
             "meta": load_json("yale_meta.json")},
}

# ── Index + embeddings (identique a app_final.py) ──────────────────────────
def build_index(dataset):
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

def load_embedding_for_image(dataset, image_url):
    folder = "ORL_EMBEDDINGS" if dataset == "orl" else "YALE_EMBEDDINGS"
    rel = image_url.lstrip("/")
    parts = rel.split("/")
    person, fname = parts[-2], parts[-1]
    stem = os.path.splitext(fname)[0]
    npy_path = os.path.join(DATA_DIR, folder, "test", person, stem + ".npy")
    if not os.path.exists(npy_path):
        return None
    emb = np.load(npy_path)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-9)

def cosine_sim(e1, e2):
    return float(np.dot(e1, e2))

def build_feature_vector(e1, e2):
    return np.abs(e1 - e2).astype(np.float32)

def run_decision(dataset, fv):
    m = MODELS[dataset]
    X = m["scaler"].transform(fv.reshape(1, -1))
    proba = float(m["svm"].predict_proba(X)[0, 1])
    thresh = float(m["meta"].get("eer_threshold", 0.5))
    tlow, thigh = thresh - MARGIN, thresh + MARGIN
    tlow, thigh = max(0.05, tlow), min(0.95, thigh)
    if proba >= thigh:
        zone = "ACCEPTE"
    elif proba < tlow:
        zone = "REJETE"
    else:
        zone = "INCERTAIN"
    return proba, thresh, tlow, thigh, zone

# ── Construction des templates (images[0] par identite, comme app_final.py) ─
def build_templates():
    templates = {}  # (dataset, person) -> (image_url, embedding)
    for dataset in ("orl", "yale"):
        idx = build_index(dataset)
        for person, images in idx.items():
            if not images:
                continue
            img_url = images[0]
            emb = load_embedding_for_image(dataset, img_url)
            if emb is not None:
                templates[(dataset, person)] = (img_url, emb)
    return templates

def best_match(e_probe, templates):
    """Reproduit best_match_across_datasets() de app_final.py."""
    best = None
    for (dataset, person), (img_url, e_ref) in templates.items():
        sim = cosine_sim(e_probe, e_ref)
        if best is None or sim > best["sim"]:
            best = dict(sim=sim, dataset=dataset, person=person, image=img_url)
    return best

# ── Boucle principale : chaque image (hors templates) jouee comme "upload" ──
def main():
    templates = build_templates()
    print(f"[INFO] {len(templates)} templates charges (1 image par identite, ORL+Yale).\n")

    results = []
    for dataset in ("orl", "yale"):
        idx = build_index(dataset)
        for person, images in idx.items():
            for img_url in images:
                if img_url == templates.get((dataset, person), (None,))[0]:
                    continue  # on ne teste pas le template contre lui-meme
                e_probe = load_embedding_for_image(dataset, img_url)
                if e_probe is None:
                    continue
                match = best_match(e_probe, templates)
                e_ref = load_embedding_for_image(match["dataset"], match["image"])
                fv = build_feature_vector(e_ref, e_probe)
                proba, thresh, tlow, thigh, zone = run_decision(match["dataset"], fv)
                results.append(dict(
                    true_dataset=dataset, true_person=person, probe_image=img_url,
                    matched_dataset=match["dataset"], matched_person=match["person"],
                    matched_template=match["image"], cos_sim=match["sim"],
                    svm_proba=proba, eer_threshold=thresh, tlow=tlow, thigh=thigh,
                    zone=zone, correct_id=(match["dataset"] == dataset and match["person"] == person),
                ))

    total = len(results)
    incertains = [r for r in results if r["zone"] == "INCERTAIN"]

    print(f"[RESULTATS] {total} images sondes testees (methode plateforme : template unique)")
    print(f"  ACCEPTE   : {sum(r['zone']=='ACCEPTE' for r in results)}")
    print(f"  INCERTAIN : {len(incertains)}")
    print(f"  REJETE    : {sum(r['zone']=='REJETE' for r in results)}\n")

    print("=" * 100)
    print("DETAIL DES CAS INCERTAIN (a re-uploader sur la plateforme pour verifier)")
    print("=" * 100)
    for r in incertains:
        ok = "identification CORRECTE" if r["correct_id"] else "identification INCORRECTE"
        print(f"- Image sonde : {r['probe_image']}")
        print(f"    Vraie identite   : {r['true_person']} ({r['true_dataset'].upper()})")
        print(f"    Identite trouvee : {r['matched_person']} ({r['matched_dataset'].upper()}) [{ok}]")
        print(f"    Template compare : {r['matched_template']}")
        print(f"    cos_sim={r['cos_sim']:.4f}  proba={r['svm_proba']:.4f}  "
              f"seuil={r['eer_threshold']:.4f}  [Tlow={r['tlow']:.4f} Thigh={r['thigh']:.4f}]")
        print()

    out_path = os.path.join(BASE, "trace_incertain_results.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[OK] Resultats complets exportes vers : {out_path}")

if __name__ == "__main__":
    main()
