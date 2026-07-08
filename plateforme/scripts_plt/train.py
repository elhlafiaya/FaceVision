"""
Reproduit le pipeline du notebook 'notebook_corrige_avec_analyse_scientifique.ipynb'
pour generer les modeles + metadonnees utilises par la plateforme FaceVision v2.0.

Pipeline :
  1. Split 50/50 stratifie par identite (SEED=42)
  2. Pretraitement (CLAHE) + MTCNN (detection/alignement) + FaceNet 512D (normalise L2)
  3. Feature engineering : X = |e1 - e2| in R^512
  4. SVM RBF + GridSearchCV 5-fold + Calibration Platt-Sigmoid -> P(genuine)
  5. Decision ternaire (Tlow = seuil_EER-0.15, Thigh = seuil_EER+0.15)

Sources attendues (apres conversion/split) :
  data/ORL_MTCNN/{train,test}/<person>/<id>.png
  data/ORL_EMBEDDINGS/{train,test}/<person>/<id>.npy  (FaceNet 512D normalise L2)
  data/YALE_MTCNN / YALE_EMBEDDINGS (idem)

Sortie -> models/ :
  orl_svm_calibrated.pkl / orl_scaler.pkl / orl_meta.json / orl_seuil_meta.json
  yale_svm_calibrated.pkl / yale_scaler.pkl / yale_meta.json / yale_seuil_meta.json
"""
import os, random, json
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)

MARGIN = 0.15  # marge fixe autour du seuil EER pour la zone INCERTAIN


def load_img_paths(root):
    d = {}
    if not os.path.isdir(root): return d
    for p in sorted(os.listdir(root)):
        pp = os.path.join(root, p)
        if os.path.isdir(pp):
            d[p] = [os.path.join(pp, f) for f in sorted(os.listdir(pp))]
    return d


def load_emb_by_id(root):
    """Charge les embeddings .npy, normalises L2."""
    e = {}
    if not os.path.isdir(root): return e
    for p in os.listdir(root):
        pp = os.path.join(root, p)
        if not os.path.isdir(pp): continue
        for f in os.listdir(pp):
            if f.endswith('.npy'):
                emb = np.load(os.path.join(pp, f))
                norm = np.linalg.norm(emb)
                e[os.path.splitext(f)[0]] = emb / (norm + 1e-9)
    return e


def build_feature_vector(e1, e2):
    """X = |e1 - e2| in R^512 (difference absolue, invariante a l'ordre de la paire)."""
    return np.abs(e1 - e2).astype(np.float32)


def build_dataset(img_paths, embeddings, name='', seed=SEED):
    """Construit le dataset de paires genuine/impostor a partir du split."""
    rng_b = random.Random(seed)
    X, y = [], []
    persons = list(img_paths.keys())

    for p in persons:
        imgs = img_paths[p]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                id1 = os.path.splitext(os.path.basename(imgs[i]))[0]
                id2 = os.path.splitext(os.path.basename(imgs[j]))[0]
                if id1 not in embeddings or id2 not in embeddings: continue
                X.append(build_feature_vector(embeddings[id1], embeddings[id2]))
                y.append(1)

    n_gen = len(X)
    imp = 0
    for _ in range(n_gen * 15):
        if imp >= n_gen: break
        p1, p2 = rng_b.sample(persons, 2)
        im1 = rng_b.choice(img_paths[p1]); im2 = rng_b.choice(img_paths[p2])
        id1 = os.path.splitext(os.path.basename(im1))[0]
        id2 = os.path.splitext(os.path.basename(im2))[0]
        if id1 not in embeddings or id2 not in embeddings: continue
        X.append(build_feature_vector(embeddings[id1], embeddings[id2]))
        y.append(0)
        imp += 1

    print(f'  [{name}] Genuine:{n_gen} | Impostor:{imp} | Total:{n_gen + imp} | Features: |e1-e2| in R^512')
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def compute_eer(y_true, y_score, n=2000):
    thresholds = np.linspace(float(y_score.min()), float(y_score.max()), n)
    fars, frrs = [], []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1)); tn = np.sum((y_true == 0) & (yp == 0))
        fp = np.sum((y_true == 0) & (yp == 1)); fn = np.sum((y_true == 1) & (yp == 0))
        fars.append(fp / (fp + tn + 1e-9)); frrs.append(fn / (fn + tp + 1e-9))
    fars, frrs = np.array(fars), np.array(frrs)
    idx = np.argmin(np.abs(fars - frrs))
    return (fars[idx] + frrs[idx]) / 2, thresholds[idx], fars, frrs, thresholds


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0: continue
        conf_mean = y_prob[mask].mean()
        acc_mean = y_true[mask].mean()
        ece += (n_in_bin / len(y_prob)) * abs(conf_mean - acc_mean)
    return ece


def fixed_threshold_system(te_imgs, te_emb, name):
    """Baseline : similarite cosine + seuil fixe EER (embeddings deja normalises L2)."""
    def cos_sim(e1, e2):
        return float(np.dot(e1, e2))

    rng_f = np.random.RandomState(SEED)
    scores, labels = [], []
    persons = list(te_imgs.keys())
    for p in persons:
        imgs = te_imgs[p]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                id1 = os.path.splitext(os.path.basename(imgs[i]))[0]
                id2 = os.path.splitext(os.path.basename(imgs[j]))[0]
                if id1 not in te_emb or id2 not in te_emb: continue
                scores.append(cos_sim(te_emb[id1], te_emb[id2])); labels.append(1)
    n_gen = len(scores); imp = 0
    for _ in range(n_gen * 15):
        if imp >= n_gen: break
        p1, p2 = rng_f.choice(persons, 2, replace=False)
        im1 = rng_f.choice(te_imgs[p1]); im2 = rng_f.choice(te_imgs[p2])
        id1 = os.path.splitext(os.path.basename(im1))[0]
        id2 = os.path.splitext(os.path.basename(im2))[0]
        if id1 not in te_emb or id2 not in te_emb: continue
        scores.append(cos_sim(te_emb[id1], te_emb[id2])); labels.append(0)
        imp += 1

    scores = np.array(scores); labels = np.array(labels)
    eer, thresh, fars, frrs, thresholds = compute_eer(labels, scores)
    auc = roc_auc_score(labels, scores)
    idx = np.argmin(np.abs(thresholds - thresh))
    far_eer = fars[idx]; frr_eer = frrs[idx]
    acc = accuracy_score(labels, (scores >= thresh).astype(int))
    print(f'  SEUIL FIXE — {name} | AUC={auc:.4f} EER={eer*100:.2f}% FAR={far_eer:.4f} FRR={frr_eer:.4f} Acc={acc:.4f} thresh={thresh:.4f}')
    return dict(auc=auc, eer=eer, eer_threshold=float(thresh), far=float(far_eer), frr=float(frr_eer), acc=acc)


def train_svm_cv(X_tr, y_tr, X_te, y_te, name):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01]}
    gs = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=SEED),
                       param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)
    best_C = gs.best_params_['C']; best_g = gs.best_params_['gamma']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    svm_cv = SVC(kernel='rbf', C=best_C, gamma=best_g, probability=True, random_state=SEED)
    cv_auc = cross_val_score(svm_cv, X_tr, y_tr, cv=skf, scoring='roc_auc')

    svm_cal = CalibratedClassifierCV(
        SVC(kernel='rbf', C=best_C, gamma=best_g, probability=True, random_state=SEED),
        method='sigmoid', cv=5)
    svm_cal.fit(X_tr, y_tr)

    y_sc = svm_cal.predict_proba(X_te)[:, 1]
    eer, thresh, fars, frrs, thresholds = compute_eer(y_te, y_sc)
    auc = roc_auc_score(y_te, y_sc)
    acc = accuracy_score(y_te, (y_sc >= thresh).astype(int))
    idx = np.argmin(np.abs(thresholds - thresh))
    far_eer = fars[idx]; frr_eer = frrs[idx]

    brier = brier_score_loss(y_te, y_sc)
    ece = expected_calibration_error(y_te, y_sc, n_bins=10)

    tlow, thigh = thresh - MARGIN, thresh + MARGIN
    tlow, thigh = max(0.05, tlow), min(0.95, thigh)
    dec = np.where(y_sc >= thigh, 'ACCEPTE', np.where(y_sc < tlow, 'REJETE', 'INCERTAIN'))
    total = len(dec)
    n_acc = int((dec == 'ACCEPTE').sum())
    n_inc = int((dec == 'INCERTAIN').sum())
    n_rej = int((dec == 'REJETE').sum())

    print(f'  [SVM] {name} | best C={best_C} gamma={best_g} | CV AUC={cv_auc.mean():.4f}+/-{cv_auc.std():.4f}')
    print(f'        AUC={auc:.4f}  EER={eer*100:.2f}%  FAR={far_eer:.4f}  FRR={frr_eer:.4f}  Acc={acc:.4f}  thresh={thresh:.4f}')
    print(f'        Brier={brier:.4f}  ECE={ece:.4f}')
    print(f'        Zones: ACCEPTE={n_acc} ({n_acc/total*100:.1f}%)  INCERTAIN={n_inc} ({n_inc/total*100:.1f}%)  REJETE={n_rej} ({n_rej/total*100:.1f}%)')

    meta = dict(
        dataset=name, model='SVM RBF |e1-e2| 512D + Calibration Platt',
        auc=float(auc), eer=float(eer), eer_threshold=float(thresh),
        far=float(far_eer), frr=float(frr_eer), accuracy=float(acc),
        cv_auc_mean=float(cv_auc.mean()), cv_auc_std=float(cv_auc.std()),
        brier_score=float(brier), ece=float(ece),
        best_C=best_C, best_gamma=str(best_g),
        margin=MARGIN, tlow=float(tlow), thigh=float(thigh),
        n_train=len(X_tr), n_test=len(X_te), feature_dim=512,
        zones=dict(accepte=n_acc, incertain=n_inc, rejete=n_rej, total=total),
    )
    return svm_cal, meta


def main():
    print(f'SEED={SEED} — split 50/50 stratifie par identite attendu dans data/*_MTCNN/{{train,test}}')

    # ── ORL ──────────────────────────────────────────────────────────
    print('\n=== ORL ===')
    orl_tr_imgs = load_img_paths(os.path.join(DATA, 'ORL_MTCNN', 'train'))
    orl_te_imgs = load_img_paths(os.path.join(DATA, 'ORL_MTCNN', 'test'))
    orl_tr_emb  = load_emb_by_id(os.path.join(DATA, 'ORL_EMBEDDINGS', 'train'))
    orl_te_emb  = load_emb_by_id(os.path.join(DATA, 'ORL_EMBEDDINGS', 'test'))
    X_orl_tr, y_orl_tr = build_dataset(orl_tr_imgs, orl_tr_emb, 'ORL train')
    X_orl_te, y_orl_te = build_dataset(orl_te_imgs, orl_te_emb, 'ORL test')
    sc_orl = StandardScaler()
    X_orl_tr_sc = sc_orl.fit_transform(X_orl_tr)
    X_orl_te_sc = sc_orl.transform(X_orl_te)

    # ── Yale ─────────────────────────────────────────────────────────
    print('\n=== Yale ===')
    yal_tr_imgs = load_img_paths(os.path.join(DATA, 'YALE_MTCNN', 'train'))
    yal_te_imgs = load_img_paths(os.path.join(DATA, 'YALE_MTCNN', 'test'))
    yal_tr_emb  = load_emb_by_id(os.path.join(DATA, 'YALE_EMBEDDINGS', 'train'))
    yal_te_emb  = load_emb_by_id(os.path.join(DATA, 'YALE_EMBEDDINGS', 'test'))
    X_yal_tr, y_yal_tr = build_dataset(yal_tr_imgs, yal_tr_emb, 'Yale train')
    X_yal_te, y_yal_te = build_dataset(yal_te_imgs, yal_te_emb, 'Yale test')
    sc_yal = StandardScaler()
    X_yal_tr_sc = sc_yal.fit_transform(X_yal_tr)
    X_yal_te_sc = sc_yal.transform(X_yal_te)

    print('\n--- Seuil fixe (cosine) ---')
    seuil_orl  = fixed_threshold_system(orl_te_imgs, orl_te_emb, 'ORL')
    seuil_yale = fixed_threshold_system(yal_te_imgs, yal_te_emb, 'Yale')

    print('\n--- SVM RBF |e1-e2| 512D + Calibration Platt ---')
    svm_orl,  meta_orl  = train_svm_cv(X_orl_tr_sc, y_orl_tr, X_orl_te_sc, y_orl_te, 'ORL')
    svm_yale, meta_yale = train_svm_cv(X_yal_tr_sc, y_yal_tr, X_yal_te_sc, y_yal_te, 'Yale')

    joblib.dump(svm_orl,  os.path.join(OUT, 'orl_svm_calibrated.pkl'))
    joblib.dump(sc_orl,   os.path.join(OUT, 'orl_scaler.pkl'))
    joblib.dump(svm_yale, os.path.join(OUT, 'yale_svm_calibrated.pkl'))
    joblib.dump(sc_yal,   os.path.join(OUT, 'yale_scaler.pkl'))

    json.dump(meta_orl,  open(os.path.join(OUT, 'orl_meta.json'), 'w'), indent=2)
    json.dump(meta_yale, open(os.path.join(OUT, 'yale_meta.json'), 'w'), indent=2)
    json.dump(seuil_orl,  open(os.path.join(OUT, 'orl_seuil_meta.json'), 'w'), indent=2)
    json.dump(seuil_yale, open(os.path.join(OUT, 'yale_seuil_meta.json'), 'w'), indent=2)

    print('\n[OK] Modeles et metadonnees sauvegardes dans', OUT)


if __name__ == '__main__':
    main()
