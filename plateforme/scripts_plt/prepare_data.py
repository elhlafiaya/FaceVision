"""
Reproduit EXACTEMENT les Cellules 2 et 3 du notebook
'notebook_corrige_avec_analyse_scientifique.ipynb' pour generer, a partir des
images brutes ORL/Yale, les dossiers attendus par train_from_notebook.py et app.py :

  data/ORL_MTCNN/{train,test}/<person>/<id>.png         (visages alignes 160x160)
  data/ORL_EMBEDDINGS/{train,test}/<person>/<id>.npy     (FaceNet 512D, normalise L2)
  data/YALE_MTCNN / YALE_EMBEDDINGS                       (idem)

Pipeline (identique au notebook) :
  1. Conversion Yale (GIF/PGM sans extension) -> PNG (Pillow, magic bytes)
  2. Split 50/50 stratifie par identite (SEED=42, reproductible)
  3. Pretraitement : resize 160x160 + CLAHE (canal L, LAB)
  4. MTCNN : detection + alignement facial (landmarks, fallback image entiere documente)
  5. FaceNet InceptionResnetV1 (VGGFace2) -> embedding 512D, normalise L2

Pre-requis :
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install facenet-pytorch opencv-python-headless pillow tqdm numpy

Donnees brutes attendues (a adapter si besoin, voir RAW_* ci-dessous) :
  DATA/ORL_DATA/<person>/<image>.pgm        (40 sujets x 10 images, format ORL/AT&T classique)
  DATA/Yale_grouped/<person>/<image...>     (15 sujets, fichiers Yale sans extension ou .gif/.pgm)

Usage :
  python prepare_data.py
"""
import os, shutil, random, warnings
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Chemins (modifier ici si vos dossiers bruts sont ailleurs) ────────────
ORL_RAW_DIR   = os.path.join(BASE, 'DATA', 'ORL_DATA')
YALE_RAW_DIR  = os.path.join(BASE, 'DATA', 'Yale_grouped')

YALE_CONV_DIR  = os.path.join(BASE, 'YALE_converted')
ORL_SPLIT_DIR  = os.path.join(BASE, 'ORL_SPLIT')
YALE_SPLIT_DIR = os.path.join(BASE, 'YALE_SPLIT')

ORL_PRE_DIR    = os.path.join(BASE, 'ORL_PREPROCESSED')
YALE_PRE_DIR   = os.path.join(BASE, 'YALE_PREPROCESSED')

DATA_OUT = os.path.join(BASE, 'data')
ORL_MTCNN_DIR   = os.path.join(DATA_OUT, 'ORL_MTCNN')
YALE_MTCNN_DIR  = os.path.join(DATA_OUT, 'YALE_MTCNN')
ORL_EMB_DIR     = os.path.join(DATA_OUT, 'ORL_EMBEDDINGS')
YALE_EMB_DIR    = os.path.join(DATA_OUT, 'YALE_EMBEDDINGS')

VALID_EXT = {'.pgm', '.jpg', '.jpeg', '.png', '.bmp'}
IMG_SIZE = 160

YALE_EXPRESSIONS = {
    '.centerlight', '.glasses', '.happy', '.leftlight',
    '.noglasses', '.normal', '.rightlight', '.sad',
    '.sleepy', '.surprised', '.wink'
}


# ═══════════════════════════════════════════════════════════════════════
#  CELLULE 2 — Split 50/50 des donnees brutes
# ═══════════════════════════════════════════════════════════════════════
def read_any_image(path):
    """Lecture robuste via Pillow (magic bytes). Gere GIF, PGM, BMP, TIFF, fichiers Yale sans extension."""
    try:
        with Image.open(path) as im:
            if im.mode not in ('L', 'RGB', 'RGBA'):
                im = im.convert('L')
            arr = np.array(im)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                arr = 255.0 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                arr = arr.astype(np.uint8)
            return arr
    except Exception:
        return None


def convert_yale_extensions_subfolders(yale_raw_dir, yale_converted_dir):
    """Convertit les fichiers Yale (GIF/PGM sans extension) en PNG lisibles par OpenCV."""
    os.makedirs(yale_converted_dir, exist_ok=True)
    converted = 0; failed = 0
    for subj in sorted(os.listdir(yale_raw_dir)):
        subj_dir = os.path.join(yale_raw_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        out_dir = os.path.join(yale_converted_dir, subj)
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(os.listdir(subj_dir)):
            src = os.path.join(subj_dir, f)
            if os.path.isdir(src):
                continue
            name, ext = os.path.splitext(f)
            ext_lower = ext.lower()
            if ext_lower in {'.png', '.jpg', '.jpeg', '.pgm', '.bmp'}:
                dst_name = name + '.png'
            elif ext_lower in YALE_EXPRESSIONS:
                dst_name = name + '_' + ext_lower.lstrip('.') + '.png'
            elif ext == '':
                dst_name = name + '.png'
            else:
                dst_name = name + '_' + ext_lower.lstrip('.') + '.png'
            dst = os.path.join(out_dir, dst_name)
            arr = read_any_image(src)
            if arr is None:
                print(f'  Echec : {src}'); failed += 1; continue
            if len(arr.shape) == 2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.shape[2] == 4:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst, bgr)
            converted += 1
    print(f'  Yale : {converted} convertis, {failed} echecs')
    return yale_converted_dir


def split_dataset_5050(raw_dir, split_dir, name='', seed=SEED):
    """Split 50/50 PAR IDENTITE : chaque sujet va entierement en train OU en test,
    jamais les deux (split disjoint par identite, pas par image).
    C'est indispensable pour une evaluation honnete d'un systeme de
    verification/identification faciale : le test doit porter sur des
    visages jamais vus pendant l'entrainement, pas sur d'autres photos
    d'une identite deja connue du SVM."""
    rng_s = random.Random(seed)
    os.makedirs(os.path.join(split_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'test'), exist_ok=True)

    subjects = [s for s in sorted(os.listdir(raw_dir))
                if os.path.isdir(os.path.join(raw_dir, s))]
    rng_s.shuffle(subjects)
    n_train_subj = (len(subjects) + 1) // 2
    train_subjects = set(subjects[:n_train_subj])
    test_subjects = set(subjects[n_train_subj:])

    total_train = 0; total_test = 0
    for subj in subjects:
        subj_path = os.path.join(raw_dir, subj)
        imgs = [f for f in sorted(os.listdir(subj_path))
                if os.path.splitext(f)[1].lower() in VALID_EXT]
        if len(imgs) == 0:
            continue
        split = 'train' if subj in train_subjects else 'test'
        out = os.path.join(split_dir, split, subj)
        os.makedirs(out, exist_ok=True)
        for f in imgs:
            shutil.copy2(os.path.join(subj_path, f), os.path.join(out, f))
        if split == 'train':
            total_train += len(imgs)
        else:
            total_test += len(imgs)

    ratio = total_train / (total_train + total_test) * 100 if (total_train + total_test) > 0 else 0
    print(f'  [{name}] {len(train_subjects)} sujets train | {len(test_subjects)} sujets test '
          f'| Images train:{total_train} | Images test:{total_test} | {ratio:.1f}%/{100-ratio:.1f}%')
    print(f'  [{name}] IMPORTANT : sujets train et test disjoints (aucune identite commune).')


# ═══════════════════════════════════════════════════════════════════════
#  CELLULE 3 — Pretraitement (CLAHE) + MTCNN + FaceNet 512D
# ═══════════════════════════════════════════════════════════════════════
def load_img(path):
    """Lecture robuste (np.fromfile + imdecode, fallback imageio pour PGM 16-bit)."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None
    if img is None:
        try:
            import imageio
            img = np.array(imageio.imread(path))
        except Exception:
            img = None
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = 255.0 * (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = img.astype(np.uint8)
    return img


def preprocess(img):
    """CLAHE sur canal L (LAB) pour normalisation d'eclairage."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR).astype(np.uint8)


def run_preprocessing(input_dir, output_dir, name=''):
    saved = 0; failed = 0
    for split in ['train', 'test']:
        sp = os.path.join(input_dir, split)
        if not os.path.isdir(sp):
            continue
        for subj in tqdm(os.listdir(sp), desc=f'[{name}] Preprocess {split}'):
            pi = os.path.join(sp, subj); po = os.path.join(output_dir, split, subj)
            if not os.path.isdir(pi):
                continue
            os.makedirs(po, exist_ok=True)
            for f in os.listdir(pi):
                if os.path.splitext(f)[1].lower() not in VALID_EXT:
                    continue
                img = load_img(os.path.join(pi, f))
                if img is None:
                    failed += 1; continue
                cv2.imwrite(os.path.join(po, os.path.splitext(f)[0] + '.png'), preprocess(img))
                saved += 1
    print(f'  [{name}] saved:{saved} failed:{failed}')


def run_mtcnn(input_dir, output_dir, name='', device='cpu'):
    """MTCNN : detection + alignement facial. Fallback documente : image entiere."""
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, min_face_size=20,
                  thresholds=[0.5, 0.6, 0.6], device=device)
    det = 0; fall = 0
    for split in ['train', 'test']:
        si = os.path.join(input_dir, split); so = os.path.join(output_dir, split)
        if not os.path.isdir(si):
            continue
        for p in tqdm(os.listdir(si), desc=f'[{name}] MTCNN {split}'):
            pi = os.path.join(si, p); po = os.path.join(so, p)
            if not os.path.isdir(pi):
                continue
            os.makedirs(po, exist_ok=True)
            for fn in os.listdir(pi):
                if os.path.splitext(fn)[1].lower() not in VALID_EXT:
                    continue
                bgr = cv2.imread(os.path.join(pi, fn))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                boxes, probs, lmarks = mtcnn.detect(rgb, landmarks=True)
                if boxes is not None and lmarks is not None:
                    det += 1
                    le = np.array(lmarks[0][0]); re = np.array(lmarks[0][1])
                    ang = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
                    ctr = (int((le[0] + re[0]) / 2), int((le[1] + re[1]) / 2))
                    M = cv2.getRotationMatrix2D(ctr, ang, 1.0)
                    aligned = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]),
                                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                    b2, _ = mtcnn.detect(aligned)
                    if b2 is not None:
                        x1, y1, x2, y2 = [max(0, int(v)) for v in b2[0]]
                        x2 = min(aligned.shape[1], x2); y2 = min(aligned.shape[0], y2)
                        face = aligned[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else aligned
                    else:
                        face = aligned
                else:
                    fall += 1; face = rgb  # FALLBACK DOCUMENTE
                out = cv2.cvtColor(cv2.resize(face, (160, 160)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(po, fn), out)
    tot = det + fall
    if tot > 0:
        print(f'  [{name}] Detectes:{det} Fallback:{fall} Taux:{det/tot*100:.1f}%')
    if fall > 0:
        print(f'  [{name}] {fall} images sans detection MTCNN -> fallback image entiere '
              f'(attendu sur bases frontales controlees ORL/Yale).')


def extract_embeddings(input_dir, output_dir, facenet, device='cpu', name=''):
    """Extrait l'embedding FaceNet 512D normalise L2 pour chaque visage aligne."""
    import torch

    def get_embedding(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.tensor(rgb).permute(2, 0, 1).float()
        t = (t - 127.5) / 128.0
        with torch.no_grad():
            e = facenet(t.unsqueeze(0).to(device))
        emb = e.cpu().numpy()[0]
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-9)

    n = 0
    for split in ['train', 'test']:
        si = os.path.join(input_dir, split); so = os.path.join(output_dir, split)
        if not os.path.isdir(si):
            continue
        for p in tqdm(os.listdir(si), desc=f'[{name}] Embeddings {split}'):
            pi = os.path.join(si, p); po = os.path.join(so, p)
            if not os.path.isdir(pi):
                continue
            os.makedirs(po, exist_ok=True)
            for fn in os.listdir(pi):
                if not fn.lower().endswith(('.png', '.jpg', '.pgm')):
                    continue
                img = cv2.imread(os.path.join(pi, fn))
                if img is None:
                    continue
                np.save(os.path.join(po, os.path.splitext(fn)[0] + '.npy'), get_embedding(img))
                n += 1
    print(f'  [{name}] {n} embeddings sauvegardes (normalises L2, 512D).')


def main():
    if not os.path.isdir(ORL_RAW_DIR) or not os.path.isdir(YALE_RAW_DIR):
        print('[ERREUR] Dossiers de donnees brutes introuvables :')
        print(f'  ORL  attendu dans : {ORL_RAW_DIR}')
        print(f'  Yale attendu dans : {YALE_RAW_DIR}')
        print('Placez vos donnees ORL/Yale brutes a ces emplacements (ou modifiez')
        print('ORL_RAW_DIR / YALE_RAW_DIR en haut de ce fichier) puis relancez.')
        return

    try:
        import torch
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        print('[ERREUR] torch / facenet-pytorch non installes.')
        print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu')
        print('  pip install facenet-pytorch')
        return

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')

    print('=' * 60)
    print('  ETAPE 1 — Conversion extensions Yale -> .png')
    print('=' * 60)
    convert_yale_extensions_subfolders(YALE_RAW_DIR, YALE_CONV_DIR)

    print('\n' + '=' * 60)
    print('  ETAPE 2 — Split 50/50 stratifie (SEED=42)')
    print('=' * 60)
    split_dataset_5050(ORL_RAW_DIR, ORL_SPLIT_DIR, name='ORL')
    split_dataset_5050(YALE_CONV_DIR, YALE_SPLIT_DIR, name='Yale')

    print('\n' + '=' * 60)
    print('  ETAPE 3 — Pretraitement CLAHE')
    print('=' * 60)
    run_preprocessing(ORL_SPLIT_DIR, ORL_PRE_DIR, 'ORL')
    run_preprocessing(YALE_SPLIT_DIR, YALE_PRE_DIR, 'Yale')

    print('\n' + '=' * 60)
    print('  ETAPE 4 — MTCNN (detection + alignement)')
    print('=' * 60)
    run_mtcnn(ORL_PRE_DIR, ORL_MTCNN_DIR, 'ORL', device=device)
    run_mtcnn(YALE_PRE_DIR, YALE_MTCNN_DIR, 'Yale', device=device)

    print('\n' + '=' * 60)
    print('  ETAPE 5 — FaceNet 512D (InceptionResnetV1, VGGFace2)')
    print('=' * 60)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    extract_embeddings(ORL_MTCNN_DIR, ORL_EMB_DIR, facenet, device=device, name='ORL')
    extract_embeddings(YALE_MTCNN_DIR, YALE_EMB_DIR, facenet, device=device, name='Yale')

    print('\n[OK] data/ORL_MTCNN, data/ORL_EMBEDDINGS, data/YALE_MTCNN, data/YALE_EMBEDDINGS prets.')
    print('     Prochaine etape : python train_from_notebook.py')


if __name__ == '__main__':
    main()
