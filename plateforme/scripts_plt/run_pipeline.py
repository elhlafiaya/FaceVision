"""
Lance la chaine complete : donnees brutes -> data/ -> models/

  1. prepare_data.py        (Cellules 2-3 du notebook : split + CLAHE + MTCNN + FaceNet)
  2. train_from_notebook.py (Cellules 5-6 du notebook : Seuil Fixe + SVM RBF + Calibration Platt)

Usage :
  python run_all.py
"""
import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))

def run(script):
    print(f"\n{'='*70}\n  LANCEMENT : {script}\n{'='*70}")
    r = subprocess.run([sys.executable, os.path.join(BASE, script)])
    if r.returncode != 0:
        print(f"[ERREUR] {script} a echoue (code {r.returncode}) — arret.")
        sys.exit(r.returncode)

if __name__ == "__main__":
    run("prepare_data.py")
    run("train_from_notebook.py")
    print("\n[OK] Pipeline complet termine. Lancez : python app.py")
