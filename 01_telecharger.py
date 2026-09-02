"""
01_telecharger.py — ÉTAPE 1 : récupérer l'historique de prix.

À lancer dans le terminal :

    python 01_telecharger.py              # tous les titres du S&P 500
    python 01_telecharger.py --test       # 20 titres seulement (rapide)
    python 01_telecharger.py --refresh    # rafraîchit aussi la liste des titres

Durée attendue : 5 à 15 minutes pour les 500 titres, quelques secondes
en mode --test. Le mode --test sert à vérifier que tout fonctionne avant
de lancer le téléchargement complet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src import telechargement, univers


def main() -> int:
    mode_test = "--test" in sys.argv
    refresh = "--refresh" in sys.argv

    depart = time.time()

    print("=" * 74)
    print("ÉTAPE 1 — TÉLÉCHARGEMENT DES PRIX".center(74))
    print("=" * 74)
    print()

    tickers = univers.liste_tickers(forcer_telechargement=refresh)

    if mode_test:
        tickers = tickers[:20]
        print()
        print(f"MODE TEST : {len(tickers)} titres seulement")
        print(f"  {', '.join(tickers)}")

    print()
    donnees = telechargement.telecharger(tickers)

    print()
    telechargement.sauvegarder(donnees)

    duree = time.time() - depart
    print()
    print(f"Terminé en {duree / 60:.1f} minutes.")
    print()
    print("Étape suivante :  python 02_verifier.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
