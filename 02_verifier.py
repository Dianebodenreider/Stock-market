"""
02_verifier.py — ÉTAPE 2 : contrôler la qualité des données téléchargées.

À lancer après l'étape 1 :

    python 02_verifier.py

Produit :
  - un résumé lisible dans le terminal
  - un fichier data/rapports/qualite_donnees.csv (une ligne par titre)
  - un fichier data/raw/univers_valide.csv (les titres à utiliser)

Ne lance JAMAIS un backtest avant d'avoir lu la sortie de ce script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from src import qualite, telechargement


def main() -> int:
    print("=" * 74)
    print("ÉTAPE 2 — CONTRÔLE QUALITÉ".center(74))
    print("=" * 74)
    print()

    donnees = telechargement.charger()
    print(f"Chargé : {donnees['ticker'].nunique()} titres, "
          f"{len(donnees):,} lignes".replace(",", " "))
    print("Analyse en cours...")

    rapport = qualite.controler(donnees)
    qualite.afficher_resume(rapport, donnees)

    chemin_rapport = qualite.sauvegarder_rapport(rapport)
    print(f"  Rapport détaillé : {chemin_rapport}")

    valides = qualite.tickers_valides(rapport)
    chemin_valides = config.DOSSIER_BRUT / "univers_valide.csv"
    pd.DataFrame({"ticker": valides}).to_csv(chemin_valides, index=False)
    print(f"  Univers validé   : {chemin_valides}  ({len(valides)} titres)")
    print()

    if not valides:
        print("  AUCUN titre ne passe les contrôles. Ne va pas plus loin.")
        return 1

    print("  Étape suivante : le calcul du momentum 12-1, puis le backtest.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
