"""
08_construire_base_propre.py — ÉTAPE 3 : produire la base de travail.

Les étapes 1 et 2 téléchargeaient et diagnostiquaient. Celle-ci DÉCIDE et
produit le fichier sur lequel tout le reste travaillera.

Toutes les règles appliquées ici ont été vérifiées sur données réelles par
les scripts 03 à 07. Aucune n'est appliquée sur intuition.

CE QUI EST EXCLU (et pourquoi, avec la preuve)
  1. Historique trop court (< 300 séances exploitables)
     Un momentum 12 mois est impossible. FDXF, HONA, Q.
  2. Division déclarée mais NON répercutée dans les cours
     Vérifié sur les 288 vraies divisions de la fenêtre : 1 seul cas, MNST
     (11/08/2026, ratio 2:1, ratio de prix 0,504). Le cours y oscille entre
     deux échelles pendant trois semaines.

CE QUI EST TRONQUÉ PLUTÔT QU'EXCLU
  3. Séances à volume nul en début d'historique
     Yahoo remplit l'avant d'une cotation américaine avec des lignes vides
     (changement de place, fusion). AMCR, FERG, SW. On garde la fenêtre
     continue la plus longue au lieu de jeter le titre.

CE QUI EST SIGNALÉ MAIS CONSERVÉ
  4. Mouvements extrêmes de 2008-2009 : AIG, HIG, PNC.
     Vérifié : trajectoires d'effondrement réelles, pas des artefacts.
     Les exclure reviendrait à retirer les perdants de la crise —
     un biais du survivant fabriqué à la main.

CE QUI RESTE, ET QU'AUCUN NETTOYAGE NE CORRIGERA
  L'univers est la composition ACTUELLE du S&P 500. Les sociétés sorties
  de l'indice en sont absentes. Compter 1 à 4 points de performance
  annuelle en trop sur tout backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from src import telechargement

FICHIER_SPLITS = config.DOSSIER_BRUT / "splits.parquet"
FICHIER_PROPRE = config.DOSSIER_BRUT / "prix_propres.parquet"
FICHIER_UNIVERS_FINAL = config.DOSSIER_BRUT / "univers_final.csv"

SPLIT_MINIMUM = 1.5
TOLERANCE_RELATIVE = 0.05
ECART_MIN_A_UN = 0.20


def fenetre_exploitable(volume: np.ndarray, seuil: float) -> int:
    """
    Indice de départ de la plus longue fenêtre finale dont la part de
    séances à volume nul reste sous le seuil.

    Un titre sain (aucun volume nul) démarre à 0 : rien n'est tronqué.
    Un titre à volume nul massif en tête démarre après cette zone.
    """
    n = len(volume)
    if n == 0:
        return 0
    nuls = (volume == 0)[::-1]
    part = np.cumsum(nuls) / np.arange(1, n + 1)
    admissibles = np.where(part <= seuil)[0]
    if len(admissibles) == 0:
        return n
    return n - 1 - int(admissibles.max())


def splits_non_repercutes(donnees: pd.DataFrame, splits: pd.DataFrame) -> set[str]:
    """Titres dont une vraie division n'apparaît pas dans les cours."""
    debut, fin = donnees["date"].min(), donnees["date"].max()
    vrais = splits[
        (splits.date >= debut) & (splits.date <= fin)
        & ((splits.ratio >= SPLIT_MINIMUM) | (splits.ratio <= 1 / SPLIT_MINIMUM))
    ]
    prix = {t: b.sort_values("date").reset_index(drop=True)
            for t, b in donnees.groupby("ticker")}

    fautifs = set()
    for _, s in vrais.iterrows():
        bloc = prix.get(s["ticker"])
        if bloc is None or len(bloc) < 2:
            continue
        pos = bloc.index[bloc["date"] >= s["date"]]
        if len(pos) == 0 or pos[0] == 0:
            continue
        i = int(pos[0])
        close = bloc["close"].astype(float)
        ratio = float(close[i] / close[i - 1])
        attendu = 1 / float(s["ratio"])
        if abs(ratio - attendu) < TOLERANCE_RELATIVE * attendu and abs(ratio - 1) > ECART_MIN_A_UN:
            fautifs.add(s["ticker"])
    return fautifs


def main() -> int:
    print("=" * 74)
    print("ÉTAPE 3 — CONSTRUCTION DE LA BASE DE TRAVAIL".center(74))
    print("=" * 74)
    print()

    donnees = telechargement.charger()
    splits = pd.read_parquet(FICHIER_SPLITS)
    print(f"Entrée : {donnees['ticker'].nunique()} titres, "
          f"{len(donnees):,} lignes".replace(",", " "))

    fautifs = splits_non_repercutes(donnees, splits)
    print(f"Divisions non répercutées : {sorted(fautifs) or 'aucune'}")
    print()

    journal = []
    conserves = []

    for ticker, bloc in donnees.groupby("ticker", sort=True):
        bloc = bloc.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
        avant = len(bloc)

        if ticker in fautifs:
            journal.append({"ticker": ticker, "statut": "exclu",
                            "motif": "division déclarée non répercutée",
                            "lignes_avant": avant, "lignes_apres": 0})
            continue

        volume = bloc["volume"].astype(float).fillna(0).to_numpy()
        depart = fenetre_exploitable(volume, config.PART_MAX_VOLUME_NUL)
        tronquees = depart
        bloc = bloc.iloc[depart:].reset_index(drop=True)

        incoherentes = (
            (bloc["low"] > bloc["high"]) | (bloc["low"] > bloc["open"])
            | (bloc["low"] > bloc["close"]) | (bloc["high"] < bloc["open"])
            | (bloc["high"] < bloc["close"])
        )
        invalides = incoherentes | (bloc[["open", "high", "low", "close", "adj_close"]]
                                    .astype(float) <= 0).any(axis=1)
        nb_lignes_retirees = int(invalides.sum())
        bloc = bloc[~invalides].reset_index(drop=True)

        if len(bloc) < config.JOURS_MINIMUM:
            journal.append({"ticker": ticker, "statut": "exclu",
                            "motif": f"historique court ({len(bloc)} séances)",
                            "lignes_avant": avant, "lignes_apres": 0})
            continue

        motif = []
        if tronquees:
            motif.append(f"{tronquees} lignes tronquées (volume nul)")
        if nb_lignes_retirees:
            motif.append(f"{nb_lignes_retirees} ligne(s) invalide(s) retirée(s)")
        journal.append({"ticker": ticker, "statut": "conservé",
                        "motif": " ; ".join(motif), "lignes_avant": avant,
                        "lignes_apres": len(bloc)})
        conserves.append(bloc)

    propre = pd.concat(conserves, ignore_index=True).sort_values(["ticker", "date"])
    propre = propre.reset_index(drop=True)
    journal = pd.DataFrame(journal)

    exclus = journal[journal.statut == "exclu"]
    modifies = journal[(journal.statut == "conservé") & (journal.motif != "")]

    print("-" * 74)
    print(f"  Titres exclus : {len(exclus)}")
    print("-" * 74)
    for _, l in exclus.iterrows():
        print(f"  {l['ticker']:<8} {l['motif']}")

    print()
    print("-" * 74)
    print(f"  Titres conservés après nettoyage partiel : {len(modifies)}")
    print("-" * 74)
    for _, l in modifies.iterrows():
        print(f"  {l['ticker']:<8} {int(l['lignes_avant']):>5} -> "
              f"{int(l['lignes_apres']):>5}   {l['motif']}")

    print()
    print("=" * 74)
    print("  BASE DE TRAVAIL".center(74))
    print("=" * 74)
    print(f"  Titres      : {propre['ticker'].nunique()} "
          f"(sur {donnees['ticker'].nunique()} téléchargés)")
    print(f"  Lignes      : {len(propre):,}".replace(",", " "))
    print(f"  Période     : {propre['date'].min():%d/%m/%Y} -> "
          f"{propre['date'].max():%d/%m/%Y}")
    print()

    propre.to_parquet(FICHIER_PROPRE, index=False, compression="snappy")
    pd.DataFrame({"ticker": sorted(propre["ticker"].unique())}).to_csv(
        FICHIER_UNIVERS_FINAL, index=False)
    journal.to_csv(config.DOSSIER_RAPPORTS / "journal_nettoyage.csv", index=False)

    print(f"  Prix propres  : {FICHIER_PROPRE}")
    print(f"  Univers final : {FICHIER_UNIVERS_FINAL}")
    print(f"  Journal       : {config.DOSSIER_RAPPORTS / 'journal_nettoyage.csv'}")
    print()
    print("  RAPPEL : univers = composition ACTUELLE du S&P 500.")
    print("  Retrancher 1 à 4 points de performance annuelle à tout backtest.")
    print()
    print("  Étape suivante : le calcul du momentum 12-1.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
