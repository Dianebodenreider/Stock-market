"""
05_diagnostic_exclusions.py — Examine les 5 exclusions non évidentes.

Trois exclusions sont claires (FDXF, HONA, Q : cotations trop récentes).
Les cinq autres reposent sur des détecteurs que je n'ai pas vérifiés sur
données réelles. On regarde avant d'accepter.

QUESTION 1 — les "splits non retraités" de HIG et MNST
  Yahoo livre un close déjà retraité des divisions. Un split non retraité
  ne devrait donc pas exister. Deux possibilités :
    (a) Yahoo a un vrai défaut sur ces titres ;
    (b) mon détecteur confond une forte baisse avec un split.
  On affiche les cours autour de chaque date signalée, et on les compare
  aux dates de split que Yahoo déclare lui-même.

QUESTION 2 — le volume nul de AMCR, FERG et SW
  Après la première séance échangée, il reste 38 à 60 % de séances à volume
  nul. Où sont-elles ? Si elles sont groupées avant une date pivot, c'est un
  changement de cotation (fusion, transfert de place). Si elles sont
  dispersées, le titre est réellement inexploitable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

import config
from src import telechargement
from src.qualite import RATIOS_SPLIT_COURANTS, TOLERANCE_SPLIT

SPLITS_A_EXAMINER = ["HIG", "MNST"]
VOLUME_A_EXAMINER = ["AMCR", "FERG", "SW"]


def examiner_splits(donnees: pd.DataFrame, ticker: str) -> None:
    bloc = donnees[donnees.ticker == ticker].sort_values("date").reset_index(drop=True)
    close = bloc["close"].astype(float)
    adj = bloc["adj_close"].astype(float)
    ratio = close / close.shift(1)
    rend = adj.pct_change()
    suspects = rend.abs() > config.SEUIL_RENDEMENT_SUSPECT

    signale = pd.Series(False, index=bloc.index)
    for r in RATIOS_SPLIT_COURANTS:
        candidat = ((ratio - 1 / r).abs() < TOLERANCE_SPLIT / r) | (
            (ratio - r).abs() < TOLERANCE_SPLIT * r
        )
        signale |= candidat & suspects

    print(f"\n{'=' * 74}\n  {ticker} — {int(signale.sum())} date(s) signalée(s)\n{'=' * 74}")

    try:
        splits = yf.Ticker(ticker).splits
        idx = pd.DatetimeIndex(splits.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        splits_yahoo = {d.date(): float(v) for d, v in zip(idx, splits.values)}
    except Exception:  # noqa: BLE001
        splits_yahoo = {}

    print(f"  Splits déclarés par Yahoo : "
          f"{ {str(k): v for k, v in splits_yahoo.items()} or 'aucun'}")
    print()
    print(f"  {'date':<12}{'close J-1':>11}{'close J':>11}{'ratio':>8}"
          f"{'rend. ajusté':>14}{'  split déclaré ce jour ?'}")
    print("  " + "-" * 70)

    for i in bloc.index[signale]:
        date = bloc.loc[i, "date"]
        declare = splits_yahoo.get(date.date())
        proche = [
            f"{v}:1 le {d}"
            for d, v in splits_yahoo.items()
            if abs((pd.Timestamp(d) - date).days) <= 3
        ]
        marque = f"OUI ({declare}:1)" if declare else (
            f"proche: {proche[0]}" if proche else "NON"
        )
        print(f"  {date:%d/%m/%Y}{close[i-1]:>11.2f}{close[i]:>11.2f}"
              f"{ratio[i]:>8.3f}{rend[i]:>13.1%}   {marque}")

    print()
    print("  LECTURE : si 'split déclaré' = OUI, Yahoo connaît la division mais")
    print("  ne l'a pas répercutée -> vrai défaut, le titre doit être écarté.")
    print("  Si 'NON' partout, ce sont de vraies variations de cours et mon")
    print("  détecteur produit des faux positifs -> il faut le corriger.")


def examiner_volume(donnees: pd.DataFrame, ticker: str) -> None:
    bloc = donnees[donnees.ticker == ticker].sort_values("date").reset_index(drop=True)
    volume = bloc["volume"].astype(float).fillna(0)
    echanges = volume > 0
    premier = int(np.argmax(echanges.to_numpy()))
    effectif = bloc.iloc[premier:]
    vol_eff = effectif["volume"].astype(float).fillna(0)

    print(f"\n{'=' * 74}\n  {ticker}\n{'=' * 74}")
    print(f"  Première séance échangée : {bloc.loc[premier, 'date']:%d/%m/%Y}")
    print(f"  Séances après cette date : {len(effectif)}")
    print(f"  dont volume nul          : {int((vol_eff == 0).sum())} "
          f"({(vol_eff == 0).mean():.1%})")
    print()
    print(f"  {'année':<8}{'séances':>9}{'volume nul':>12}{'part':>8}")
    print("  " + "-" * 37)
    par_an = effectif.assign(nul=(vol_eff == 0).values, an=effectif["date"].dt.year)
    tableau = par_an.groupby("an").agg(seances=("nul", "size"), nuls=("nul", "sum"))
    for an, ligne in tableau.iterrows():
        part = ligne["nuls"] / ligne["seances"]
        marque = "  <<<" if part > 0.5 else ""
        print(f"  {an:<8}{ligne['seances']:>9}{int(ligne['nuls']):>12}{part:>7.0%}{marque}")

    derniere_nulle = effectif.loc[vol_eff.to_numpy() == 0, "date"]
    if len(derniere_nulle):
        pivot = derniere_nulle.max()
        apres = effectif[effectif["date"] > pivot]
        print()
        print(f"  Dernière séance à volume nul : {pivot:%d/%m/%Y}")
        print(f"  Séances propres après        : {len(apres)}")
        if len(apres) >= config.JOURS_MINIMUM:
            print(f"  -> RÉCUPÉRABLE en tronquant l'historique à partir du "
                  f"{pivot:%d/%m/%Y}")
        else:
            print(f"  -> NON récupérable : {len(apres)} séances propres, "
                  f"il en faut {config.JOURS_MINIMUM}")


def main() -> int:
    donnees = telechargement.charger()

    print("=" * 74)
    print("DIAGNOSTIC — LES 5 EXCLUSIONS NON ÉVIDENTES".center(74))
    print("=" * 74)
    print("\n\n### QUESTION 1 — splits non retraités\n")
    for t in SPLITS_A_EXAMINER:
        examiner_splits(donnees, t)

    print("\n\n### QUESTION 2 — volume nul\n")
    for t in VOLUME_A_EXAMINER:
        examiner_volume(donnees, t)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
