"""
04_diagnostic_fenetre.py — Deuxième test, hypothèse affinée.

CE QUI S'EST PASSÉ
Le test précédent a échoué : NVR, TYL et MNST versent des dividendes et se
trouvent pourtant dans le groupe A (close == adj_close). L'hypothèse
"sociétés sans dividende" est donc fausse telle qu'elle était formulée.

HYPOTHÈSE AFFINÉE
Les facteurs d'ajustement de Yahoo se propagent VERS LE PASSÉ. Un dividende
détaché en 2024 abaisse tous les cours antérieurs ; un dividende détaché en
1998 n'abaisse que les cours antérieurs à 1998.

Notre fenêtre commence le 1er janvier 2005. Donc :
  close == adj_close  <=>  aucun dividende détaché DEPUIS le 01/01/2005

PRÉDICTION
Les 85 titres du groupe A ont zéro dividende de date >= 2005-01-01.
On teste les 85, pas un échantillon : la question mérite d'être tranchée.

Si un seul titre du groupe A a versé un dividende après 2005, l'hypothèse
tombe une deuxième fois et il faut chercher ailleurs.
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

DEBUT = pd.Timestamp(config.DATE_DEBUT)
ECHANTILLON_B = 20


def ecart_par_ticker(donnees: pd.DataFrame) -> pd.Series:
    close = donnees["close"].astype(float).replace(0, np.nan)
    adj = donnees["adj_close"].astype(float)
    return ((close - adj).abs() / close).groupby(donnees["ticker"]).max()


def dividendes(ticker: str):
    """Renvoie (total, nb_depuis_2005, derniere_date) ou None si l'appel échoue."""
    try:
        serie = yf.Ticker(ticker).dividends
        if serie is None or len(serie) == 0:
            return 0, 0, None
        index = pd.DatetimeIndex(serie.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        recents = index[index >= DEBUT]
        return len(index), len(recents), index.max()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    print("=" * 74)
    print("DIAGNOSTIC 2 — LES DIVIDENDES SONT-ILS ANTÉRIEURS À LA FENÊTRE ?".center(74))
    print("=" * 74)
    print(f"\nFenêtre de données : à partir du {DEBUT:%d/%m/%Y}\n")

    ecarts = ecart_par_ticker(telechargement.charger())
    groupe_a = sorted(ecarts[ecarts < 1e-6].index)
    groupe_b = sorted(ecarts[ecarts >= 1e-6].index)

    print(f"Groupe A : {len(groupe_a)} titres — interrogation des 85, patiente ~90 s")
    print()

    lignes = []
    for i, ticker in enumerate(groupe_a, 1):
        res = dividendes(str(ticker))
        if res is None:
            lignes.append({"ticker": ticker, "total": None, "depuis_2005": None, "derniere": None})
        else:
            total, recents, derniere = res
            lignes.append({"ticker": ticker, "total": total, "depuis_2005": recents,
                           "derniere": derniere})
        if i % 20 == 0:
            print(f"  ... {i}/{len(groupe_a)}")

    a = pd.DataFrame(lignes)
    valides = a[a.depuis_2005.notna()]
    fautifs = valides[valides.depuis_2005 > 0]

    print()
    print("-" * 74)
    print("  Titres du groupe A ayant versé des dividendes AVANT 2005")
    print("-" * 74)
    anciens = valides[(valides.total > 0) & (valides.depuis_2005 == 0)]
    if len(anciens) == 0:
        print("  aucun")
    for _, l in anciens.iterrows():
        print(f"  {l['ticker']:<8} {int(l['total']):>4} au total, dernier le "
              f"{l['derniere']:%d/%m/%Y}")

    print()
    print("=" * 74)
    print("  VERDICT".center(74))
    print("=" * 74)
    print(f"  Groupe A interrogés avec succès    : {len(valides)} / {len(groupe_a)}")
    print(f"  Sans aucun dividende depuis 2005   : {int((valides.depuis_2005 == 0).sum())}")
    print(f"  AVEC dividende depuis 2005 (faille): {len(fautifs)}")
    print()

    if len(fautifs) == 0 and len(valides) > 0:
        print("  HYPOTHÈSE CONFIRMÉE.")
        print("  close == adj_close signifie exactement : aucun dividende détaché")
        print("  depuis le début de la fenêtre. Les données sont SAINES.")
        print("  Le contrôle 'ajustement absent' est un faux positif à corriger.")
        code = 0
    else:
        print("  HYPOTHÈSE NON CONFIRMÉE — titres à examiner :")
        for _, l in fautifs.head(15).iterrows():
            print(f"    {l['ticker']:<8} {int(l['depuis_2005'])} dividende(s) depuis 2005, "
                  f"dernier le {l['derniere']:%d/%m/%Y}")
        print()
        print("  Ne corrige rien. Il y a autre chose.")
        code = 1

    print()
    print("-" * 74)
    print(f"  Contre-épreuve — {ECHANTILLON_B} titres du groupe B")
    print("-" * 74)
    tirage = np.random.default_rng(7).choice(groupe_b, ECHANTILLON_B, replace=False)
    sans_recent = []
    for ticker in tirage:
        res = dividendes(str(ticker))
        if res is None:
            continue
        _, recents, derniere = res
        if recents == 0:
            sans_recent.append(str(ticker))
        print(f"  {ticker:<8} {recents:>4} dividende(s) depuis 2005")
    print()
    if sans_recent:
        print(f"  ANOMALIE : {sans_recent} n'ont pas de dividende depuis 2005")
        print("  alors que leur close diffère de leur adj_close. À creuser.")
        code = 1
    else:
        print("  Contre-épreuve OK : tous ont versé un dividende depuis 2005.")
    print()

    return code


if __name__ == "__main__":
    raise SystemExit(main())
