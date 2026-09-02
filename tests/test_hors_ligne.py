"""
test_hors_ligne.py — Vérifie la logique SANS accès internet.

Le test est coupé en deux : "doivent passer" et "doivent être écartés".

La première moitié compte autant que la seconde. Le bug corrigé le
02/09/2026 était un FAUX POSITIF : le contrôle voyait une anomalie là où il
n'y en avait pas, et retirait 85 titres sur 503. Un test qui ne vérifiait
que les vrais positifs ne l'aurait jamais attrapé.

À lancer :   python tests/test_hors_ligne.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

import config
from src import qualite

CALENDRIER = pd.bdate_range("2020-01-01", "2024-12-31")


def _titre_sain(ticker: str, dates=None, avec_dividende: bool = True) -> pd.DataFrame:
    dates = CALENDRIER if dates is None else dates
    gen = np.random.default_rng(abs(hash(ticker)) % (2**32))
    adj = 100 * np.exp(np.cumsum(gen.normal(0.0004, 0.015, len(dates))))
    facteur = np.linspace(1.0, 1.18, len(dates)) if avec_dividende else np.ones(len(dates))
    close = adj * facteur
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": close * 0.998,
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "adj_close": adj,
            "volume": gen.integers(500_000, 5_000_000, len(dates)),
        }
    )


def construire_jeu_de_test() -> pd.DataFrame:
    blocs = []
    milieu = len(CALENDRIER) // 2

    # --- doivent PASSER ---
    for t in ("SAIN1", "SAIN2", "SAIN3"):
        blocs.append(_titre_sain(t))

    # Split 2:1 correctement retraité : le brut est divisé, l'ajusté non.
    bloc = _titre_sain("SPLITOK")
    bloc.loc[milieu:, ["open", "high", "low", "close"]] /= 2
    blocs.append(bloc)

    # LE FAUX POSITIF CORRIGÉ — société sans dividende sur la période.
    blocs.append(_titre_sain("SANSDIV", avec_dividende=False))

    # Une seule ligne OHLC corrompue sur ~1300 : incident isolé.
    bloc = _titre_sain("OHLC1")
    bloc.loc[150, "low"] = bloc.loc[150, "high"] * 1.5
    blocs.append(bloc)

    # Introduction en bourse tardive : volume nul avant cotation (cas Amcor).
    bloc = _titre_sain("IPOTARD")
    bloc.loc[: milieu - 1, "volume"] = 0
    blocs.append(bloc)

    # Trou d'un mois : suspension de cotation légitime.
    bloc = _titre_sain("TROU")
    blocs.append(bloc.drop(index=bloc.index[300:322]))

    # --- doivent être ÉCARTÉS ---
    bloc = _titre_sain("SPLITKO")
    bloc.loc[milieu:, ["open", "high", "low", "close", "adj_close"]] /= 2
    blocs.append(bloc)

    bloc = _titre_sain("DOUBLON")
    blocs.append(pd.concat([bloc, bloc.iloc[100:105]], ignore_index=True))

    bloc = _titre_sain("ZERO")
    bloc.loc[200, ["open", "high", "low", "close", "adj_close"]] = 0.0
    blocs.append(bloc)

    # Volume nul DISPERSÉ sur 33 % des séances : vraie illiquidité.
    bloc = _titre_sain("ILLIQUIDE")
    bloc.loc[bloc.index[::3], "volume"] = 0
    blocs.append(bloc)

    blocs.append(_titre_sain("JEUNE", CALENDRIER[-120:]))

    # OHLC corrompu sur 10 % des lignes : flux douteux.
    bloc = _titre_sain("OHLCKO")
    idx = bloc.index[::10]
    bloc.loc[idx, "low"] = bloc.loc[idx, "high"] * 1.5
    blocs.append(bloc)

    return pd.concat(blocs, ignore_index=True)


ATTENDUS = {
    "SAIN1": None,
    "SAIN2": None,
    "SAIN3": None,
    "SPLITOK": None,
    "SANSDIV": None,
    "OHLC1": None,
    "IPOTARD": None,
    "TROU": None,
    "SPLITKO": "splits_probablement_non_retraites",
    "DOUBLON": "dates_en_double",
    "ZERO": "prix_nuls_ou_negatifs",
    "ILLIQUIDE": "volume_nul_excessif",
    "JEUNE": "historique_insuffisant",
    "OHLCKO": "ohlc_massivement_incoherent",
}


def main() -> int:
    print("=" * 74)
    print("TEST HORS LIGNE DU CONTRÔLE QUALITÉ".center(74))
    print("=" * 74)
    print()

    donnees = construire_jeu_de_test()
    print(f"Jeu de test : {donnees['ticker'].nunique()} titres, {len(donnees)} lignes")
    print()

    rapport = qualite.controler(donnees).set_index("ticker")
    echecs = []

    print("  --- doivent PASSER ---")
    for ticker, attendu in ATTENDUS.items():
        if attendu is not None:
            continue
        ligne = rapport.loc[ticker]
        if bool(ligne["a_ecarter"]):
            echecs.append(f"{ticker} : écarté à tort — {ligne['motif']}")
            print(f"  ÉCHEC  {ticker:<10} écarté à tort : {ligne['motif']}")
        else:
            print(f"  ok     {ticker:<10}")

    print()
    print("  --- doivent être ÉCARTÉS ---")
    for ticker, attendu in ATTENDUS.items():
        if attendu is None:
            continue
        ligne = rapport.loc[ticker]
        valeur = ligne[attendu]
        detecte = bool(valeur) if isinstance(valeur, (bool, np.bool_)) else valeur > 0
        if not detecte or not bool(ligne["a_ecarter"]):
            echecs.append(f"{ticker} : '{attendu}' non détecté")
            print(f"  ÉCHEC  {ticker:<10} attendu -> {attendu}")
        else:
            print(f"  ok     {ticker:<10} {attendu}")

    print()
    print("  --- contrôle global de l'ajustement ---")
    ok, message = qualite.controler_ajustement_global(rapport.reset_index())
    print(f"  {'ok    ' if ok else 'ÉCHEC '} {message.splitlines()[0]}")
    if not ok:
        echecs.append("contrôle global de l'ajustement en échec")

    # Cas limite : un jeu SANS aucun ajustement doit déclencher l'alerte.
    sans_ajustement = donnees.copy()
    sans_ajustement["adj_close"] = sans_ajustement["close"]
    rapport_ko = qualite.controler(sans_ajustement)
    ok_ko, _ = qualite.controler_ajustement_global(rapport_ko)
    if ok_ko:
        echecs.append("l'alerte globale ne se déclenche pas sur un jeu non ajusté")
        print("  ÉCHEC  l'alerte ne se déclenche pas quand adj_close == close partout")
    else:
        print("  ok     l'alerte se déclenche bien si AUCUN titre n'est ajusté")

    print()
    print("-" * 74)
    if echecs:
        print(f"{len(echecs)} TEST(S) EN ÉCHEC :")
        for e in echecs:
            print(f"  - {e}")
        return 1

    print("Tous les tests passent : 8 titres sains conservés, 6 anomalies")
    print("détectées, alerte globale fonctionnelle, aucun faux positif.")
    print()
    qualite.afficher_resume(rapport.reset_index(), donnees)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
