"""
test_hors_ligne.py — Vérifie la logique SANS accès internet.

Principe : on fabrique un jeu de données dont on connaît exactement les
défauts, on le passe dans le contrôle qualité, et on vérifie que chaque
défaut est bien détecté.

C'est le seul moyen d'avoir confiance dans un contrôle : lui montrer une
erreur qu'on a mise là exprès et vérifier qu'il la voit. Un contrôle qui
ne signale jamais rien n'est pas un contrôle qui rassure — c'est un
contrôle qui ne marche pas.

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


def _titre_sain(ticker: str, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Un titre normal : marche aléatoire, dividendes retraités, volume réel."""
    dates = CALENDRIER if dates is None else dates
    generateur = np.random.default_rng(abs(hash(ticker)) % (2**32))
    rendements = generateur.normal(0.0004, 0.015, len(dates))
    adj = 100 * np.exp(np.cumsum(rendements))
    # Le prix brut est supérieur à l'ajusté : effet cumulé des dividendes.
    facteur = np.linspace(1.0, 1.18, len(dates))
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
            "volume": generateur.integers(500_000, 5_000_000, len(dates)),
        }
    )


def construire_jeu_de_test() -> pd.DataFrame:
    blocs = []

    # 1. Trois titres parfaitement sains — le témoin.
    for ticker in ("SAIN1", "SAIN2", "SAIN3"):
        blocs.append(_titre_sain(ticker))

    # 2. Split 2 pour 1 CORRECTEMENT retraité.
    #    Le prix brut est divisé par deux, l'ajusté ne bouge pas.
    #    Le contrôle ne doit PAS écarter ce titre.
    bloc = _titre_sain("SPLITOK")
    milieu = len(bloc) // 2
    bloc.loc[milieu:, ["open", "high", "low", "close"]] /= 2
    blocs.append(bloc)

    # 3. Split 2 pour 1 NON retraité — l'erreur qui coûte cher.
    #    L'ajusté chute de 50 % en une séance. Le signal va lire un krach.
    bloc = _titre_sain("SPLITKO")
    bloc.loc[milieu:, ["open", "high", "low", "close", "adj_close"]] /= 2
    blocs.append(bloc)

    # 4. Aucun ajustement : adj_close identique à close.
    bloc = _titre_sain("NOADJ")
    bloc["adj_close"] = bloc["close"]
    blocs.append(bloc)

    # 5. Dates en double.
    bloc = _titre_sain("DOUBLON")
    blocs.append(pd.concat([bloc, bloc.iloc[100:105]], ignore_index=True))

    # 6. Prix à zéro.
    bloc = _titre_sain("ZERO")
    bloc.loc[200, ["open", "high", "low", "close", "adj_close"]] = 0.0
    blocs.append(bloc)

    # 7. Volume nul sur 30 % des séances — titre illiquide.
    bloc = _titre_sain("ILLIQUIDE")
    bloc.loc[bloc.index[::3], "volume"] = 0
    blocs.append(bloc)

    # 8. Introduction en bourse récente : 120 séances seulement.
    blocs.append(_titre_sain("JEUNE", CALENDRIER[-120:]))

    # 9. Trou d'un mois dans l'historique.
    bloc = _titre_sain("TROU")
    a_retirer = bloc.index[300:322]
    blocs.append(bloc.drop(index=a_retirer))

    # 10. OHLC incohérent : plus-bas au-dessus du plus-haut.
    bloc = _titre_sain("OHLCKO")
    bloc.loc[150, "low"] = bloc.loc[150, "high"] * 1.5
    blocs.append(bloc)

    return pd.concat(blocs, ignore_index=True)


ATTENDUS = {
    "SAIN1": None,
    "SAIN2": None,
    "SAIN3": None,
    "SPLITOK": None,                                  # ne doit PAS être écarté
    "SPLITKO": "splits_probablement_non_retraites",
    "NOADJ": "ajustement_absent",
    "DOUBLON": "dates_en_double",
    "ZERO": "prix_nuls_ou_negatifs",
    "ILLIQUIDE": "volume_nul_excessif",
    "JEUNE": "historique_insuffisant",
    "TROU": "nb_trous",
    "OHLCKO": "lignes_ohlc_incoherentes",
}


def main() -> int:
    print("=" * 74)
    print("TEST HORS LIGNE DU CONTRÔLE QUALITÉ".center(74))
    print("=" * 74)
    print()

    donnees = construire_jeu_de_test()
    print(f"Jeu de test : {donnees['ticker'].nunique()} titres, "
          f"{len(donnees)} lignes")
    print()

    rapport = qualite.controler(donnees).set_index("ticker")

    echecs = []
    for ticker, colonne_attendue in ATTENDUS.items():
        ligne = rapport.loc[ticker]

        if colonne_attendue is None:
            # On attend un titre propre.
            if bool(ligne["a_ecarter"]):
                echecs.append(
                    f"{ticker} : écarté à tort — {ligne['motif']}"
                )
                verdict = "ÉCHEC"
            else:
                verdict = "ok"
            print(f"  {verdict:<6} {ticker:<10} attendu sain")
        else:
            valeur = ligne[colonne_attendue]
            detecte = bool(valeur) if isinstance(valeur, (bool, np.bool_)) else valeur > 0
            if not detecte:
                echecs.append(
                    f"{ticker} : anomalie '{colonne_attendue}' NON détectée"
                )
                verdict = "ÉCHEC"
            else:
                verdict = "ok"
            print(f"  {verdict:<6} {ticker:<10} attendu → {colonne_attendue}")

    print()
    print("-" * 74)
    if echecs:
        print(f"{len(echecs)} TEST(S) EN ÉCHEC :")
        for echec in echecs:
            print(f"  - {echec}")
        return 1

    print("Tous les tests passent. Le contrôle qualité détecte bien")
    print("les 9 anomalies injectées et ne signale aucun faux positif.")
    print()

    # Aperçu du rendu réel.
    qualite.afficher_resume(rapport.reset_index(), donnees)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
