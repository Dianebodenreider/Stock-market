"""
11_illustration_biais.py — Ce que rapporte le fait de connaître l'avenir.

Ce script ne teste PAS le sélecteur. Il ne peut pas : la liste des 25 a été
établie en septembre 2026 à partir de ratios portant sur 2025-2026. La
soumettre à l'année 2025 revient à choisir ses actions en connaissant déjà
le résultat.

Ce qu'il mesure : l'écart entre ce portefeuille et l'univers complet sur
2025. Cet écart est l'ampleur du biais, pas une performance.

À retenir : tout backtest de stratégie fondamentale bâti sur des données
non point-in-time produit exactement ce chiffre-là, en le présentant comme
un résultat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config

SELECTION = ["NEM", "INCY", "SNDK", "CF", "CINF", "MU", "DECK", "NVDA", "RDDT",
             "EOG", "ALL", "EIX", "ED", "CPRT", "BKNG", "WDC", "APP", "VICI",
             "ATO", "PLTR", "GOOG", "FSLR", "GOOGL", "SNA", "VEEV"]

ANNEE = 2025


def rendements(donnees: pd.DataFrame, annee: int) -> pd.Series:
    """Rendement de l'année civile, par titre, sur cours ajustés."""
    d = donnees[donnees["date"].dt.year == annee]
    premier = d.sort_values("date").groupby("ticker")["adj_close"].first()
    dernier = d.sort_values("date").groupby("ticker")["adj_close"].last()
    seances = d.groupby("ticker")["date"].count()
    complet = seances[seances > 200].index  # présent toute l'année
    return (dernier / premier - 1).loc[complet]


def main() -> int:
    donnees = pd.read_parquet(config.DOSSIER_BRUT / "prix_propres.parquet")
    donnees["date"] = pd.to_datetime(donnees["date"])

    r = rendements(donnees, ANNEE)
    presents = [t for t in SELECTION if t in r.index]
    absents = [t for t in SELECTION if t not in r.index]

    print("=" * 70)
    print(f"ANNÉE {ANNEE} — SÉLECTION ÉTABLIE EN SEPTEMBRE 2026".center(70))
    print("=" * 70)
    print()

    if absents:
        print(f"  Non détenables toute l'année {ANNEE} : {', '.join(absents)}")
        print("  (cotation trop récente ou historique incomplet)")
        print()

    detail = r.loc[presents].sort_values(ascending=False)
    print(f"  {'titre':<9}{'rendement ' + str(ANNEE):>18}")
    print("  " + "-" * 28)
    for ticker, valeur in detail.items():
        print(f"  {ticker:<9}{valeur:>17.1%}")
    print("  " + "-" * 28)

    portefeuille = float(detail.mean())
    univers = float(r.mean())
    mediane = float(detail.median())

    print()
    print("=" * 70)
    print(f"  Portefeuille équipondéré ({len(detail)} titres) : {portefeuille:>8.1%}")
    print(f"  Médiane des 25                          : {mediane:>8.1%}")
    print(f"  Univers complet ({len(r)} titres)           : {univers:>8.1%}")
    print(f"  ÉCART                                   : {portefeuille - univers:>+8.1%}")
    print("=" * 70)
    print()
    print("  CET ÉCART N'EST PAS UNE PERFORMANCE.")
    print("  C'est la mesure de ce que rapporte le fait de sélectionner")
    print("  des sociétés sur des chiffres postérieurs à la période testée.")
    print()
    print("  Un backtest fondamental bâti sur des données Yahoo produit")
    print("  exactement ce biais, à chaque date de rééquilibrage.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
