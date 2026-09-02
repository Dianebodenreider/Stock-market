"""
10_selecteur.py — Sélecteur fondamental sur les 499 titres.

CE QUE FAIT CET OUTIL
  Il note chaque société sur quatre piliers — qualité, solidité,
  valorisation, croissance — puis produit un classement et une liste courte
  à étudier. Il ne prédit rien, il ordonne.

CE QU'IL NE FAIT PAS, ET POURQUOI C'EST DÉLIBÉRÉ
  Aucun backtest. Les données fondamentales de Yahoo ne sont pas
  "point-in-time" : le bénéfice 2015 qu'il renvoie est le chiffre RETRAITÉ
  d'aujourd'hui, pas celui publié à l'époque. Backtester dessus revient à
  utiliser des informations que personne n'avait — le résultat est toujours
  flatteur et toujours faux.
  En travaillant sur les chiffres ACTUELS pour décider AUJOURD'HUI, le
  problème disparaît. L'outil ne prétend rien démontrer.

TROIS PRÉCAUTIONS INTÉGRÉES

  1. CLASSEMENT SECTORIEL
     Chaque ratio est comparé aux sociétés du MÊME secteur. Un PER de 12
     est cher pour une banque et bon marché pour un éditeur de logiciels.
     Sans cette précaution, le sélecteur ne fait que classer les secteurs.

  2. RATIOS INAPPLICABLES ÉCARTÉS
     Dette/EBITDA et VE/EBITDA n'ont aucun sens pour une banque ou une
     foncière : leur bilan ne fonctionne pas ainsi. Ces métriques sont
     neutralisées pour la finance et l'immobilier plutôt que calculées
     de travers.

  3. BÉNÉFICES NÉGATIFS
     Un PER négatif signifie que la société perd de l'argent. Le classer
     comme "très bon marché" serait absurde. Ces valeurs sont écartées et
     la société est signalée séparément.

LE SCORE FINAL EST UN OUTIL DE TRI, PAS UN VERDICT
  Une note de 85 signifie "à regarder en premier", pas "à acheter". Le
  détail par pilier est affiché pour que tu voies d'où vient la note :
  un titre excellent en valorisation et médiocre partout ailleurs est
  souvent bon marché pour une raison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

import config

FICHIER_FONDAMENTAUX = config.DOSSIER_BRUT / "fondamentaux.parquet"
FICHIER_UNIVERS_FINAL = config.DOSSIER_BRUT / "univers_final.csv"

SECTEURS_BILAN_PARTICULIER = {"Financials", "Financial Services", "Real Estate"}

PILIERS: dict[str, list[tuple[str, int]]] = {
    "qualite": [
        ("returnOnEquity", +1),
        ("returnOnAssets", +1),
        ("operatingMargins", +1),
        ("grossMargins", +1),
    ],
    "solidite": [
        ("debtToEquity", -1),
        ("currentRatio", +1),
        ("dette_nette_ebitda", -1),
    ],
    "valorisation": [
        ("trailingPE", -1),
        ("priceToBook", -1),
        ("enterpriseToEbitda", -1),
        ("rendement_fcf", +1),
    ],
    "croissance": [
        ("revenueGrowth", +1),
        ("earningsGrowth", +1),
        ("croissance_ca_3ans", +1),
    ],
}

INAPPLICABLES = {"dette_nette_ebitda", "enterpriseToEbitda", "debtToEquity", "currentRatio"}

CHAMPS_INFO = [
    "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook",
    "enterpriseToEbitda", "returnOnEquity", "returnOnAssets", "operatingMargins",
    "grossMargins", "profitMargins", "debtToEquity", "currentRatio", "totalDebt",
    "totalCash", "ebitda", "freeCashflow", "revenueGrowth", "earningsGrowth",
]

COUVERTURE_MINIMALE = 0.60


def croissance_chiffre_affaires(objet) -> float:
    """Croissance annualisée du chiffre d'affaires, depuis le compte de résultat."""
    try:
        compte = objet.income_stmt
        if compte is None or compte.empty:
            return np.nan
        for libelle in ("Total Revenue", "TotalRevenue", "Operating Revenue"):
            if libelle in compte.index:
                serie = compte.loc[libelle].dropna().astype(float).sort_index()
                if len(serie) < 3:
                    return np.nan
                debut, fin = float(serie.iloc[0]), float(serie.iloc[-1])
                annees = len(serie) - 1
                if debut <= 0 or fin <= 0:
                    return np.nan
                return (fin / debut) ** (1 / annees) - 1
    except Exception:  # noqa: BLE001
        pass
    return np.nan


def telecharger(tickers: list[str], forcer: bool = False) -> pd.DataFrame:
    """Récupère les fondamentaux, avec mise en cache."""
    if FICHIER_FONDAMENTAUX.exists() and not forcer:
        df = pd.read_parquet(FICHIER_FONDAMENTAUX)
        print(f"Fondamentaux chargés depuis le cache : {len(df)} titres")
        print(f"  ({FICHIER_FONDAMENTAUX})")
        return df

    print(f"Récupération des fondamentaux pour {len(tickers)} titres.")
    print("Compte 10 à 20 minutes — deux appels par société.\n")

    lignes, echecs = [], []
    for i, ticker in enumerate(tickers, 1):
        ligne = {"ticker": ticker}
        try:
            objet = yf.Ticker(ticker)
            infos = objet.get_info() or {}
            for champ in CHAMPS_INFO:
                ligne[champ] = infos.get(champ, np.nan)
            ligne["croissance_ca_3ans"] = croissance_chiffre_affaires(objet)
        except Exception as erreur:  # noqa: BLE001
            echecs.append(f"{ticker} ({type(erreur).__name__})")
            for champ in CHAMPS_INFO:
                ligne.setdefault(champ, np.nan)
            ligne["croissance_ca_3ans"] = np.nan
        lignes.append(ligne)

        if i % 25 == 0:
            print(f"  ... {i}/{len(tickers)}")
        time.sleep(0.15)

    df = pd.DataFrame(lignes)
    df.to_parquet(FICHIER_FONDAMENTAUX, index=False)
    print(f"\nEnregistré : {FICHIER_FONDAMENTAUX}")
    if echecs:
        print(f"Échecs ({len(echecs)}) : {', '.join(echecs[:15])}")
    return df


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les ratios dérivés et neutralise les valeurs absurdes."""
    d = df.copy()
    for champ in CHAMPS_INFO + ["croissance_ca_3ans"]:
        if champ in d.columns and champ not in ("sector", "industry"):
            d[champ] = pd.to_numeric(d[champ], errors="coerce")

    d["dette_nette_ebitda"] = np.where(
        d["ebitda"] > 0,
        (d["totalDebt"].fillna(0) - d["totalCash"].fillna(0)) / d["ebitda"],
        np.nan,
    )

    d["rendement_fcf"] = np.where(
        (d["marketCap"] > 0) & d["freeCashflow"].notna(),
        d["freeCashflow"] / d["marketCap"],
        np.nan,
    )

    # PRÉCAUTION 3 — un multiple négatif ne veut pas dire "bon marché".
    d["benefice_negatif"] = (d["trailingPE"].notna() & (d["trailingPE"] <= 0)) | (
        d["ebitda"].notna() & (d["ebitda"] <= 0)
    )
    for champ in ("trailingPE", "priceToBook", "enterpriseToEbitda"):
        d[champ] = d[champ].where(d[champ] > 0)

    d["trailingPE"] = d["trailingPE"].where(d["trailingPE"] < 200)

    # PRÉCAUTION 2 — ratios inapplicables aux bilans financiers.
    financier = d["sector"].isin(SECTEURS_BILAN_PARTICULIER)
    for champ in INAPPLICABLES:
        d.loc[financier, champ] = np.nan

    return d


def rangs_sectoriels(d: pd.DataFrame, champ: str, sens: int) -> pd.Series:
    """
    PRÉCAUTION 1 — percentile à l'intérieur du secteur, de 0 à 100.

    Un secteur trop petit (< 5 sociétés notées) est classé globalement.
    """
    valeurs = d[champ] * sens
    rangs = pd.Series(np.nan, index=d.index)
    for _, groupe in d.groupby("sector"):
        v = valeurs.loc[groupe.index].dropna()
        if len(v) >= 5:
            rangs.loc[v.index] = v.rank(pct=True) * 100
    manquants = rangs.isna() & valeurs.notna()
    if manquants.any():
        rangs.loc[manquants] = valeurs[manquants].rank(pct=True) * 100
    return rangs


def scorer(d: pd.DataFrame) -> pd.DataFrame:
    resultat = d[["ticker", "sector", "marketCap", "benefice_negatif"]].copy()
    couverture = pd.DataFrame(index=d.index)

    for pilier, metriques in PILIERS.items():
        rangs = pd.DataFrame(index=d.index)
        for champ, sens in metriques:
            if champ in d.columns:
                rangs[champ] = rangs_sectoriels(d, champ, sens)
        resultat[pilier] = rangs.mean(axis=1, skipna=True)
        couverture[pilier] = rangs.notna().sum(axis=1) / len(metriques)

    resultat["couverture"] = couverture.mean(axis=1)
    resultat["score"] = resultat[list(PILIERS)].mean(axis=1, skipna=True)
    resultat.loc[resultat["couverture"] < COUVERTURE_MINIMALE, "score"] = np.nan
    return resultat.sort_values("score", ascending=False)


def main() -> int:
    print("=" * 78)
    print("SÉLECTEUR FONDAMENTAL".center(78))
    print("=" * 78)
    print()

    tickers = pd.read_csv(FICHIER_UNIVERS_FINAL)["ticker"].tolist()
    if "--test" in sys.argv:
        tickers = tickers[:30]
        print(f"MODE TEST : {len(tickers)} titres\n")

    brut = telecharger(tickers, forcer="--refresh" in sys.argv)
    d = preparer(brut)
    notes = scorer(d)

    print()
    print("-" * 78)
    print("  COUVERTURE DES DONNÉES (part de titres renseignés)")
    print("-" * 78)
    tous = [c for m in PILIERS.values() for c, _ in m]
    for champ in tous:
        if champ in d.columns:
            part = d[champ].notna().mean()
            marque = "  " if part > 0.80 else ("~ " if part > 0.50 else "! ")
            print(f"  {marque}{champ:<26}{part:>6.0%}")
    print()
    notees = notes["score"].notna().sum()
    print(f"  Titres notés : {notees} / {len(notes)} "
          f"(couverture minimale exigée : {COUVERTURE_MINIMALE:.0%})")
    print(f"  Bénéfice négatif signalé : {int(d['benefice_negatif'].sum())} titres")

    top = notes.dropna(subset=["score"]).head(25)
    print()
    print("=" * 78)
    print("  LES 25 PREMIERS".center(78))
    print("=" * 78)
    print(f"  {'#':<4}{'titre':<8}{'secteur':<24}{'score':>7}"
          f"{'qual':>7}{'solid':>7}{'valo':>7}{'crois':>7}")
    print("  " + "-" * 74)
    for rang, (_, l) in enumerate(top.iterrows(), 1):
        secteur = str(l["sector"])[:22] if pd.notna(l["sector"]) else "?"
        def n(x):
            return f"{x:>7.0f}" if pd.notna(x) else f"{'-':>7}"
        print(f"  {rang:<4}{l['ticker']:<8}{secteur:<24}{l['score']:>7.1f}"
              f"{n(l['qualite'])}{n(l['solidite'])}{n(l['valorisation'])}"
              f"{n(l['croissance'])}")

    print()
    print("-" * 78)
    print("  Répartition sectorielle des 25 premiers")
    print("-" * 78)
    for secteur, nombre in top["sector"].value_counts().items():
        print(f"    {str(secteur):<30}{nombre:>3}")

    chemin = config.DOSSIER_RAPPORTS / "selecteur_fondamental.csv"
    notes.to_csv(chemin, index=False)

    print()
    print("=" * 78)
    print(f"  Classement complet : {chemin}")
    print()
    print("  COMMENT LIRE CE TABLEAU")
    print("  Les notes sont des percentiles SECTORIELS : 90 en valorisation")
    print("  signifie moins cher que 90 % des sociétés du même secteur, pas")
    print("  moins cher que le marché.")
    print()
    print("  Un score élevé porté par un seul pilier mérite la méfiance.")
    print("  Une société très bien notée en valorisation et faible ailleurs")
    print("  est généralement bon marché pour une raison.")
    print()
    print("  Cet outil trie. Il ne décide pas, et il ne prédit rien.")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
