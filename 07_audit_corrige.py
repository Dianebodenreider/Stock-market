"""
07_audit_corrige.py — Correction du faux positif de l'audit A.

LE BUG
audit_a comparait le ratio de prix à 1/r, où r est le ratio déclaré par
Yahoo. Mais Yahoo déclare aussi des "divisions" de ratio ~1,0087 (scissions,
dividendes en actions). Pour r proche de 1, le ratio attendu vaut ~1,0 —
et un jour de bourse NORMAL a un ratio de 1,0.

Résultat : toute séance ordinaire déclenchait l'alerte. 44 faux positifs
sur 46, dont Disney, IBM, Verizon, Merck.

C'est le même défaut de conception que le contrôle "ajustement absent"
corrigé la veille : une condition d'anomalie que les données saines
satisfont. Deux garde-fous en découlent, appliqués ci-dessous.

LES CORRECTIONS
  1. Ne retenir que les VRAIES divisions : ratio >= 1,5 ou <= 0,67.
     En deçà, ce n'est pas une division de cours.
  2. Exiger que le ratio de prix soit LOIN de 1 (écart > 20 %).
     Un jour normal ne peut plus déclencher l'alerte, quel que soit r.
  3. Tolérance resserrée à 5 % relatif autour de 1/r.

VÉRIFICATION INTÉGRÉE
Le script affiche d'abord la distribution des ratios déclarés, pour qu'on
voie de nos yeux combien sont des faux "1:1". Puis il examine en détail les
4 titres de l'audit B, pour distinguer un vrai défaut d'un vrai krach.
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

SPLIT_MINIMUM = 1.5      # en deçà, ce n'est pas une division de cours
TOLERANCE_RELATIVE = 0.05
ECART_MIN_A_UN = 0.20    # le ratio de prix doit être loin d'un jour normal

SUSPECTS_B = ["MNST", "AIG", "HIG", "PNC"]


def main() -> int:
    donnees = telechargement.charger()
    splits = pd.read_parquet(FICHIER_SPLITS)
    debut, fin = donnees["date"].min(), donnees["date"].max()
    fenetre = splits[(splits.date >= debut) & (splits.date <= fin)].copy()

    print("=" * 74)
    print("AUDIT A CORRIGÉ".center(74))
    print("=" * 74)

    print(f"\n  Ratios déclarés par Yahoo dans la fenêtre ({len(fenetre)} lignes)")
    print("  " + "-" * 50)
    faux = fenetre[(fenetre.ratio < SPLIT_MINIMUM) & (fenetre.ratio > 1 / SPLIT_MINIMUM)]
    vrais = fenetre[(fenetre.ratio >= SPLIT_MINIMUM) | (fenetre.ratio <= 1 / SPLIT_MINIMUM)]
    print(f"  ratio proche de 1 (PAS une division) : {len(faux):>4}")
    print(f"  vraies divisions                     : {len(vrais):>4}")
    print()
    print("  Exemples de ratios 'proches de 1' que l'ancien code traitait")
    print("  comme des divisions :")
    for _, l in faux.head(6).iterrows():
        print(f"    {l['ticker']:<8} {l['date']:%d/%m/%Y}  ratio = {l['ratio']:.6f}")

    prix = {t: b.sort_values("date").reset_index(drop=True)
            for t, b in donnees.groupby("ticker")}

    resultats = []
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
        r = float(s["ratio"])
        attendu = 1 / r
        non_retraite = (
            abs(ratio - attendu) < TOLERANCE_RELATIVE * attendu
            and abs(ratio - 1) > ECART_MIN_A_UN
        )
        resultats.append({
            "ticker": s["ticker"], "date": s["date"], "ratio_split": r,
            "ratio_prix": round(ratio, 4), "non_retraite": non_retraite,
        })

    a = pd.DataFrame(resultats)
    defauts = a[a.non_retraite] if len(a) else pd.DataFrame()

    print()
    print("  " + "-" * 60)
    print(f"  Vraies divisions vérifiées   : {len(a)}")
    print(f"  Divisions NON répercutées    : {len(defauts)}")
    print()
    if len(defauts):
        print(f"  {'ticker':<9}{'date':<13}{'split':>10}{'ratio prix':>13}")
        print("  " + "-" * 46)
        for _, l in defauts.iterrows():
            print(f"  {l['ticker']:<9}{l['date']:%d/%m/%Y}  {l['ratio_split']:>8.2f}:1"
                  f"{l['ratio_prix']:>13.3f}")
    else:
        print("  Aucune. Yahoo pré-retraite correctement les vraies divisions.")

    print()
    print("=" * 74)
    print("EXAMEN DÉTAILLÉ DES 4 TITRES DE L'AUDIT B".center(74))
    print("=" * 74)
    print("\n  Question : vrai défaut de données, ou vrai krach de 2008 ?")
    print("  Un défaut fait osciller le cours entre deux échelles fixes.")
    print("  Un krach produit des mouvements violents mais sans retour exact.\n")

    for ticker in SUSPECTS_B:
        bloc = prix.get(ticker)
        if bloc is None:
            continue
        adj = bloc["adj_close"].astype(float)
        rend = adj.pct_change().to_numpy()
        gros = np.where(np.abs(rend) > 0.35)[0]
        print(f"\n  {'=' * 68}")
        print(f"  {ticker} — {len(gros)} séance(s) au-delà de 35 %")
        print(f"  {'=' * 68}")
        print(f"  {'date':<13}{'clôture ajustée':>17}{'variation':>12}")
        print("  " + "-" * 42)
        for i in gros:
            print(f"  {bloc.loc[i, 'date']:%d/%m/%Y}{adj[i]:>17.2f}{rend[i]:>11.1%}")
        if len(gros):
            fenetre_idx = range(max(0, gros[0] - 2), min(len(bloc), gros[-1] + 3))
            valeurs = adj[list(fenetre_idx)]
            print(f"  amplitude sur la période : {valeurs.min():.2f} à {valeurs.max():.2f} "
                  f"(rapport {valeurs.max() / max(valeurs.min(), 0.01):.2f})")

    print()
    print("=" * 74)
    print("  LECTURE : un rapport max/min proche d'un ratio rond (2,00, 3,00)")
    print("  avec des allers-retours signale deux échelles de prix mélangées.")
    print("  Un rapport quelconque signale un vrai mouvement de marché.")
    print("=" * 74)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
