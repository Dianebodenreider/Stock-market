"""
03_diagnostic_ajustement.py — Vérifie une hypothèse, ne la suppose pas.

HYPOTHÈSE À TESTER
Le contrôle "ajustement absent" écarte 85 titres. L'hypothèse est qu'il
s'agit de sociétés SANS DIVIDENDE, pour lesquelles close et adj_close sont
légitimement identiques (Yahoo livre déjà un close retraité des splits ;
adj_close n'ajoute que l'effet des dividendes).

Si l'hypothèse est vraie, le contrôle est un faux positif et doit être
corrigé. Si elle est fausse, il y a un vrai problème de données.

MÉTHODE
On ne se fie pas à ce qu'on croit savoir des sociétés. On interroge Yahoo
sur leur historique de dividendes et on compare aux deux groupes.

  Groupe A : les titres où close == adj_close (les 85 signalés)
  Groupe B : les titres où close != adj_close (les autres)

Prédiction si l'hypothèse est vraie :
  - groupe A : 0 dividende versé
  - groupe B : au moins un dividende versé

Une prédiction qui peut échouer, c'est ce qui distingue une vérification
d'une justification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from src import telechargement

TAILLE_ECHANTILLON = 15


def ecart_par_ticker(donnees: pd.DataFrame) -> pd.Series:
    """Écart relatif maximum entre le cours brut et le cours ajusté."""
    close = donnees["close"].astype(float).replace(0, np.nan)
    adj = donnees["adj_close"].astype(float)
    ecart = (close - adj).abs() / close
    return ecart.groupby(donnees["ticker"]).max()


def compter_dividendes(ticker: str) -> int | None:
    """Nombre de dividendes versés selon Yahoo. None si l'appel échoue."""
    try:
        dividendes = yf.Ticker(ticker).dividends
        return int(len(dividendes))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    print("=" * 74)
    print("DIAGNOSTIC — LE CONTRÔLE 'AJUSTEMENT ABSENT' EST-IL VALIDE ?".center(74))
    print("=" * 74)
    print()

    donnees = telechargement.charger()
    ecarts = ecart_par_ticker(donnees)

    groupe_a = sorted(ecarts[ecarts < 1e-6].index)   # signalés
    groupe_b = sorted(ecarts[ecarts >= 1e-6].index)  # non signalés

    print(f"Groupe A — close == adj_close : {len(groupe_a)} titres (signalés)")
    print(f"Groupe B — close != adj_close : {len(groupe_b)} titres")
    print()
    print(f"Interrogation de Yahoo sur les dividendes "
          f"({TAILLE_ECHANTILLON} titres par groupe)...")
    print()

    generateur = np.random.default_rng(42)
    echantillon_a = list(generateur.choice(groupe_a, min(TAILLE_ECHANTILLON, len(groupe_a)), replace=False))
    echantillon_b = list(generateur.choice(groupe_b, min(TAILLE_ECHANTILLON, len(groupe_b)), replace=False))

    resultats: list[dict] = []
    for groupe, echantillon in (("A", echantillon_a), ("B", echantillon_b)):
        for ticker in echantillon:
            nb = compter_dividendes(str(ticker))
            resultats.append({"groupe": groupe, "ticker": str(ticker), "dividendes": nb})

    tableau = pd.DataFrame(resultats)

    print("-" * 74)
    print(f"  {'groupe':<8}{'ticker':<10}{'dividendes versés':<22}{'écart close/ajusté'}")
    print("-" * 74)
    for _, ligne in tableau.iterrows():
        nb = ligne["dividendes"]
        texte = "échec de l'appel" if nb is None else f"{nb}"
        ecart = ecarts.get(ligne["ticker"], float("nan"))
        print(f"  {ligne['groupe']:<8}{ligne['ticker']:<10}{texte:<22}{ecart:.4f}")
    print()

    a_valides = tableau[(tableau.groupe == "A") & tableau.dividendes.notna()]
    b_valides = tableau[(tableau.groupe == "B") & tableau.dividendes.notna()]

    a_sans_div = int((a_valides.dividendes == 0).sum())
    b_avec_div = int((b_valides.dividendes > 0).sum())

    print("=" * 74)
    print("  VERDICT".center(74))
    print("=" * 74)
    print(f"  Groupe A sans aucun dividende : {a_sans_div} / {len(a_valides)}")
    print(f"  Groupe B avec dividendes      : {b_avec_div} / {len(b_valides)}")
    print()

    hypothese_confirmee = (
        len(a_valides) > 0
        and len(b_valides) > 0
        and a_sans_div == len(a_valides)
        and b_avec_div == len(b_valides)
    )

    if hypothese_confirmee:
        print("  HYPOTHÈSE CONFIRMÉE.")
        print("  Les titres signalés sont des sociétés sans dividende.")
        print("  Le contrôle 'ajustement absent' est un FAUX POSITIF :")
        print("  il ne doit pas écarter ces titres.")
    else:
        print("  HYPOTHÈSE NON CONFIRMÉE.")
        print("  Des titres du groupe A versent des dividendes, ou des titres")
        print("  du groupe B n'en versent pas. Il y a un vrai problème de")
        print("  données. Ne corrige rien avant d'avoir compris quoi.")
        anomalies_a = a_valides[a_valides.dividendes > 0]
        anomalies_b = b_valides[b_valides.dividendes == 0]
        if len(anomalies_a):
            print(f"    groupe A avec dividendes : {list(anomalies_a.ticker)}")
        if len(anomalies_b):
            print(f"    groupe B sans dividende  : {list(anomalies_b.ticker)}")
    print()

    return 0 if hypothese_confirmee else 1


if __name__ == "__main__":
    raise SystemExit(main())
