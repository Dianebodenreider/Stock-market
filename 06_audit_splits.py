"""
06_audit_splits.py — Audit systématique du défaut découvert sur MNST.

CE QU'ON A TROUVÉ
Autour de sa division du 11/08/2026, le cours de MNST oscille entre 97 et
47 pendant trois semaines : Yahoo livre des lignes retraitées et non
retraitées EN ALTERNANCE. Ce n'est pas un split manqué, c'est une base
incohérente sur la fenêtre entourant une division récente.

POURQUOI C'EST GRAVE
Le détecteur ne l'a vu que parce que les écarts dépassent 40 %. Une
corruption plus discrète passerait inaperçue et fausserait le signal sans
lever aucune alerte. La question n'est donc pas "MNST est-il défectueux"
mais "combien de titres le sont".

DEUX AUDITS INDÉPENDANTS

  A. AUTOUR DES DIVISIONS DÉCLARÉES
     Yahoo pré-retraite ses cours : au jour d'une division, le ratio
     close/close_veille doit être proche de 1, comme un jour normal.
     S'il est proche de 1/r, la division n'a pas été répercutée.
     On teste TOUTES les divisions déclarées dans la fenêtre, sur les 503.

  B. OSCILLATIONS (indépendant des divisions)
     Un mouvement de plus de 35 % suivi, dans les 5 séances, d'un
     mouvement inverse de plus de 35 %. Sur une grande capitalisation
     c'est physiquement invraisemblable : c'est la signature d'une
     série mélangeant deux échelles de prix.

     Cet audit ne suppose aucune connaissance des divisions. Il attraperait
     le défaut MNST même si Yahoo ne déclarait rien.

Les résultats des deux audits sont croisés à la fin : un titre signalé par
les deux méthodes indépendantes est un défaut certain.
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
from src import telechargement

FICHIER_SPLITS = config.DOSSIER_BRUT / "splits.parquet"

SEUIL_OSCILLATION = 0.35
FENETRE_OSCILLATION = 5      # séances
TOLERANCE_RATIO = 0.10       # écart admis autour de 1/r


def charger_splits(tickers: list[str], forcer: bool = False) -> pd.DataFrame:
    """Récupère les divisions déclarées par Yahoo, avec mise en cache."""
    if FICHIER_SPLITS.exists() and not forcer:
        df = pd.read_parquet(FICHIER_SPLITS)
        print(f"Divisions chargées depuis le cache : {len(df)} lignes")
        return df

    print(f"Récupération des divisions pour {len(tickers)} titres (~4 min)...")
    lignes = []
    for i, ticker in enumerate(tickers, 1):
        try:
            serie = yf.Ticker(ticker).splits
            if serie is not None and len(serie):
                idx = pd.DatetimeIndex(serie.index)
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                for date, ratio in zip(idx, serie.values):
                    lignes.append({"ticker": ticker, "date": date.normalize(),
                                   "ratio": float(ratio)})
        except Exception:  # noqa: BLE001
            pass
        if i % 50 == 0:
            print(f"  ... {i}/{len(tickers)}")
        time.sleep(0.05)

    df = pd.DataFrame(lignes)
    df.to_parquet(FICHIER_SPLITS, index=False)
    print(f"Divisions enregistrées : {len(df)} lignes -> {FICHIER_SPLITS}")
    return df


def audit_a(donnees: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Vérifie la continuité du cours brut autour de chaque division."""
    debut, fin = donnees["date"].min(), donnees["date"].max()
    dans_fenetre = splits[(splits.date >= debut) & (splits.date <= fin)]
    print(f"\nDivisions déclarées dans la fenêtre : {len(dans_fenetre)}")

    prix = {t: b.sort_values("date").reset_index(drop=True)
            for t, b in donnees.groupby("ticker")}

    resultats = []
    for _, s in dans_fenetre.iterrows():
        bloc = prix.get(s["ticker"])
        if bloc is None or len(bloc) < 2:
            continue
        positions = bloc.index[bloc["date"] >= s["date"]]
        if len(positions) == 0 or positions[0] == 0:
            continue
        i = int(positions[0])
        close = bloc["close"].astype(float)
        ratio = close[i] / close[i - 1]
        r = s["ratio"]
        attendu_si_non_retraite = 1 / r if r > 1 else r
        non_retraite = abs(ratio - attendu_si_non_retraite) < TOLERANCE_RATIO * max(
            attendu_si_non_retraite, 1
        )
        resultats.append({
            "ticker": s["ticker"],
            "date_split": s["date"],
            "ratio_split": r,
            "ratio_prix": round(float(ratio), 4),
            "non_retraite": bool(non_retraite),
        })

    return pd.DataFrame(resultats)


def audit_b(donnees: pd.DataFrame) -> pd.DataFrame:
    """Détecte les oscillations : gros mouvement puis gros retour inverse."""
    resultats = []
    for ticker, bloc in donnees.groupby("ticker"):
        bloc = bloc.sort_values("date").reset_index(drop=True)
        rend = bloc["adj_close"].astype(float).pct_change().to_numpy()
        gros = np.where(np.abs(rend) > SEUIL_OSCILLATION)[0]
        nb = 0
        premiere = None
        for i in gros:
            for j in range(i + 1, min(i + 1 + FENETRE_OSCILLATION, len(rend))):
                if abs(rend[j]) > SEUIL_OSCILLATION and np.sign(rend[j]) != np.sign(rend[i]):
                    nb += 1
                    if premiere is None:
                        premiere = bloc.loc[i, "date"]
                    break
        if nb:
            resultats.append({"ticker": ticker, "nb_oscillations": nb,
                              "premiere_date": premiere})
    return pd.DataFrame(resultats)


def main() -> int:
    print("=" * 74)
    print("AUDIT SYSTÉMATIQUE — COMBIEN DE TITRES SONT TOUCHÉS ?".center(74))
    print("=" * 74)
    print()

    donnees = telechargement.charger()
    tickers = sorted(donnees["ticker"].unique())
    splits = charger_splits(tickers, forcer="--refresh" in sys.argv)

    a = audit_a(donnees, splits)
    defauts_a = a[a.non_retraite] if len(a) else pd.DataFrame()

    print()
    print("=" * 74)
    print("  AUDIT A — continuité du cours autour des divisions déclarées")
    print("=" * 74)
    print(f"  Divisions vérifiées        : {len(a)}")
    print(f"  Divisions NON répercutées  : {len(defauts_a)}")
    print()
    if len(defauts_a):
        print(f"  {'ticker':<9}{'date':<13}{'split':>8}{'ratio prix':>13}")
        print("  " + "-" * 44)
        for _, l in defauts_a.iterrows():
            print(f"  {l['ticker']:<9}{l['date_split']:%d/%m/%Y}  "
                  f"{l['ratio_split']:>6.0f}:1{l['ratio_prix']:>13.3f}")
    else:
        print("  Aucune. Yahoo pré-retraite correctement.")

    b = audit_b(donnees)
    print()
    print("=" * 74)
    print("  AUDIT B — oscillations (méthode indépendante)")
    print("=" * 74)
    print(f"  Seuil : mouvement > {SEUIL_OSCILLATION:.0%} suivi d'un retour inverse "
          f"> {SEUIL_OSCILLATION:.0%} sous {FENETRE_OSCILLATION} séances")
    print(f"  Titres touchés : {len(b)}")
    print()
    if len(b):
        print(f"  {'ticker':<9}{'oscillations':>14}{'première date':>18}")
        print("  " + "-" * 42)
        for _, l in b.sort_values("nb_oscillations", ascending=False).head(30).iterrows():
            print(f"  {l['ticker']:<9}{int(l['nb_oscillations']):>14}"
                  f"      {l['premiere_date']:%d/%m/%Y}")
        if len(b) > 30:
            print(f"  ... et {len(b) - 30} autres")

    set_a = set(defauts_a["ticker"]) if len(defauts_a) else set()
    set_b = set(b["ticker"]) if len(b) else set()

    print()
    print("=" * 74)
    print("  CROISEMENT DES DEUX MÉTHODES".center(74))
    print("=" * 74)
    print(f"  Signalés par les DEUX (défaut certain) : {sorted(set_a & set_b)}")
    print(f"  Par A seulement                        : {sorted(set_a - set_b)}")
    print(f"  Par B seulement                        : {sorted(set_b - set_a)}")
    print()
    total = len(set_a | set_b)
    print(f"  Titres à écarter pour incohérence : {total} sur {len(tickers)} "
          f"({total / len(tickers):.1%})")
    print()

    chemin = config.DOSSIER_RAPPORTS / "audit_splits.csv"
    pd.DataFrame({"ticker": sorted(set_a | set_b)}).to_csv(chemin, index=False)
    print(f"  Liste enregistrée : {chemin}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
