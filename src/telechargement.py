"""
telechargement.py — Récupère l'historique de prix via yfinance et le stocke.

Deux principes qui gouvernent tout ce fichier :

1. ON GARDE LES DEUX PRIX.
   `close`     = le cours réellement coté ce jour-là.
   `adj_close` = le cours retraité des dividendes et des divisions d'actions.

   Tous les calculs de rendement se font sur `adj_close`. Sinon :
     - une action qui verse 5 % de dividende par an voit sa performance
       amputée d'autant sur toute la période ;
     - une action qui fait un split 4 pour 1 apparaît comme une chute
       de -75 % en une séance, et ton signal croit voir un krach.

   On garde quand même `close`, parce que l'écart entre les deux est
   la preuve que l'ajustement a bien eu lieu. C'est un contrôle, pas
   une donnée de travail.

2. ON STOCKE EN FORMAT LONG.
   Une ligne = un titre, un jour. Colonnes : date, ticker, open, high,
   low, close, adj_close, volume.

   C'est moins lisible qu'un tableau large mais infiniment plus solide :
   pas de colonnes vides quand un titre entre ou sort de l'univers, et
   les jointures avec d'autres données (fondamentaux, secteurs) sont
   triviales.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

COLONNES_FINALES = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def _aplatir(brut: pd.DataFrame, tickers_demandes: list[str]) -> pd.DataFrame:
    """
    Transforme la sortie de yfinance (tableau large à colonnes imbriquées)
    en format long.

    yfinance renvoie une structure différente selon qu'on demande un seul
    titre ou plusieurs. Cette fonction absorbe les deux cas — c'est
    exactement le genre de détail qui casse un script trois semaines
    plus tard sans prévenir.
    """
    if brut is None or brut.empty:
        return pd.DataFrame(columns=COLONNES_FINALES)

    if isinstance(brut.columns, pd.MultiIndex):
        # Cas normal : plusieurs titres. On empile le niveau des tickers.
        # group_by="ticker" donne (ticker, champ) ; on vérifie l'ordre
        # au lieu de le supposer — yfinance l'a déjà inversé par le passé.
        niveau_0 = set(brut.columns.get_level_values(0))
        niveau_tickers = 0 if (niveau_0 & set(tickers_demandes)) else 1

        try:
            long = brut.stack(level=niveau_tickers, future_stack=True)
        except TypeError:
            # pandas < 2.1 ne connaît pas future_stack
            long = brut.stack(level=niveau_tickers, dropna=False)

        long.index.names = ["date", "ticker"]
        long = long.reset_index()
    else:
        # Cas d'un seul titre : pas de niveau ticker, on l'ajoute.
        long = brut.reset_index()
        long["ticker"] = tickers_demandes[0]

    # Normalisation des noms de colonnes : "Adj Close" -> "adj_close"
    long.columns = [
        str(c).strip().lower().replace(" ", "_") for c in long.columns
    ]

    if "date" not in long.columns and "datetime" in long.columns:
        long = long.rename(columns={"datetime": "date"})

    # Selon la version de yfinance, "Adj Close" peut manquer si les données
    # sont déjà ajustées. On le signale plutôt que de le fabriquer en douce.
    if "adj_close" not in long.columns:
        raise RuntimeError(
            "La colonne 'Adj Close' est absente de la réponse yfinance. "
            "Vérifie que le téléchargement utilise bien auto_adjust=False. "
            "Sans prix ajusté, tous les rendements calculés seront faux."
        )

    for colonne in COLONNES_FINALES:
        if colonne not in long.columns:
            long[colonne] = pd.NA

    long = long[COLONNES_FINALES]

    # Yahoo renvoie parfois des dates avec fuseau horaire, parfois sans,
    # selon le marché et la version de la bibliothèque. On force le format
    # naïf : sinon deux téléchargements successifs produisent des dates
    # incomparables, et la déduplication ne fonctionne plus.
    dates = pd.to_datetime(long["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    long["date"] = dates.dt.normalize()

    # On jette les lignes entièrement vides : yfinance renvoie des NaN
    # pour les dates où un titre n'était pas encore coté.
    long = long.dropna(subset=["adj_close"])

    return long


def _telecharger_lot(
    tickers: list[str], date_debut: str, date_fin: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Télécharge un paquet de titres, avec réessais.

    Renvoie (données, tickers_en_echec).
    """
    derniere_erreur = None

    for tentative in range(1, config.TENTATIVES_MAX + 1):
        try:
            brut = yf.download(
                tickers=tickers,
                start=date_debut,
                end=date_fin,
                auto_adjust=False,   # <-- indispensable : garde 'Adj Close'
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )
            donnees = _aplatir(brut, tickers)
            obtenus = set(donnees["ticker"].unique())
            manquants = sorted(set(tickers) - obtenus)
            return donnees, manquants

        except Exception as erreur:  # noqa: BLE001
            derniere_erreur = erreur
            attente = config.PAUSE_ENTRE_LOTS * tentative * 2
            print(
                f"    tentative {tentative}/{config.TENTATIVES_MAX} échouée "
                f"({type(erreur).__name__}) — nouvelle tentative dans {attente:.0f}s"
            )
            time.sleep(attente)

    print(f"    LOT ABANDONNÉ après {config.TENTATIVES_MAX} tentatives : {derniere_erreur}")
    return pd.DataFrame(columns=COLONNES_FINALES), list(tickers)


def telecharger(
    tickers: list[str],
    date_debut: str | None = None,
    date_fin: str | None = None,
) -> pd.DataFrame:
    """Télécharge l'historique complet pour une liste de titres."""
    date_debut = date_debut or config.DATE_DEBUT
    date_fin = date_fin or config.DATE_FIN

    lots = [
        tickers[i : i + config.TAILLE_LOT]
        for i in range(0, len(tickers), config.TAILLE_LOT)
    ]

    morceaux: list[pd.DataFrame] = []
    tous_manquants: list[str] = []

    print(f"Téléchargement de {len(tickers)} titres en {len(lots)} lots")
    print(f"Période : {date_debut} → {date_fin or 'aujourd’hui'}")
    print()

    for numero, lot in enumerate(lots, start=1):
        print(f"  Lot {numero}/{len(lots)} ({len(lot)} titres)...", flush=True)
        donnees, manquants = _telecharger_lot(lot, date_debut, date_fin)

        if not donnees.empty:
            morceaux.append(donnees)
        if manquants:
            tous_manquants.extend(manquants)
            print(f"    non récupérés : {', '.join(manquants)}")

        if numero < len(lots):
            time.sleep(config.PAUSE_ENTRE_LOTS)

    if not morceaux:
        raise RuntimeError(
            "Aucune donnée récupérée. Vérifie ta connexion internet, "
            "ou que Yahoo Finance n'est pas bloqué depuis cet environnement."
        )

    resultat = pd.concat(morceaux, ignore_index=True)
    resultat = resultat.sort_values(["ticker", "date"]).reset_index(drop=True)

    print()
    print(f"Récupéré : {resultat['ticker'].nunique()} titres, "
          f"{len(resultat):,} lignes".replace(",", " "))
    if tous_manquants:
        print(f"Échecs : {len(tous_manquants)} titres — {', '.join(sorted(set(tous_manquants)))}")

    return resultat


def sauvegarder(donnees: pd.DataFrame) -> None:
    """Écrit le fichier Parquet, en fusionnant avec l'existant s'il y en a un."""
    if config.FICHIER_PRIX.exists():
        ancien = pd.read_parquet(config.FICHIER_PRIX)
        avant = len(ancien)
        donnees = pd.concat([ancien, donnees], ignore_index=True)
        # En cas de doublon (date, ticker), on garde la version la plus
        # récemment téléchargée : Yahoo corrige parfois ses données a posteriori.
        donnees = donnees.drop_duplicates(subset=["date", "ticker"], keep="last")
        donnees = donnees.sort_values(["ticker", "date"]).reset_index(drop=True)
        print(f"Fusion avec l'existant : {avant:,} → {len(donnees):,} lignes"
              .replace(",", " "))

    donnees.to_parquet(config.FICHIER_PRIX, index=False, compression="snappy")
    taille_mo = config.FICHIER_PRIX.stat().st_size / 1_048_576
    print(f"Enregistré : {config.FICHIER_PRIX} ({taille_mo:.1f} Mo)")


def charger() -> pd.DataFrame:
    """Relit le fichier de prix. À utiliser par tous les scripts en aval."""
    if not config.FICHIER_PRIX.exists():
        raise FileNotFoundError(
            f"{config.FICHIER_PRIX} n'existe pas. "
            "Lance d'abord :  python 01_telecharger.py"
        )
    donnees = pd.read_parquet(config.FICHIER_PRIX)
    donnees["date"] = pd.to_datetime(donnees["date"])
    return donnees


def derniere_date_par_ticker() -> pd.Series:
    """Dernière date disponible pour chaque titre — sert au mode incrémental."""
    donnees = charger()
    return donnees.groupby("ticker")["date"].max()
