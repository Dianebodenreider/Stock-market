"""
univers.py — Récupère la liste des titres du S&P 500.

=============================================================================
AVERTISSEMENT IMPORTANT — BIAIS DU SURVIVANT
=============================================================================
Ce module récupère la composition ACTUELLE du S&P 500.

Ce n'est PAS la composition historique. Les sociétés sorties de l'indice
(faillites, rachats, effondrements) n'y figurent pas.

Conséquence concrète : un backtest lancé sur cette liste ne pourra jamais
acheter Lehman Brothers en 2007, Enron en 2000, ou First Republic en 2022.
On teste une stratégie sur un univers dont les désastres ont été retirés
à l'avance. La littérature chiffre cette surestimation entre 1 et 4 points
de performance annuelle.

Traduction : si ton backtest sort +12 % par an, la vraie performance
est probablement entre +8 % et +11 %. Retiens ce chiffre, il servira
au moment de décider si la stratégie mérite un euro.

Pour supprimer ce biais il faut un fournisseur qui conserve les titres
délistés (Norgate, CRSP, Refinitiv). Ce n'est pas gratuit. Pour apprendre
et prototyper, on accepte le biais — mais on l'écrit noir sur blanc.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

URL_WIKIPEDIA = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _normaliser_ticker(ticker: str) -> str:
    """
    Yahoo Finance utilise un tiret là où Wikipédia utilise un point.

    Exemple : la classe B de Berkshire Hathaway s'écrit BRK.B chez S&P,
    mais BRK-B chez Yahoo. Sans cette conversion, on perd silencieusement
    quelques titres — et « silencieusement » est le mot dangereux.
    """
    return ticker.strip().upper().replace(".", "-")


def telecharger_univers() -> pd.DataFrame:
    """
    Va chercher la table des constituants sur Wikipédia.

    Renvoie un DataFrame avec les colonnes :
        ticker, nom, secteur, sous_secteur, date_ajout_indice
    """
    tables = pd.read_html(URL_WIKIPEDIA)

    # La première table de la page est celle des constituants.
    # On vérifie quand même : Wikipédia change de temps en temps.
    df = None
    for table in tables:
        colonnes = {str(c).strip() for c in table.columns}
        if "Symbol" in colonnes and "Security" in colonnes:
            df = table
            break

    if df is None:
        raise RuntimeError(
            "Impossible de trouver la table des constituants sur Wikipédia. "
            "La structure de la page a probablement changé."
        )

    # On renomme en français et on ne garde que ce qui sert.
    correspondances = {
        "Symbol": "ticker",
        "Security": "nom",
        "GICS Sector": "secteur",
        "GICS Sub-Industry": "sous_secteur",
        "Date added": "date_ajout_indice",
    }
    colonnes_presentes = {
        ancien: nouveau
        for ancien, nouveau in correspondances.items()
        if ancien in df.columns
    }
    df = df[list(colonnes_presentes)].rename(columns=colonnes_presentes)

    df["ticker"] = df["ticker"].astype(str).map(_normaliser_ticker)
    df = df.drop_duplicates(subset="ticker").sort_values("ticker")
    df = df.reset_index(drop=True)

    return df


def charger_univers(forcer_telechargement: bool = False) -> pd.DataFrame:
    """
    Renvoie l'univers, depuis le cache local si possible.

    On met en cache pour deux raisons :
      1. Ne pas dépendre de la disponibilité de Wikipédia à chaque exécution.
      2. Surtout : garder une liste FIGÉE. Si la liste change entre le
         téléchargement des prix et le backtest, on compare des choses
         différentes sans s'en rendre compte.

    forcer_telechargement=True pour rafraîchir volontairement.
    """
    if config.FICHIER_UNIVERS.exists() and not forcer_telechargement:
        df = pd.read_csv(config.FICHIER_UNIVERS)
        print(f"Univers chargé depuis le cache : {len(df)} titres")
        print(f"  ({config.FICHIER_UNIVERS})")
        return df

    print("Téléchargement de la composition du S&P 500 depuis Wikipédia...")
    df = telecharger_univers()
    df.to_csv(config.FICHIER_UNIVERS, index=False)
    print(f"Univers téléchargé et mis en cache : {len(df)} titres")
    print(f"  ({config.FICHIER_UNIVERS})")
    return df


def liste_tickers(forcer_telechargement: bool = False) -> list[str]:
    """Raccourci : renvoie juste la liste des symboles."""
    return charger_univers(forcer_telechargement)["ticker"].tolist()


if __name__ == "__main__":
    univers = charger_univers(forcer_telechargement="--refresh" in sys.argv)
    print()
    print("Aperçu :")
    print(univers.head(10).to_string(index=False))
    print()
    if "secteur" in univers.columns:
        print("Répartition sectorielle :")
        print(univers["secteur"].value_counts().to_string())
