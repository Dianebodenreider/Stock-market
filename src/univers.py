"""
univers.py — Récupère la liste des titres du S&P 500.

BIAIS DU SURVIVANT — à lire une fois, à retenir toujours.
On récupère la composition ACTUELLE du S&P 500, pas la composition
historique. Les sociétés sorties de l'indice (faillites, rachats) en sont
absentes : un backtest sur cette liste ne pourra jamais acheter Lehman en
2007 ni First Republic en 2022. Surestimation attendue : 1 à 4 points de
performance annuelle. Si le backtest sort +12 %, la vérité est plutôt
entre +8 % et +11 %.

NOTE TECHNIQUE — pourquoi pas pd.read_html(url) directement.
pandas télécharge la page en s'annonçant "Python-urllib". Wikipédia bloque
cette signature et renvoie une erreur 403. On récupère donc la page nous-
mêmes avec requests, puis on passe le HTML à pandas.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

URL_WIKIPEDIA = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
URL_API_WIKIPEDIA = (
    "https://en.wikipedia.org/w/api.php"
    "?action=parse&page=List_of_S%26P_500_companies"
    "&prop=text&format=json&formatversion=2"
)

# Wikimédia demande une signature identifiable. C'est leur règle, on la suit.
ENTETES = {
    "User-Agent": "StockMarketResearch/1.0 (projet personnel de recherche)",
    "Accept": "text/html,application/json",
}
DELAI_SECONDES = 30


def _recuperer_html() -> str:
    """Télécharge le HTML avec une signature acceptée. Deux tentatives."""
    import requests

    erreurs: list[str] = []

    try:
        r = requests.get(URL_WIKIPEDIA, headers=ENTETES, timeout=DELAI_SECONDES)
        r.raise_for_status()
        return r.text
    except Exception as e:  # noqa: BLE001
        erreurs.append(f"page HTML : {type(e).__name__} — {e}")

    try:
        r = requests.get(URL_API_WIKIPEDIA, headers=ENTETES, timeout=DELAI_SECONDES)
        r.raise_for_status()
        return r.json()["parse"]["text"]
    except Exception as e:  # noqa: BLE001
        erreurs.append(f"API Wikipédia : {type(e).__name__} — {e}")

    raise RuntimeError(
        "Impossible de récupérer la liste du S&P 500.\n\n"
        + "\n".join(f"  - {e}" for e in erreurs)
        + "\n\nSOLUTION MANUELLE (2 minutes) :\n"
        f"  1. Ouvre dans ton navigateur : {URL_WIKIPEDIA}\n"
        "  2. Copie la colonne 'Symbol' du premier tableau.\n"
        f"  3. Crée le fichier {config.FICHIER_UNIVERS} ainsi :\n"
        "         ticker\n         AAPL\n         ABBV\n         ...\n"
        "  4. Relance la commande : le fichier sera utilisé tel quel.\n"
    )


def _normaliser_ticker(ticker: str) -> str:
    """
    Yahoo écrit BRK-B là où Wikipédia écrit BRK.B.
    Sans cette conversion on perd des titres SILENCIEUSEMENT.
    """
    return ticker.strip().upper().replace(".", "-")


def _extraire_table(html: str) -> pd.DataFrame:
    """Trouve le tableau des constituants parmi tous ceux de la page."""
    tables = pd.read_html(io.StringIO(html))
    for table in tables:
        colonnes = {str(c).strip() for c in table.columns}
        if "Symbol" in colonnes and "Security" in colonnes:
            return table
    raise RuntimeError(
        f"Aucun tableau exploitable ({len(tables)} tableaux analysés). "
        "La structure de la page Wikipédia a changé."
    )


def telecharger_univers() -> pd.DataFrame:
    """Renvoie ticker, nom, secteur, sous_secteur, date_ajout_indice."""
    df = _extraire_table(_recuperer_html())

    correspondances = {
        "Symbol": "ticker",
        "Security": "nom",
        "GICS Sector": "secteur",
        "GICS Sub-Industry": "sous_secteur",
        "Date added": "date_ajout_indice",
    }
    presentes = {a: n for a, n in correspondances.items() if a in df.columns}
    df = df[list(presentes)].rename(columns=presentes)

    df["ticker"] = df["ticker"].astype(str).map(_normaliser_ticker)
    df = df.drop_duplicates(subset="ticker").sort_values("ticker")
    df = df.reset_index(drop=True)

    # Garde-fou : le S&P 500 compte ~503 lignes (doubles classes d'actions).
    # Sans ce contrôle, le jour où la page change, tu lances un backtest sur
    # un univers tronqué sans le savoir.
    if not 450 <= len(df) <= 520:
        raise RuntimeError(
            f"Nombre de titres inattendu : {len(df)} (attendu 450 à 520). "
            "Le tableau récupéré n'est probablement pas le bon."
        )

    return df


def charger_univers(forcer_telechargement: bool = False) -> pd.DataFrame:
    """
    Renvoie l'univers, depuis le cache local si possible.

    Le cache sert surtout à FIGER la liste : si elle change entre le
    téléchargement des prix et le backtest, on compare sans le savoir
    deux univers différents.
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
    if "secteur" in univers.columns:
        print()
        print("Répartition sectorielle :")
        print(univers["secteur"].value_counts().to_string())
