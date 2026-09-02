"""
qualite.py — Contrôle des données avant tout calcul.

Pourquoi ce fichier existe :

Les erreurs qui coûtent de l'argent ne font pas planter le programme.
Un prix manquant, un split non retraité, une date en double : le script
tourne, produit une courbe de performance, et cette courbe est fausse.
Personne ne voit rien.

Chaque contrôle ci-dessous correspond à une erreur classique, avec en
commentaire ce qu'elle produit si on ne la détecte pas.

Le module ne corrige rien automatiquement. Il signale, et c'est toi qui
décides. Une correction silencieuse est aussi dangereuse qu'une erreur
silencieuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Ratios de division d'actions les plus fréquents. Sert à distinguer
# un vrai split d'une vraie chute de cours.
RATIOS_SPLIT_COURANTS = [2, 3, 4, 5, 7, 10, 20]
TOLERANCE_SPLIT = 0.03  # 3 % autour du ratio théorique


def _controler_un_titre(bloc: pd.DataFrame) -> dict:
    """Applique tous les contrôles à l'historique d'un seul titre."""
    bloc = bloc.sort_values("date")
    ticker = bloc["ticker"].iloc[0]

    resultat: dict = {"ticker": ticker}

    # --- Couverture temporelle -------------------------------------------
    resultat["nb_jours"] = len(bloc)
    resultat["date_debut"] = bloc["date"].min()
    resultat["date_fin"] = bloc["date"].max()

    # Erreur évitée : un titre entré en bourse en 2021 ne peut pas avoir
    # de momentum 12 mois en 2022. Sans ce contrôle, le calcul renvoie
    # NaN, la ligne est écartée en silence, et l'univers réel du backtest
    # n'est pas celui qu'on croit.
    resultat["historique_insuffisant"] = len(bloc) < config.JOURS_MINIMUM

    # --- Doublons ---------------------------------------------------------
    # Erreur évitée : une date en double double le poids de cette séance
    # dans toutes les moyennes, et fausse tout rendement cumulé.
    resultat["dates_en_double"] = int(bloc["date"].duplicated().sum())

    # --- Valeurs manquantes ou absurdes -----------------------------------
    # Erreur évitée : un prix à zéro produit un rendement de -100 %,
    # puis une division par zéro le lendemain.
    prix = bloc[["open", "high", "low", "close", "adj_close"]].astype(float)
    resultat["valeurs_manquantes"] = int(prix.isna().sum().sum())
    resultat["prix_nuls_ou_negatifs"] = int((prix <= 0).sum().sum())

    # --- Volume nul --------------------------------------------------------
    # Erreur évitée : le backtest « achète » un jour où le titre n'a pas
    # été échangé. Aucun ordre n'aurait pu être exécuté à ce prix.
    volume = bloc["volume"].astype(float).fillna(0)
    part_volume_nul = float((volume == 0).mean())
    resultat["part_volume_nul"] = round(part_volume_nul, 4)
    resultat["volume_nul_excessif"] = part_volume_nul > config.PART_MAX_VOLUME_NUL

    # --- Cohérence OHLC ----------------------------------------------------
    # Erreur évitée : un plus-bas supérieur au plus-haut signale une
    # donnée corrompue. Rare, mais fatal pour tout calcul de stop.
    incoherences = (
        (bloc["low"] > bloc["high"])
        | (bloc["low"] > bloc["open"])
        | (bloc["low"] > bloc["close"])
        | (bloc["high"] < bloc["open"])
        | (bloc["high"] < bloc["close"])
    )
    resultat["lignes_ohlc_incoherentes"] = int(incoherences.sum())

    # --- Trous dans l'historique -------------------------------------------
    # Erreur évitée : une semaine manquante casse tout calcul glissant
    # (moyenne mobile, momentum) sans qu'aucune erreur soit levée.
    dates = pd.DatetimeIndex(bloc["date"])
    if len(dates) > 1:
        # On compte en jours ouvrés pour ne pas confondre un week-end
        # avec un vrai trou.
        ecarts = np.array(
            [
                np.busday_count(d1.date(), d2.date())
                for d1, d2 in zip(dates[:-1], dates[1:])
            ]
        )
        resultat["trou_max_jours_ouvres"] = int(ecarts.max()) if len(ecarts) else 0
        resultat["nb_trous"] = int((ecarts > config.TROU_MAX_JOURS).sum())
    else:
        resultat["trou_max_jours_ouvres"] = 0
        resultat["nb_trous"] = 0

    # --- Rendements aberrants ----------------------------------------------
    # LE contrôle central.
    #
    # On calcule le rendement sur le prix ajusté (celui qui sert aux
    # calculs) ET sur le prix brut (celui réellement coté).
    #
    # Cas 1 : le brut saute de -50 %, l'ajusté ne bouge pas
    #         -> division d'actions correctement retraitée. Tout va bien.
    # Cas 2 : l'ajusté saute de -50 %
    #         -> soit un vrai krach sur ce titre, soit un split NON
    #            retraité. C'est le cas dangereux : le signal va lire
    #            un effondrement qui n'a jamais eu lieu.
    rend_ajuste = bloc["adj_close"].astype(float).pct_change()
    rend_brut = bloc["close"].astype(float).pct_change()

    suspects = rend_ajuste.abs() > config.SEUIL_RENDEMENT_SUSPECT
    resultat["rendements_ajustes_suspects"] = int(suspects.sum())
    resultat["rendement_ajuste_min"] = round(float(rend_ajuste.min(skipna=True)), 4)
    resultat["rendement_ajuste_max"] = round(float(rend_ajuste.max(skipna=True)), 4)

    # Détection spécifique d'un split non retraité : le prix brut est
    # divisé par un ratio rond ET l'ajusté suit au lieu de rester stable.
    ratio = bloc["close"].astype(float) / bloc["close"].astype(float).shift(1)
    split_non_retraite = 0
    for r in RATIOS_SPLIT_COURANTS:
        proche_division = (ratio - 1 / r).abs() < TOLERANCE_SPLIT / r
        proche_regroupement = (ratio - r).abs() < TOLERANCE_SPLIT * r
        candidat = proche_division | proche_regroupement
        # Un split correctement retraité laisse le rendement ajusté calme.
        split_non_retraite += int((candidat & suspects).sum())
    resultat["splits_probablement_non_retraites"] = split_non_retraite

    # --- L'ajustement a-t-il eu lieu ? --------------------------------------
    # Erreur évitée : si adj_close est strictement identique à close sur
    # 20 ans, l'ajustement n'a pas été appliqué. Sur une valeur qui verse
    # 4 % de dividende par an, ça retire environ 55 % de performance
    # cumulée sur 20 ans. Silencieusement.
    ecart_relatif = (
        (bloc["close"].astype(float) - bloc["adj_close"].astype(float)).abs()
        / bloc["close"].astype(float).replace(0, np.nan)
    )
    resultat["ecart_moyen_close_vs_ajuste"] = round(
        float(ecart_relatif.mean(skipna=True)), 4
    )
    resultat["ajustement_absent"] = bool(ecart_relatif.max(skipna=True) < 1e-6)

    return resultat


def controler(donnees: pd.DataFrame) -> pd.DataFrame:
    """Applique les contrôles à tous les titres. Renvoie un rapport."""
    lignes = [
        _controler_un_titre(bloc)
        for _, bloc in donnees.groupby("ticker", sort=True)
    ]
    rapport = pd.DataFrame(lignes)

    # Verdict par titre : utilisable ou non.
    #
    # Choix assumé : les TROUS et les RENDEMENTS SUSPECTS ne suffisent pas
    # à écarter un titre. Un trou peut venir d'une suspension de cotation
    # légitime, et un rendement de -45 % peut être un vrai krach (Meta a
    # perdu 26 % en une séance en février 2022, c'était réel).
    # Ces deux points sont signalés dans le rapport pour inspection
    # manuelle, pas appliqués automatiquement. Écarter automatiquement
    # les fortes baisses reviendrait à recréer un biais du survivant
    # de ses propres mains.
    rapport["a_ecarter"] = (
        rapport["historique_insuffisant"]
        | (rapport["dates_en_double"] > 0)
        | (rapport["prix_nuls_ou_negatifs"] > 0)
        | rapport["volume_nul_excessif"]
        | (rapport["lignes_ohlc_incoherentes"] > 0)
        | (rapport["splits_probablement_non_retraites"] > 0)
        | rapport["ajustement_absent"]
    )

    # Motif lisible, pour ne pas avoir à relire 20 colonnes.
    def _motif(ligne: pd.Series) -> str:
        motifs = []
        if ligne["historique_insuffisant"]:
            motifs.append(f"historique court ({ligne['nb_jours']} j)")
        if ligne["dates_en_double"] > 0:
            motifs.append(f"{ligne['dates_en_double']} dates en double")
        if ligne["prix_nuls_ou_negatifs"] > 0:
            motifs.append(f"{ligne['prix_nuls_ou_negatifs']} prix ≤ 0")
        if ligne["volume_nul_excessif"]:
            motifs.append(f"volume nul {ligne['part_volume_nul']:.1%} des séances")
        if ligne["lignes_ohlc_incoherentes"] > 0:
            motifs.append(f"{ligne['lignes_ohlc_incoherentes']} lignes OHLC incohérentes")
        if ligne["splits_probablement_non_retraites"] > 0:
            motifs.append(
                f"{ligne['splits_probablement_non_retraites']} split(s) non retraité(s)"
            )
        if ligne["ajustement_absent"]:
            motifs.append("prix ajusté identique au brut")
        return " ; ".join(motifs)

    rapport["motif"] = rapport.apply(_motif, axis=1)

    return rapport.sort_values(["a_ecarter", "ticker"], ascending=[False, True])


def afficher_resume(rapport: pd.DataFrame, donnees: pd.DataFrame) -> None:
    """Résumé console, en français, lisible sans connaître le code."""
    total = len(rapport)
    ecartes = int(rapport["a_ecarter"].sum())

    largeur = 74
    print()
    print("=" * largeur)
    print("RAPPORT DE QUALITÉ DES DONNÉES".center(largeur))
    print("=" * largeur)
    print()
    print(f"  Titres analysés              : {total}")
    print(f"  Lignes de prix               : {len(donnees):,}".replace(",", " "))
    print(f"  Période couverte             : "
          f"{donnees['date'].min():%d/%m/%Y} → {donnees['date'].max():%d/%m/%Y}")
    print()
    print(f"  Titres utilisables           : {total - ecartes}")
    print(f"  Titres à écarter             : {ecartes}")
    print()

    print("-" * largeur)
    print("  Détail des anomalies")
    print("-" * largeur)
    controles = [
        ("historique insuffisant", rapport["historique_insuffisant"].sum()),
        ("dates en double", (rapport["dates_en_double"] > 0).sum()),
        ("prix nuls ou négatifs", (rapport["prix_nuls_ou_negatifs"] > 0).sum()),
        ("volume nul excessif", rapport["volume_nul_excessif"].sum()),
        ("lignes OHLC incohérentes", (rapport["lignes_ohlc_incoherentes"] > 0).sum()),
        ("splits non retraités", (rapport["splits_probablement_non_retraites"] > 0).sum()),
        ("ajustement absent", rapport["ajustement_absent"].sum()),
        ("trous > 5 jours ouvrés", (rapport["nb_trous"] > 0).sum()),
        ("rendements ajustés > 40 %", (rapport["rendements_ajustes_suspects"] > 0).sum()),
    ]
    for nom, nombre in controles:
        marque = "  " if nombre == 0 else "! "
        print(f"  {marque}{nom:<34} {int(nombre):>4} titres")
    print()

    if ecartes:
        print("-" * largeur)
        print("  Titres écartés (20 premiers)")
        print("-" * largeur)
        apercu = rapport[rapport["a_ecarter"]].head(20)
        for _, ligne in apercu.iterrows():
            print(f"  {ligne['ticker']:<8} {ligne['motif']}")
        if ecartes > 20:
            print(f"  ... et {ecartes - 20} autres (voir le fichier CSV)")
        print()

    print("=" * largeur)
    print()
    print("  Rappel : cet univers est la composition ACTUELLE du S&P 500.")
    print("  Les sociétés sorties de l'indice en sont absentes. Tout backtest")
    print("  lancé dessus surestime la performance d'environ 1 à 4 points par an.")
    print()


def sauvegarder_rapport(rapport: pd.DataFrame) -> Path:
    chemin = config.DOSSIER_RAPPORTS / "qualite_donnees.csv"
    rapport.to_csv(chemin, index=False)
    return chemin


def tickers_valides(rapport: pd.DataFrame) -> list[str]:
    """Liste des titres qui passent tous les contrôles."""
    return rapport.loc[~rapport["a_ecarter"], "ticker"].tolist()
