"""
qualite.py — Contrôle des données avant tout calcul.

Les erreurs qui coûtent de l'argent ne font pas planter le programme. Un
prix manquant, un split non retraité, une date en double : le script tourne,
produit une courbe de performance, et cette courbe est fausse.

Le module ne corrige rien : il signale, tu décides.

=============================================================================
CORRECTION DU 02/09/2026 — un contrôle qui était faux
=============================================================================
La version précédente écartait tout titre où close == adj_close, au motif
que "l'ajustement n'a pas eu lieu". Elle écartait ainsi 85 titres sur 503,
dont Tesla, Amazon, AMD et Berkshire.

Vérification faite (scripts 03 et 04) : Yahoo livre déjà un `close` retraité
des divisions d'actions ; `adj_close` n'ajoute que l'effet des dividendes.
Et les facteurs d'ajustement se propagent VERS LE PASSÉ — un dividende de
2024 abaisse tous les cours antérieurs, un dividende de 1999 n'abaisse que
les cours d'avant 1999.

Donc, sur une fenêtre commençant en 2005 :
    close == adj_close  <=>  aucun dividende détaché depuis 2005

Testé sur les 85 titres concernés : 85 sur 85 sans dividende depuis 2005.
Contre-épreuve sur 20 titres du groupe opposé : 20 sur 20 avec dividendes.
Les données étaient saines ; le contrôle était faux.

Le contrôle correct de l'ajustement est GLOBAL, pas par titre : si AUCUN
titre du jeu de données n'a d'écart, alors le téléchargement s'est fait sans
ajustement et tout est à refaire. Si seulement certains en ont, ce sont les
sociétés sans dividende récent, et tout va bien.

Leçon générale : un contrôle qui écarte 17 % de l'univers doit être suspecté
avant d'être appliqué. Le biais introduit par un mauvais filtre est plus
dangereux que l'anomalie qu'il cherche.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

RATIOS_SPLIT_COURANTS = [2, 3, 4, 5, 7, 10, 20]
TOLERANCE_SPLIT = 0.03

# Une ligne OHLC corrompue sur 5 000 n'est pas une raison d'écarter un titre.
PART_MAX_OHLC_INCOHERENT = 0.001  # 0,1 %

PART_MIN_TITRES_AJUSTES = 0.50


def _controler_un_titre(bloc: pd.DataFrame) -> dict:
    """Applique tous les contrôles à l'historique d'un seul titre."""
    bloc = bloc.sort_values("date")
    resultat: dict = {"ticker": bloc["ticker"].iloc[0]}

    volume = bloc["volume"].astype(float).fillna(0)

    # --- Début de cotation réel --------------------------------------------
    # Yahoo remplit parfois l'avant d'une introduction en bourse (ou d'une
    # fusion) avec des lignes à volume nul. Compter ces lignes fausse tout :
    # le titre paraît illiquide alors qu'il n'existait pas encore.
    #
    # Erreur évitée : écarter Amcor parce que 38 % de ses séances sont à
    # volume nul, alors que ces séances précèdent sa cotation de 2019.
    echanges = volume > 0
    if echanges.any():
        premier = int(np.argmax(echanges.to_numpy()))
        effectif = bloc.iloc[premier:]
    else:
        effectif = bloc.iloc[0:0]

    resultat["nb_jours"] = len(bloc)
    resultat["nb_jours_effectifs"] = len(effectif)
    resultat["date_debut"] = bloc["date"].min()
    resultat["date_premiere_cotation"] = (
        effectif["date"].min() if len(effectif) else pd.NaT
    )
    resultat["date_fin"] = bloc["date"].max()

    # Erreur évitée : le momentum 12 mois renvoie NaN, la ligne disparaît en
    # silence, et l'univers réel du backtest n'est pas celui qu'on croit.
    resultat["historique_insuffisant"] = len(effectif) < config.JOURS_MINIMUM

    # Erreur évitée : la séance compte double dans toutes les moyennes.
    resultat["dates_en_double"] = int(bloc["date"].duplicated().sum())

    # Erreur évitée : un prix à zéro donne -100 % de rendement, puis une
    # division par zéro le lendemain.
    colonnes_prix = ["open", "high", "low", "close", "adj_close"]
    prix = effectif[colonnes_prix].astype(float) if len(effectif) else bloc[colonnes_prix].astype(float)
    resultat["valeurs_manquantes"] = int(prix.isna().sum().sum())
    resultat["prix_nuls_ou_negatifs"] = int((prix <= 0).sum().sum())

    # Volume nul, mesuré sur l'historique EFFECTIF seulement.
    # Erreur évitée : le backtest "achète" un jour où le titre n'a pas été
    # échangé — aucun ordre n'aurait pu passer à ce prix.
    if len(effectif):
        part_nul = float((effectif["volume"].astype(float).fillna(0) == 0).mean())
    else:
        part_nul = 1.0
    resultat["part_volume_nul"] = round(part_nul, 4)
    resultat["part_volume_nul_brut"] = round(float((volume == 0).mean()), 4)
    resultat["volume_nul_excessif"] = part_nul > config.PART_MAX_VOLUME_NUL

    # Cohérence OHLC, mesurée en PART et non en nombre absolu : 2 lignes sur
    # 5 400 sont un incident isolé, pas un motif d'écarter 21 ans.
    base = effectif if len(effectif) else bloc
    incoherences = (
        (base["low"] > base["high"])
        | (base["low"] > base["open"])
        | (base["low"] > base["close"])
        | (base["high"] < base["open"])
        | (base["high"] < base["close"])
    )
    nb_incoherentes = int(incoherences.sum())
    resultat["lignes_ohlc_incoherentes"] = nb_incoherentes
    part_ohlc = nb_incoherentes / max(len(base), 1)
    resultat["part_ohlc_incoherentes"] = round(part_ohlc, 5)
    resultat["ohlc_massivement_incoherent"] = part_ohlc > PART_MAX_OHLC_INCOHERENT

    # Trous : signalés mais NON éliminatoires (suspension légitime possible).
    dates = pd.DatetimeIndex(base["date"])
    if len(dates) > 1:
        ecarts = np.array(
            [np.busday_count(d1.date(), d2.date()) for d1, d2 in zip(dates[:-1], dates[1:])]
        )
        resultat["trou_max_jours_ouvres"] = int(ecarts.max())
        resultat["nb_trous"] = int((ecarts > config.TROU_MAX_JOURS).sum())
    else:
        resultat["trou_max_jours_ouvres"] = 0
        resultat["nb_trous"] = 0

    # Rendements aberrants : signalés mais NON éliminatoires. -26 % en une
    # séance peut être réel (Meta, février 2022). Écarter automatiquement les
    # fortes baisses reviendrait à fabriquer un second biais du survivant.
    rend_ajuste = base["adj_close"].astype(float).pct_change()
    suspects = rend_ajuste.abs() > config.SEUIL_RENDEMENT_SUSPECT
    resultat["rendements_ajustes_suspects"] = int(suspects.sum())
    resultat["rendement_ajuste_min"] = round(float(rend_ajuste.min(skipna=True)), 4)
    resultat["rendement_ajuste_max"] = round(float(rend_ajuste.max(skipna=True)), 4)

    # Split non retraité : ÉLIMINATOIRE. Le prix brut est divisé par un ratio
    # rond ET l'ajusté suit au lieu de rester stable : le signal lirait un
    # krach qui n'a jamais eu lieu.
    ratio = base["close"].astype(float) / base["close"].astype(float).shift(1)
    non_retraites = 0
    for r in RATIOS_SPLIT_COURANTS:
        candidat = ((ratio - 1 / r).abs() < TOLERANCE_SPLIT / r) | (
            (ratio - r).abs() < TOLERANCE_SPLIT * r
        )
        non_retraites += int((candidat & suspects).sum())
    resultat["splits_probablement_non_retraites"] = non_retraites

    # Écart brut / ajusté : INFORMATIF UNIQUEMENT (voir l'en-tête du fichier).
    close = base["close"].astype(float).replace(0, np.nan)
    ecart = (close - base["adj_close"].astype(float)).abs() / close
    resultat["ecart_max_close_vs_ajuste"] = round(float(ecart.max(skipna=True)), 4)
    resultat["dividende_dans_la_fenetre"] = bool(ecart.max(skipna=True) > 1e-6)

    return resultat


def controler_ajustement_global(rapport: pd.DataFrame) -> tuple[bool, str]:
    """
    LE contrôle de l'ajustement — global, pas par titre.

    Si aucun titre n'a d'écart entre close et adj_close, le téléchargement
    s'est fait sans ajustement et tout est à refaire.
    """
    part = float(rapport["dividende_dans_la_fenetre"].mean())
    if part == 0.0:
        return False, (
            "AUCUN titre n'a d'écart entre close et adj_close.\n"
            "  L'ajustement des dividendes n'a PAS été appliqué.\n"
            "  Vérifie auto_adjust=False et la colonne 'Adj Close', puis\n"
            "  retélécharge."
        )
    if part < PART_MIN_TITRES_AJUSTES:
        return True, (
            f"Seuls {part:.0%} des titres ont un écart brut/ajusté.\n"
            "  C'est bas mais pas anormal en soi : à surveiller."
        )
    return True, f"{part:.0%} des titres ont un écart brut/ajusté — normal."


def controler(donnees: pd.DataFrame) -> pd.DataFrame:
    """Applique les contrôles à tous les titres. Renvoie un rapport."""
    rapport = pd.DataFrame(
        [_controler_un_titre(bloc) for _, bloc in donnees.groupby("ticker", sort=True)]
    )

    # Motifs ÉLIMINATOIRES uniquement. Ce qui n'est pas ici est signalé mais
    # n'écarte pas : trous, rendements extrêmes, écart brut/ajusté.
    rapport["a_ecarter"] = (
        rapport["historique_insuffisant"]
        | (rapport["dates_en_double"] > 0)
        | (rapport["prix_nuls_ou_negatifs"] > 0)
        | rapport["volume_nul_excessif"]
        | rapport["ohlc_massivement_incoherent"]
        | (rapport["splits_probablement_non_retraites"] > 0)
    )

    def _motif(ligne: pd.Series) -> str:
        motifs = []
        if ligne["historique_insuffisant"]:
            motifs.append(f"historique court ({ligne['nb_jours_effectifs']} j cotés)")
        if ligne["dates_en_double"] > 0:
            motifs.append(f"{ligne['dates_en_double']} dates en double")
        if ligne["prix_nuls_ou_negatifs"] > 0:
            motifs.append(f"{ligne['prix_nuls_ou_negatifs']} prix <= 0")
        if ligne["volume_nul_excessif"]:
            motifs.append(f"volume nul {ligne['part_volume_nul']:.1%} des séances cotées")
        if ligne["ohlc_massivement_incoherent"]:
            motifs.append(
                f"OHLC incohérent sur {ligne['part_ohlc_incoherentes']:.2%} des lignes"
            )
        if ligne["splits_probablement_non_retraites"] > 0:
            motifs.append(f"{ligne['splits_probablement_non_retraites']} split(s) non retraité(s)")
        return " ; ".join(motifs)

    rapport["motif"] = rapport.apply(_motif, axis=1)
    return rapport.sort_values(["a_ecarter", "ticker"], ascending=[False, True])


def afficher_resume(rapport: pd.DataFrame, donnees: pd.DataFrame) -> None:
    """Résumé console, lisible sans connaître le code."""
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
          f"{donnees['date'].min():%d/%m/%Y} -> {donnees['date'].max():%d/%m/%Y}")
    print()
    print(f"  Titres utilisables           : {total - ecartes}")
    print(f"  Titres à écarter             : {ecartes}")
    print()

    print("-" * largeur)
    print("  Contrôle global de l'ajustement des dividendes")
    print("-" * largeur)
    ok, message = controler_ajustement_global(rapport)
    print(f"  {'OK  ' if ok else 'ÉCHEC'} {message}")
    print()

    print("-" * largeur)
    print("  Motifs éliminatoires")
    print("-" * largeur)
    eliminatoires = [
        ("historique insuffisant", rapport["historique_insuffisant"].sum()),
        ("dates en double", (rapport["dates_en_double"] > 0).sum()),
        ("prix nuls ou négatifs", (rapport["prix_nuls_ou_negatifs"] > 0).sum()),
        ("volume nul excessif", rapport["volume_nul_excessif"].sum()),
        ("OHLC massivement incohérent", rapport["ohlc_massivement_incoherent"].sum()),
        ("splits non retraités", (rapport["splits_probablement_non_retraites"] > 0).sum()),
    ]
    for nom, nombre in eliminatoires:
        print(f"  {'  ' if nombre == 0 else '! '}{nom:<34} {int(nombre):>4} titres")
    print()

    print("-" * largeur)
    print("  Signalements NON éliminatoires (à inspecter, pas à filtrer)")
    print("-" * largeur)
    informatifs = [
        ("trous > 5 jours ouvrés", (rapport["nb_trous"] > 0).sum()),
        ("rendements ajustés > 40 %", (rapport["rendements_ajustes_suspects"] > 0).sum()),
        ("lignes OHLC isolées à exclure", (
            (rapport["lignes_ohlc_incoherentes"] > 0)
            & ~rapport["ohlc_massivement_incoherent"]
        ).sum()),
        ("sans dividende depuis le début", (~rapport["dividende_dans_la_fenetre"]).sum()),
    ]
    for nom, nombre in informatifs:
        print(f"    {nom:<34} {int(nombre):>4} titres")
    print()

    if ecartes:
        print("-" * largeur)
        print("  Titres écartés")
        print("-" * largeur)
        for _, ligne in rapport[rapport["a_ecarter"]].head(30).iterrows():
            print(f"  {ligne['ticker']:<8} {ligne['motif']}")
        if ecartes > 30:
            print(f"  ... et {ecartes - 30} autres (voir le fichier CSV)")
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
    return rapport.loc[~rapport["a_ecarter"], "ticker"].tolist()
