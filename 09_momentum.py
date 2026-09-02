"""
09_momentum.py — ÉTAPE 4 : le signal momentum 12-1 et son backtest.

LA RÈGLE, EN FRANÇAIS
  Le dernier jour de chaque mois, on classe les titres sur leur performance
  des 12 derniers mois EN EXCLUANT le mois écoulé. On achète le décile
  supérieur, à poids égaux, et on garde un mois. Puis on recommence.

POURQUOI EXCLURE LE DERNIER MOIS
  À très court terme les cours ont tendance à rebondir (retour à la moyenne).
  Acheter ce qui vient de monter sur un mois fait perdre de l'argent. La
  littérature (Jegadeesh & Titman) écarte donc le mois le plus récent.
  C'est le "-1" de "12-1".

LE CALENDRIER, PRÉCISÉMENT
  formation fin du mois t : signal = P(fin t-1) / P(fin t-12) - 1
  détention              : du début du mois t+1 à sa fin
  Le signal n'utilise QUE des prix antérieurs à la date de formation.

LE SEUIL DE BRUIT — mesuré, pas supposé
  Sur 20 marchés purement aléatoires (200 titres, 11 ans), l'écart annuel
  entre cette stratégie et son univers a une moyenne de -0,13 % et un
  écart-type de 1,6 %. Donc : un écart inférieur à ~3 % par an ne se
  distingue pas de la chance. C'est le chiffre auquel comparer le résultat.

LES TROIS PIÈGES, ET LEUR TRAITEMENT

  1. LOOK-AHEAD (utiliser le futur sans s'en rendre compte)
     Le script contient un test : il recalcule le signal à plusieurs dates
     en ne lui donnant QUE les données antérieures, et vérifie que le
     résultat est identique. Si un seul chiffre diffère, tout s'arrête.

  2. LES COÛTS
     Comptés à chaque rééquilibrage, proportionnels à la rotation réelle
     du portefeuille. Une stratégie brute est un chiffre sans signification.

  3. LE POINT DE COMPARAISON
     On compare à l'univers équipondéré, pas au S&P 500. Battre un indice
     qu'on ne détient pas ne prouve rien. Et l'univers lui-même porte le
     biais du survivant : il est déjà plus performant que la réalité.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config

FICHIER_PROPRE = config.DOSSIER_BRUT / "prix_propres.parquet"

MOIS_FORMATION = 12      # on regarde 12 mois en arrière
MOIS_EXCLUS = 1          # on saute le plus récent
PART_DECILE = 0.10       # on achète le décile supérieur
PRIX_MINIMUM = 5.0       # sous 5 $, spreads trop larges
VOLUME_DOLLAR_MIN = 5e6  # 5 M$ échangés par jour en moyenne sur 3 mois
COUT_UNE_JAMBE = 0.0005  # 5 points de base à l'achat comme à la vente


def panneau_mensuel(donnees: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Transforme les prix quotidiens en tableaux mensuels.

    Renvoie trois tableaux (lignes = fins de mois, colonnes = titres) :
      prix    : clôture ajustée du dernier jour du mois
      brut    : clôture NON ajustée (sert au filtre de prix minimum, car
                c'est le prix réellement coté qui détermine le spread)
      dollars : volume en dollars moyen du mois
    """
    d = donnees.copy()
    d["mois"] = d["date"].dt.to_period("M")
    d["dollars"] = d["close"].astype(float) * d["volume"].astype(float)

    dernier = d.sort_values("date").groupby(["mois", "ticker"]).last()
    moyen = d.groupby(["mois", "ticker"])["dollars"].mean()

    prix = dernier["adj_close"].unstack()
    brut = dernier["close"].unstack()
    dollars = moyen.unstack()

    prix.index = prix.index.to_timestamp("M")
    brut.index = brut.index.to_timestamp("M")
    dollars.index = dollars.index.to_timestamp("M")
    return {"prix": prix, "brut": brut, "dollars": dollars}


def signal_momentum(prix: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum 12-1 : P(t-1) / P(t-12) - 1, disponible à la fin du mois t.

    shift(1) et shift(12) décalent VERS LE PASSÉ : à la ligne t on lit les
    valeurs des lignes t-1 et t-12. Aucune donnée future n'entre ici.
    """
    return prix.shift(MOIS_EXCLUS) / prix.shift(MOIS_FORMATION) - 1


def eligibilite(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Qui est achetable à la fin du mois t.

    Tout est décalé d'un mois : à la formation, on ne connaît le prix et le
    volume que du mois PRÉCÉDENT clos. Utiliser ceux du mois t reviendrait
    à supposer qu'on connaît la clôture avant qu'elle n'ait lieu.
    """
    assez_cher = p["brut"].shift(1) >= PRIX_MINIMUM
    assez_liquide = p["dollars"].shift(1).rolling(3, min_periods=3).mean() >= VOLUME_DOLLAR_MIN
    existe = p["prix"].shift(1).notna()
    return assez_cher & assez_liquide & existe


def backtest(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prix = p["prix"]
    signal = signal_momentum(prix)
    admis = eligibilite(p)
    candidats = signal.where(admis & signal.notna())

    # Rendement du mois SUIVANT : c'est ce qu'on encaisse en détenant.
    rendement_suivant = prix.pct_change().shift(-1)

    dates = prix.index
    poids_precedents = pd.Series(0.0, index=prix.columns)
    lignes = []

    for date in dates:
        ligne = candidats.loc[date].dropna()
        if len(ligne) < 20:
            continue

        n = max(int(round(len(ligne) * PART_DECILE)), 5)
        selection = ligne.nlargest(n).index

        poids = pd.Series(0.0, index=prix.columns)
        poids[selection] = 1.0 / n

        # Rotation : part du portefeuille effectivement remplacée.
        rotation = float((poids - poids_precedents).abs().sum() / 2)
        cout = rotation * 2 * COUT_UNE_JAMBE

        r_titres = rendement_suivant.loc[date, selection]
        if r_titres.isna().all():
            continue
        brut = float(r_titres.mean(skipna=True))

        # Référence : l'univers éligible équipondéré, même mois.
        univers = rendement_suivant.loc[date, ligne.index]
        r_univers = float(univers.mean(skipna=True))

        lignes.append({
            "date": date, "n_candidats": len(ligne), "n_retenus": n,
            "brut": brut, "rotation": rotation, "cout": cout,
            "net": brut - cout, "univers": r_univers,
        })
        poids_precedents = poids

    return pd.DataFrame(lignes).set_index("date").dropna(subset=["net"])


def mesures(rendements: pd.Series) -> dict:
    r = rendements.dropna()
    if len(r) < 12:
        return {}
    cumul = (1 + r).prod()
    annees = len(r) / 12
    cagr = cumul ** (1 / annees) - 1
    vol = r.std() * np.sqrt(12)
    courbe = (1 + r).cumprod()
    drawdown = float((courbe / courbe.cummax() - 1).min())
    return {
        "mois": len(r),
        "cagr": cagr,
        "volatilite": vol,
        "sharpe": cagr / vol if vol else np.nan,
        "drawdown_max": drawdown,
        "mois_positifs": float((r > 0).mean()),
        "pire_mois": float(r.min()),
        "meilleur_mois": float(r.max()),
    }


def afficher(nom: str, m: dict) -> None:
    if not m:
        print(f"  {nom} : pas assez d'historique")
        return
    print(f"  {nom:<28}{m['cagr']:>9.2%}{m['volatilite']:>9.2%}"
          f"{m['sharpe']:>9.2f}{m['drawdown_max']:>11.1%}{m['mois_positifs']:>9.0%}")


def test_look_ahead(donnees: pd.DataFrame, nb_dates: int = 6) -> bool:
    """
    Recalcule le signal en ne fournissant QUE le passé, et compare.

    Si le signal du 31/03/2015 change selon qu'on lui donne les données
    jusqu'en 2026 ou seulement jusqu'en mars 2015, c'est qu'il regarde
    le futur. Ce test peut échouer : c'est ce qui le rend utile.
    """
    complet = signal_momentum(panneau_mensuel(donnees)["prix"])
    dates = complet.index[36:-2]
    if len(dates) == 0:
        return True
    choisies = dates[:: max(len(dates) // nb_dates, 1)][:nb_dates]

    print(f"  {'date de formation':<22}{'titres':>9}{'écart max':>14}")
    print("  " + "-" * 46)
    tout_bon = True
    for date in choisies:
        tronque = donnees[donnees["date"] <= date]
        partiel = signal_momentum(panneau_mensuel(tronque)["prix"])
        if date not in partiel.index:
            print(f"  {date:%d/%m/%Y}       date absente du calcul tronqué")
            tout_bon = False
            continue
        a, b = complet.loc[date], partiel.loc[date]
        communs = a.index.intersection(b.index)
        ecart = float((a[communs] - b[communs]).abs().max(skipna=True))
        etat = "" if (np.isnan(ecart) or ecart < 1e-9) else "   <-- ÉCART"
        if not (np.isnan(ecart) or ecart < 1e-9):
            tout_bon = False
        print(f"  {date:%d/%m/%Y}{len(communs):>16}{ecart:>14.2e}{etat}")
    return tout_bon


def main() -> int:
    print("=" * 74)
    print("ÉTAPE 4 — MOMENTUM 12-1".center(74))
    print("=" * 74)
    print()

    donnees = pd.read_parquet(FICHIER_PROPRE)
    donnees["date"] = pd.to_datetime(donnees["date"])
    print(f"Base : {donnees['ticker'].nunique()} titres, "
          f"{len(donnees):,} lignes".replace(",", " "))
    print()

    print("-" * 74)
    print("  TEST ANTI-LOOK-AHEAD (le signal regarde-t-il le futur ?)")
    print("-" * 74)
    if not test_look_ahead(donnees):
        print()
        print("  ÉCHEC. Le signal dépend de données postérieures à sa date.")
        print("  Tout résultat de backtest serait faux. On s'arrête ici.")
        return 1
    print("  OK — signal identique avec ou sans les données futures.")
    print()

    p = panneau_mensuel(donnees)
    resultats = backtest(p)

    print("-" * 74)
    print("  RÉSULTATS")
    print("-" * 74)
    print(f"  Période        : {resultats.index.min():%m/%Y} -> "
          f"{resultats.index.max():%m/%Y}  ({len(resultats)} mois)")
    print(f"  Titres retenus : {resultats['n_retenus'].mean():.0f} en moyenne "
          f"sur {resultats['n_candidats'].mean():.0f} candidats")
    print(f"  Rotation       : {resultats['rotation'].mean():.0%} par mois "
          f"-> {resultats['cout'].mean() * 12:.2%} de coûts par an")
    print()
    print(f"  {'':<28}{'perf/an':>9}{'volat.':>9}{'Sharpe':>9}"
          f"{'drawdown':>11}{'mois +':>9}")
    print("  " + "-" * 72)
    afficher("Momentum 12-1 (brut)", mesures(resultats["brut"]))
    afficher("Momentum 12-1 (net de frais)", mesures(resultats["net"]))
    afficher("Univers équipondéré", mesures(resultats["univers"]))
    print()

    net = mesures(resultats["net"])
    univ = mesures(resultats["univers"])
    if net and univ:
        ecart = net["cagr"] - univ["cagr"]
        print(f"  Écart net contre univers : {ecart:+.2%} par an")
        print()
        print("  SEUIL DE BRUIT MESURÉ : 1,6 % d'écart-type sur 11 ans.")
        print("  Sous ~3 % par an, le résultat ne se distingue pas de la chance.")
        print()
        print("  À RETRANCHER AVANT DE CONCLURE")
        print("  - biais du survivant : 1 à 4 points par an")
        print(f"  - il reste donc {ecart - 0.04:+.2%} à {ecart - 0.01:+.2%} par an")
        if ecart < 0.04:
            print("  -> L'avantage n'est PAS établi une fois le biais retranché.")
        else:
            print("  -> Un avantage subsiste, à confirmer hors échantillon.")
    print()

    print("-" * 74)
    print("  PERFORMANCE PAR ANNÉE (net de frais)")
    print("-" * 74)
    par_an = resultats.groupby(resultats.index.year).agg(
        momentum=("net", lambda x: (1 + x).prod() - 1),
        univers=("univers", lambda x: (1 + x).prod() - 1))
    print(f"  {'année':<8}{'momentum':>12}{'univers':>12}{'écart':>12}")
    print("  " + "-" * 44)
    for an, l in par_an.iterrows():
        d = l["momentum"] - l["univers"]
        print(f"  {an:<8}{l['momentum']:>11.1%}{l['univers']:>12.1%}{d:>12.1%}")
    print()

    resultats.to_csv(config.DOSSIER_RAPPORTS / "backtest_momentum.csv")
    print(f"  Détail mensuel : {config.DOSSIER_RAPPORTS / 'backtest_momentum.csv'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
