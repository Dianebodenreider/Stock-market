"""
config.py — Tous les paramètres du projet au même endroit.

C'est le SEUL fichier que tu modifies au quotidien.
Les autres fichiers lisent leurs réglages ici.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# EMPLACEMENTS DES FICHIERS
# ---------------------------------------------------------------------------
# RACINE = le dossier qui contient ce fichier config.py
RACINE = Path(__file__).parent

DOSSIER_DONNEES = RACINE / "data"
DOSSIER_BRUT = DOSSIER_DONNEES / "raw"          # les prix téléchargés
DOSSIER_RAPPORTS = DOSSIER_DONNEES / "rapports"  # les rapports de qualité

# Le fichier unique qui contiendra tout l'historique de prix.
# Format Parquet : 10 à 20 fois plus compact et plus rapide qu'un CSV.
FICHIER_PRIX = DOSSIER_BRUT / "prix.parquet"

# La liste des titres, mise en cache pour ne pas dépendre de Wikipedia
# à chaque exécution.
FICHIER_UNIVERS = DOSSIER_BRUT / "univers_sp500.csv"


# ---------------------------------------------------------------------------
# PÉRIODE D'HISTORIQUE
# ---------------------------------------------------------------------------
# 2005 permet d'inclure la crise de 2008. C'est important : une stratégie
# qui n'a jamais été testée sur un vrai krach n'a pas été testée.
DATE_DEBUT = "2005-01-01"

# None = jusqu'à aujourd'hui.
DATE_FIN = None


# ---------------------------------------------------------------------------
# TÉLÉCHARGEMENT
# ---------------------------------------------------------------------------
# Yahoo n'aime pas qu'on lui demande 500 titres d'un coup.
# On découpe en paquets de 50.
TAILLE_LOT = 50

# Pause entre deux lots, en secondes. Évite de se faire couper l'accès.
PAUSE_ENTRE_LOTS = 1.5

# Nombre de tentatives si un lot échoue.
TENTATIVES_MAX = 3


# ---------------------------------------------------------------------------
# SEUILS DES CONTRÔLES QUALITÉ
# ---------------------------------------------------------------------------
# Un titre avec moins de N jours de cotation est écarté :
# pas assez d'historique pour calculer un momentum 12 mois.
JOURS_MINIMUM = 300

# Un rendement quotidien au-delà de ce seuil est signalé comme suspect.
# 40 % en une séance sur une grande capitalisation, c'est presque toujours
# un split mal ajusté, pas un vrai mouvement.
SEUIL_RENDEMENT_SUSPECT = 0.40

# Nombre maximum de jours consécutifs sans cotation avant de signaler un trou.
# 5 jours ouvrés = une semaine entière manquante.
TROU_MAX_JOURS = 5

# Part maximale de séances à volume nul tolérée (5 %).
# Un volume nul signifie que le titre n'a pas été échangé : on ne peut
# ni acheter ni vendre ce jour-là.
PART_MAX_VOLUME_NUL = 0.05


# ---------------------------------------------------------------------------
# CRÉATION AUTOMATIQUE DES DOSSIERS
# ---------------------------------------------------------------------------
for _dossier in (DOSSIER_DONNEES, DOSSIER_BRUT, DOSSIER_RAPPORTS):
    _dossier.mkdir(parents=True, exist_ok=True)
