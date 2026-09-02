# Stock-market — Étape 1 : le socle de données

Ce dépôt construit une base de prix historiques propre et vérifiée pour le
S&P 500. Rien d'autre. Pas encore de stratégie, pas encore de backtest.

C'est volontaire. Un backtest bâti sur des données non vérifiées produit une
courbe de performance qui a l'air correcte et qui est fausse. L'erreur ne
fait pas planter le programme, elle ne lève aucune alerte, elle se découvre
en argent réel.

---

## Ce qu'on lance, dans l'ordre

Ouvre le terminal du Codespace (celui en bas de l'écran, avec `$`) et tape
les commandes suivantes, une par une.

### Une seule fois — l'installation

```bash
pip install -r requirements.txt
```

### À chaque fois — les trois commandes

```bash
# 0. Vérifier que le contrôle qualité fonctionne (5 secondes, sans internet)
python tests/test_hors_ligne.py

# 1. Télécharger les prix — commence par le mode test
python 01_telecharger.py --test     # 20 titres, ~30 secondes
python 01_telecharger.py            # les 500 titres, 5 à 15 minutes

# 2. Contrôler la qualité
python 02_verifier.py
```

Lance toujours `--test` avant le téléchargement complet. Si quelque chose ne
va pas, autant le découvrir en 30 secondes qu'en 15 minutes.

---

## Ce que fait chaque fichier

| Fichier | Rôle |
|---|---|
| `config.py` | Tous les réglages. **Le seul fichier que tu modifies.** |
| `src/univers.py` | Récupère la liste des 500 titres depuis Wikipédia |
| `src/telechargement.py` | Télécharge les prix via yfinance, les stocke en Parquet |
| `src/qualite.py` | Les 9 contrôles de qualité |
| `01_telecharger.py` | Commande à lancer pour l'étape 1 |
| `02_verifier.py` | Commande à lancer pour l'étape 2 |
| `tests/test_hors_ligne.py` | Vérifie que les contrôles détectent bien les erreurs |

Fichiers produits, dans `data/` :

- `raw/prix.parquet` — tout l'historique de prix (~150 Mo pour 500 titres sur 20 ans)
- `raw/univers_sp500.csv` — la liste des titres, figée
- `raw/univers_valide.csv` — les titres qui passent les contrôles
- `rapports/qualite_donnees.csv` — une ligne par titre, tous les diagnostics

---

## Les deux décisions structurantes

### 1. On garde le prix brut ET le prix ajusté

`close` est le cours réellement coté. `adj_close` est ce cours retraité des
dividendes et des divisions d'actions.

**Tous les calculs de rendement se font sur `adj_close`.** Sinon :

- une valeur qui verse 4 % de dividende par an voit sa performance amputée
  d'environ 55 % en cumulé sur 20 ans ;
- une action qui fait un split 4 pour 1 apparaît comme une chute de −75 % en
  une séance, et le signal croit voir un krach.

On conserve quand même `close`, parce que l'écart entre les deux séries est
la preuve que l'ajustement a bien eu lieu. C'est une pièce à conviction, pas
une donnée de travail.

### 2. Le biais du survivant est présent, assumé, et chiffré

L'univers est la composition **actuelle** du S&P 500. Les sociétés sorties de
l'indice — faillites, rachats, effondrements — n'y figurent pas.

Un backtest lancé sur cette liste ne pourra jamais acheter Lehman Brothers en
2007 ni First Republic en 2022. On teste une stratégie sur un univers dont
les désastres ont été retirés à l'avance.

**Ordre de grandeur : 1 à 4 points de performance annuelle surestimée.**

Concrètement : si le backtest sort +12 % par an, la vraie performance est
probablement entre +8 % et +11 %. Ce chiffre doit être retranché mentalement
au moment de décider si la stratégie mérite un euro.

Supprimer ce biais demande un fournisseur qui conserve les titres délistés
(Norgate ~60 $/mois, CRSP, Refinitiv). Pour apprendre, on accepte le biais.
Mais on l'écrit noir sur blanc, à trois endroits du code, pour ne jamais
l'oublier au moment de lire un résultat flatteur.

---

## Les 9 contrôles de qualité

Chacun correspond à une erreur classique. En regard, ce qu'elle produit si
on ne la détecte pas.

| Contrôle | Ce qui arrive si on ne le fait pas |
|---|---|
| Historique insuffisant (< 300 séances) | Le momentum 12 mois renvoie `NaN`, la ligne disparaît en silence, l'univers réel du backtest n'est pas celui qu'on croit |
| Dates en double | La séance compte double dans toutes les moyennes et tout rendement cumulé |
| Prix nuls ou négatifs | Rendement de −100 %, puis division par zéro le lendemain |
| Volume nul excessif | Le backtest « achète » un jour où le titre n'a pas été échangé — aucun ordre n'aurait pu passer |
| Lignes OHLC incohérentes | Un plus-bas au-dessus du plus-haut fausse tout calcul de stop |
| Split non retraité | Le signal lit un effondrement de −50 % qui n'a jamais eu lieu |
| Ajustement absent | La performance est amputée de tous les dividendes, silencieusement |
| Trous > 5 jours ouvrés | Les calculs glissants sont faussés sans qu'aucune erreur ne soit levée |
| Rendements ajustés > 40 % | Signale un split résiduel ou une donnée corrompue |

Les deux derniers sont **signalés mais n'écartent pas** le titre : un trou
peut venir d'une suspension de cotation légitime, et −26 % en une séance
peut être réel (Meta, février 2022). Écarter automatiquement les fortes
baisses reviendrait à fabriquer soi-même un second biais du survivant.

Le module **ne corrige rien**. Il signale, tu décides. Une correction
silencieuse est aussi dangereuse qu'une erreur silencieuse.

---

## Comment on sait que les contrôles marchent

`tests/test_hors_ligne.py` fabrique 12 titres synthétiques dont on connaît
exactement les défauts — dont un split correctement retraité qui ne doit
**pas** être écarté — et vérifie que chaque anomalie est détectée, sans
faux positif.

C'est le seul moyen d'avoir confiance dans un contrôle : lui montrer une
erreur mise là exprès et vérifier qu'il la voit. Un contrôle qui ne signale
jamais rien n'est pas rassurant, il est cassé.

Ce test tourne sans internet, en 5 secondes.

---

## Étape suivante

Une fois `02_verifier.py` passé proprement : le calcul du momentum 12-1
(rendement sur 12 mois hors dernier mois), puis le backtest avec coûts de
transaction et mesure du drawdown.
