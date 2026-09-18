# Comprendre le fonctionnement d'un modèle GPT

L'objectif de ce projet est de comprendre comment fonctionne un modèle de langage GPT. Il existe déjà [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) d'Andrej Karpathy. Ici, l'objectif est d'aller un petit plus loin en ajoutant :

- un tokenizer
- l'utilisation de PyTorch
- la visualisation de l'entraînement avec des graphiques
- l'ajout de quelques techniques influant sur l'entraînement
- la sauvegarde et le chargement d'un modèle pour pouvoir jouer avec sans avoir à le ré-entraîner
- une ligne de commande enrichie pour lancer les différentes expérimentations et apprendre au fur et à mesure

Ce projet retrace mon apprentissage en suivant la même progression.

## Pré-requis

Le projet est écrit en Python et utilise `uv` : la première étape est donc de [l'installer](https://docs.astral.sh/uv/getting-started/installation/). Il faudra ensuite lancer `uv sync`.

## Le voyage commence

Les modèles de langage (LLM) qui alimentent l'IA de nos jours sont l'héritage de longues recherches dans le domaine du Machine Learning (ML) puis du Deep Learning et en parallèle du Natural Language Processing (NLP). Je ne vais pas rentrer dans le détail sur le NLP mais il est essentiel de comprendre le Machine Learning pour aller plus loin.

### Le Machine Learning (ou apprentissage automatique)

> Je voudrais un moyen de connaître les émissions en CO₂ d'un véhicule

Si je devais écrire un programme pour répondre à cette question, il me faudrait connaître la formule permettant de déterminer ce niveau d'émission en fonction de différentes caractéristiques du véhicule.

Je n'ai pas cette formule mais par contre j'ai accès à [un grand nombre de données sur les véhicules commercialisés en France](https://www.data.gouv.fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france) : émissions de CO₂ et différentes caractéristiques du véhicule.

C'est là que le Machine Learning peut m'aider. Le principe de base est le suivant : la "machine" apprend à partir d'un grand nombre d'exemples et généralise pour être en mesure de faire des prédictions sur des données qu'elle n'a jamais vues.

Le cas le plus simple de Machine Learning est la **régression linéaire**.

Nous allons prendre nos données sur les véhicules dans le fichier [data/co2/fic_etiq_edition_40-mars-2015.csv](./data/co2/fic_etiq_edition_40-mars-2015.csv) (téléchargé automatiquement au premier lancement) et nous intéresser à la colonne `co2_mixte` qui représente les émissions en CO₂ en g/km ainsi qu'à la colonne `conso_mixte` qui représente la consommation en L/100km.

La commande `uv run learn-gpt linear data` permet de représenter visuellement les données sous la forme d'un graphique dans le fichier [output/linear/data.png](./output/linear/data.png).

On peut ainsi voir qu'il semble exister une relation linéaire entre les deux informations. Nous allons créer un modèle qui va prédire les émissions en CO₂, notées $\hat{y}$, en fonction de la consommation, notée $x$ :

```math
\hat{y} = wx + b
```

Où $w$ est la pente (ou poids) et $b$ l'ordonnée à l'origine (ou biais).

La technique pour trouver les meilleures valeurs de $w$ et $b$ est la **descente de gradient** :

1. On initialise $w$ et $b$ avec des valeurs arbitraires ou aléatoires
2. Pour chaque exemple $x_i$, on calcule la prédiction $\hat{y} = wx_i + b$
3. On calcule la perte, c'est-à-dire l'écart à la prédiction
4. On calcule les dérivées partielles de la perte par rapport à $w$ et $b$
5. On ajuste $w$ et $b$ en utilisant leurs dérivées partielles et un coefficient d'apprentissage
6. On recommence

Il existe plusieurs façons de calculer la perte. Pour notre exemple, on utilise l'erreur quadratique moyenne (MSE, Mean Square Error).

$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - (w x_i + b) \right)^2$$

[Détail du calcul des dérivées partielles](./docs/derivees_partielles.md)

La commande `uv run learn-gpt linear train` permet de lancer l'entraînement du modèle, de visualiser le résultat dans [output/linear/data.png](./output/linear/regression.png) et la courbe d'apprentissage dans [output/linear/data.png](./output/linear/learn.png).

Le code associé est dans [src/learn_gpt/linear/train.py](./src/learn_gpt/linear/train.py). Il utilise les tenseurs de PyTorch : une petite introduction est présente dans [docs/tensor.md](./docs/tensor.md).
