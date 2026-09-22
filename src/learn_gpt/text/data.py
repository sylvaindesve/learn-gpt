from collections import Counter
from math import log

from learn_gpt.text.tokenizer import CharTokenizer

# La liste de noms d'animaux pour cette section
MOTS = [
    "chat",
    "chien",
    "loup",
    "ours",
    "renard",
    "lapin",
    "souris",
    "grenouille",
    "oiseau",
    "serpent",
    "abeille",
    "papillon",
    "escargot",
    "tourterelle",
    "sanglier",
    "blaireau",
]


# Encode les mots puis les découpe en fenêtres glissantes de taille
# block_size (nécessaire à partir de la v2).
# Renvoie la liste des fenêtres et la liste des tokens qui viennent
# après
def to_train_data(
    words: list[str], block_size: int, tokenizer: CharTokenizer, bos_id: int
) -> tuple[list[list[int]], list[int]]:
    # Chaque exemple est une liste de block_size tokens
    xs: list[list[int]] = []
    ys: list[int] = []

    # Pour chaque mot du corpus
    for word in words:
        # Mot encadré par <bos> et <eos>
        token_ids = tokenizer.encode(word)

        # On ajoute (block_size - 1) <bos> à gauche
        # Par exemple avec block_size = 3, ça donne
        # [<bos>, <bos>, <bos>, c, h, a, t, <eos>]
        padded = [bos_id] * (block_size - 1) + token_ids

        for i in range(len(padded) - block_size):
            # x contient les blocs successifs de block_size tokens
            xs.append(padded[i : i + block_size])
            # y est le token qui vient juste après
            ys.append(padded[i + block_size])

    return xs, ys


# Le corpus sous forme de flux continu : les mots encadrés par <bos> et <eos>
# sont mis bout à bout (nécessaire à partir de la v5).
# C'est sous cette forme qu'un vrai GPT est entraîné : on tire ensuite des
# fenêtres au hasard dans ce flux, et une fenêtre peut donc à cheval sur deux
# mots.
def to_stream(words: list[str], tokenizer: CharTokenizer) -> list[int]:
    return [token for word in words for token in tokenizer.encode(word)]


# Les paires (préfixe, token suivant) vues à chaque position de chaque fenêtre
# du flux.
# Elles servent à calculer le plancher lorsque la perte porte sur toutes les
# positions : à la position j d'une fenêtre, le modèle ne dispose que du
# préfixe de longueur j + 1 pour prédire le token suivant.
def stream_pairs(
    stream: list[int], block_size: int
) -> tuple[list[list[int]], list[int]]:
    contexts: list[list[int]] = []
    targets: list[int] = []

    for i in range(len(stream) - block_size):
        for j in range(block_size):
            contexts.append(stream[i : i + j + 1])
            targets.append(stream[i + j + 1])

    return contexts, targets


# Perte minimale que l'on peut atteindre sur un jeu de données : l'entropie
# conditionnelle empirique de la cible sachant le contexte.
# Un contexte qui mène toujours au même token n'apporte aucune incertitude.
# Un contexte ambigu, comme [<bos>, c, h] qui peut être suivi de 'a' (chat)
# ou de 'i' (chien), en apporte : aucun modèle ne peut prédire mieux que la
# distribution observée dans le corpus.
def loss_floor(xs: list[list[int]], ys: list[int]) -> float:
    # Pour chaque contexte, la distribution des tokens qui le suivent
    counts_by_context: dict[tuple[int, ...], Counter[int]] = {}
    for context, target in zip(xs, ys, strict=True):
        counts_by_context.setdefault(tuple(context), Counter())[target] += 1

    total = len(xs)
    floor = 0.0

    for counts in counts_by_context.values():
        n = sum(counts.values())
        # Entropie de la distribution des cibles pour ce contexte
        entropy = -sum((c / n) * log(c / n) for c in counts.values())
        # Pondérée par la fréquence du contexte dans le jeu de données
        floor += (n / total) * entropy

    return floor
