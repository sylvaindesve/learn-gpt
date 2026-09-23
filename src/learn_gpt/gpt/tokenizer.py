import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

# Tokens spéciaux
# <pad> permet de compléter une séquence plus courte que la taille d'un lot lors de
#       l'entraînement du modèle de langage
# <bos> indique le début d'une séquence
# <eos> indique la fin d'une séquence
# <unk> indique un token inconnu (i.e. un token non vu par le tokenizer lors de son
#       entraînement)
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# Un tokenize utilisant l'algorithme Byte Pair Encoding
# Le principe est d'itérer sur le texte en remplaçant progressivement les
# paires les plus courantes.
class BPETokenizer:
    # On initialise un tokenizer à partir de :
    # - la liste des fusions qu'il applique : ces fusions sont dans l'ordre où elles ont
    #   été apprises
    # - son vocabulaire, i.e. le dictionnaire token -> ID de token
    def __init__(self, merges: list[tuple[str, str]], vocab: dict[str, int]):
        self.merges = merges
        # Les rangs des fusions, de la forme paire -> rang de fusion
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.vocab = vocab
        # Vocabulaire inverse : ID de token -> token
        self.inv_vocab = {i: tok for tok, i in vocab.items()}

    # Création d'un BPETokenizer par entraînement sur le texte fournie
    # Le tokenizer s'entraîne jusqu'à atteindre la taille de vocabulaire cible.
    # L'apprentissage des fusions est long (des minutes sur un vrai corpus), d'où
    # le `logger` qui permet de suivre l'avancement.
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        logger: Callable[[str], None] = lambda _: None,
    ) -> "BPETokenizer":
        # On commence par obtenir la fréquence des mots
        word_freqs = _get_word_freqs(text)
        merges = []

        # Le vocabulaire initial : les caractères qui composent les mots
        vocab_chars = {c for word in word_freqs for c in word}

        # Il faut compter les tokens spéciaux dans la taille cible
        target = vocab_size - len(SPECIAL_TOKENS)

        # Tant qu'on est pas à la taille cible:
        while len(vocab_chars) < target:
            # On récupère la fréquences des paires dans les mots
            pairs = _get_pair_freqs(word_freqs)

            # Cas extrême où l'on a tout consommé
            if not pairs:
                break

            # La paire la plus fréquente
            # `most_common` trie les paires par fréquence décroissante, on ne
            # garde que la première
            best_pair = pairs.most_common(1)[0][0]

            # On met à jour les mots en y fusionnant la meilleure paire
            word_freqs = _merge_pair(best_pair, word_freqs)

            # On stocke cette paire apprise
            merges.append(best_pair)

            # Et on rajoute la paire fusionnée à notre vocabulaire
            vocab_chars.add(best_pair[0] + best_pair[1])

            if len(merges) % 50 == 0:
                logger(
                    f"{len(merges)} fusions, "
                    f"vocabulaire {len(vocab_chars) + len(SPECIAL_TOKENS)}/{vocab_size}"
                )

        # Maintenant qu'on a atteint la taille cible, on représente notre vocabulaire
        # sous la forme { token: ID de token }

        # Les premiers sont les tokens spéciaux
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

        # Viennent ensuite les autres, triés
        for tok in sorted(vocab_chars):
            vocab.setdefault(tok, len(vocab))

        # On renvoie un nouveau BPETokenizer avec ces fusions et ce vocabulaire
        return cls(merges, vocab)

    # Encodage d'un mot en appliquant les fusions apprises
    # Par exemple, "mot" pourrait devenir [m, ot, </w>]
    def _encode_word(self, word: str) -> list[str]:
        # On transforme le mot en une liste de ses caractères et on ajoute
        # '</w>' à la fin car c'est comme ça que le tokenizer a appris
        symbols = [*list(word), "</w>"]

        # Tant qu'il reste des symboles à traiter dans le mot source:
        while len(symbols) > 1:
            # On prend toutes les paires successives
            # A la première itération, "mot" donne [(m, o), (o, t), (t, </w>)]
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]

            # On liste les fusions possibles des paires du mot en regardant si la
            # paire est connue des fusions de ce tokenizer et avec quel rang de fusion
            # Par exemple, si ce tokenizer fusionne (o, t) et (t, </w>) on aurait
            # [(1, (o, t)), (2, (t, </w>))]
            ranked = [(self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks]

            # Si vide c'est qu'on a fusionné tout ce qu'on pouvait fusionner
            if not ranked:
                break

            # Sinon, on prend le plus petit rang (ie. la fusion apprise en premier)
            # (le min trie selon le premier élément du tuple)
            _, best_pair = min(ranked)

            new_symbols, i = [], 0
            # Tant qu'on a pas consommé tous les symboles du mot:
            while i < len(symbols):
                # Si on repère notre meilleure paire
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    # On la remplace
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    # Et on avance de 2 car on a remplacé une paire
                    i += 2
                else:
                    # Sinon, on ajoute juste le symbole
                    new_symbols.append(symbols[i])
                    i += 1
            # On remplace et c'est reparti pour identifier la prochaine fusion à faire
            symbols = new_symbols
        return symbols

    # Encodage d'un texte en une liste d'ID de tokens
    # add_special indique s'il faut encadrer avec <bos> et <eos>
    def encode(self, text: str, add_special: bool = True) -> list[int]:
        # S'il faut on met <bos> au début
        ids = [BOS_ID] if add_special else []

        # On découpe en mots
        for word in re.findall(r"\S+", text):
            # On applique les fusions apprises sur le mot
            for tok in self._encode_word(word):
                # Puis on utilise le vocabulaire pour transformer en ID
                ids.append(self.vocab.get(tok, UNK_ID))
        # S'il faut on met <eos> à la fin
        if add_special:
            ids.append(EOS_ID)
        return ids

    # Decodage d'une liste d'ID de tokens en un texte
    def decode(self, ids: list[int]) -> str:
        # On utilise simplement notre vocabulaire inverse
        # - en remplaçant les inconnus par <unk>
        # - en ignorant les autres tokens spéciaux
        toks = [
            self.inv_vocab.get(i, "<unk>")
            for i in ids
            if i not in (PAD_ID, BOS_ID, EOS_ID)
        ]

        # On met tout ça bout à bout
        text = "".join(toks)

        # Et on remet les espaces après les fins de mots
        return text.replace("</w>", " ").strip()

    # Pour mettre toutes les séquences d'ID de tokens à la même taille
    # en complétant à droite par <pad>
    @staticmethod
    def pad_batch(sequences: list[list[int]]) -> list[list[int]]:
        # La taille de la séquence la plus longue
        max_len = max(len(s) for s in sequences)

        # On rajoute <pad> à droite pour les séquences plus courtes
        return [s + [PAD_ID] * (max_len - len(s)) for s in sequences]

    # Sauvegarde ce tokenizer dans un fichier JSON
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            # Il suffit de garder les fusions et le vocabulaire
            json.dump(
                {"merges": self.merges, "vocab": self.vocab}, f, ensure_ascii=False
            )

    # Charge un tokenizer à partir d'une sauvegarde JSON
    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # JSON ne gère pas les tuples donc y'a une deserialisation sur les fusions
        merges = [tuple(p) for p in data["merges"]]
        return cls(merges, data["vocab"])


# Renvoie les fréquences des mots dans le texte
def _get_word_freqs(text: str) -> dict[tuple[str, ...], int]:
    # Liste de tous les mots dans le corpus
    words = re.findall(r"\S+", text)
    # `Counter` compte la fréquence de chaque mot
    freqs = Counter(words)
    # La fréquence des mots est envoyée sous la forme:
    # { ('m', 'o', 't', '</w>'): 3 }
    # '</w>' indique la fin du mot
    return {(*tuple(w), "</w>"): c for w, c in freqs.items()}


# Renvoie la fréquence des paires en partant de la fréquence des mots
def _get_pair_freqs(
    word_freqs: dict[tuple[str, ...], int],
) -> Counter[tuple[str, str]]:
    # On initialise un `Counter` vide
    pairs: Counter[tuple[str, str]] = Counter()
    # Pour chaque mot dans le corpus:
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            # On parcourt les lettres du mot 2 par 2
            # et on ajoute la fréquence du mot à la fréquence
            # de la paire
            # Quand `Counter` ne connaît pas une clé, il
            # commence à zéro
            pairs[(word[i], word[i + 1])] += freq
    # A noter que les paires renvoyées peuvent contenir
    # par exemple ('e', '</w>').
    # Cela permet d'indiquer la fréquence d'un 'e' en fin
    # de mot.
    return pairs


# Fusionne la paire dans les mots
# Par exemple { ('m', 'o', 't', '</w>'): 3 } devient { ('mo', 't', '</w>'): 3 }
def _merge_pair(
    pair: tuple[str, str], word_freqs: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    a, b = pair
    new_freqs: dict[tuple[str, ...], int] = {}

    # Pour chaque mot dans le corpus:
    for word, freq in word_freqs.items():
        new_word, i = [], 0
        # On parcout le mot
        while i < len(word):
            # Si on est pas arrivé à la fin du mot et qu'on repère la paire:
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                # On ajoute la paire fusionnée
                new_word.append(a + b)
                # On avance de 2 car on a traité une paire
                i += 2
            else:
                # Sinon on ajoute seulement l'existant
                new_word.append(word[i])
                i += 1
        new_freqs[tuple(new_word)] = freq
    return new_freqs
