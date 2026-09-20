SPECIAL_TOKENS = ["<bos>", "<eos>", "<unk>"]
BOS_ID, EOS_ID, UNK_ID = 0, 1, 2


class CharTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {i: tok for tok, i in vocab.items()}

    # Créer un nouveau CharTokenizer en "s'entraînant" sur le
    # texte fourni.
    # Ici, il ne s'agit pas vraiment d'un entraînement mais
    # cela permet de coller à la structure de tokenizers plus
    # complexes.
    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        uchars = sorted(set(text))

        # Dictionnaire token -> ID token
        # Les premiers sont les tokens spéciaux
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok in uchars:
            vocab.setdefault(tok, len(vocab))

        return cls(vocab)

    # Encodage d'un texte en une liste d'ID de tokens
    # add_special indique s'il faut encadrer avec <bos> et <eos>
    def encode(self, text: str, add_special: bool = True) -> list[int]:
        # S'il faut on met <bos> au début
        ids = [BOS_ID] if add_special else []

        for c in text:
            ids.append(self.vocab.get(c, UNK_ID))

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
            self.inv_vocab.get(i, "<unk>") for i in ids if i not in (BOS_ID, EOS_ID)
        ]

        # On met tout ça bout à bout
        return "".join(toks)
