from pathlib import Path

from learn_gpt.commons.data import ensure_file, read_csv
from learn_gpt.gpt.tokenizer import UNK_ID, BPETokenizer

# Le corpus french_CEFR : des phrases françaises, séparées en trois jeux
#   https://huggingface.co/datasets/vekkt/french_CEFR
CEFR_BASE_URL = "https://huggingface.co/datasets/vekkt/french_CEFR/resolve/main"
CEFR_DIR = Path.cwd() / "data" / "french_CEFR"
CEFR_SPLITS = ("train", "val", "test")


# Télécharge les fichiers du corpus s'ils sont absents du disque.
# Renvoie la liste des jeux qui ont effectivement été téléchargés
def download_cefr() -> list[str]:
    downloaded: list[str] = []

    for split in CEFR_SPLITS:
        path = CEFR_DIR / f"{split}.csv"
        if ensure_file(path, f"{CEFR_BASE_URL}/{split}.csv"):
            downloaded.append(split)

    return downloaded


# Charge les phrases des trois jeux du corpus : (train, val, test)
def load_cefr() -> tuple[list[str], list[str], list[str]]:
    train = read_csv(CEFR_DIR / "train.csv")["sentence"]
    val = read_csv(CEFR_DIR / "val.csv")["sentence"]
    test = read_csv(CEFR_DIR / "test.csv")["sentence"]
    return train, val, test


# Le chemin du tokenizer sauvegardé pour une taille de vocabulaire donnée.
# Apprendre les fusions du BPE prend des minutes sur un vrai corpus : on le fait
# une fois, puis on recharge le résultat depuis ce fichier.
def tokenizer_path(vocab_size: int) -> Path:
    return CEFR_DIR / f"bpe-{vocab_size}.json"


# Le corpus sous forme de flux continu : les phrases encadrées par <bos> et
# <eos> sont mises bout à bout. On y tire ensuite des fenêtres au hasard.
def to_stream(texts: list[str], tokenizer: BPETokenizer) -> list[int]:
    return [token for text in texts for token in tokenizer.encode(text)]


# Caractères par token et part de tokens inconnus : les deux chiffres qui
# disent si le tokenizer est adapté au corpus.
# Le second est le plus parlant sur le jeu de validation, que le tokenizer n'a
# pas vu lors de son apprentissage.
def token_stats(texts: list[str], tokenizer: BPETokenizer) -> tuple[float, float]:
    n_characters = sum(len(text) for text in texts)
    ids = [i for text in texts for i in tokenizer.encode(text)]
    unknown = sum(1 for i in ids if i == UNK_ID)
    return n_characters / len(ids), unknown / len(ids)
