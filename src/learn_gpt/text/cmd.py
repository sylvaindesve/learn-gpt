import torch
from torch import nn

from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title
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


def cmd_show() -> None:
    print_title("Représenter les mots sous forme de nombres")
    print_new_line()

    print_indented("Un petit corpus de noms d'animaux:", 1)
    for mot in MOTS:
        print_indented(mot, 2)
    print_new_line()

    uchars = sorted(set("".join(MOTS)))

    print_indented(f"{len(MOTS)} mots, {len(uchars)} caractères uniques", 1)
    print_new_line()

    print_indented("Création du tokenizer", 1)
    tokenizer = CharTokenizer.train("".join(MOTS))

    mot = "chat"
    mot_tokens = tokenizer.encode("chat", add_special=False)

    sanglier_tokens = tokenizer.encode("sanglier", add_special=False)

    print_indented(f"'{mot}' -> {mot_tokens}", 2)
    print_indented(
        f"{sanglier_tokens} -> '{tokenizer.decode(sanglier_tokens)}'",
        2,
    )
    print_new_line()

    print_indented("Création de la matrice d'embedding", 1)
    torch.manual_seed(0)
    c_tensor = torch.tensor(tokenizer.encode("c", add_special=False))
    wte = nn.Embedding(len(tokenizer.vocab), 4)
    print_indented(f"Embedding pour 'c': {wte(c_tensor).tolist()}", 2)
