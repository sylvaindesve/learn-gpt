from pathlib import Path

import torch
from torch import nn

from learn_gpt.commons.plotting import plt, save_figure, set_title
from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title
from learn_gpt.text.models import CharacterModel, top_next_tokens, train
from learn_gpt.text.tokenizer import CharTokenizer

OUTPUT_DIR = Path.cwd() / "output" / "text"

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


# Softmax pour obtenir une distribution de probabilités
# à partir de logits
# Pour la suite, nous utiliserons plutôt les fonctions proposées
# par PyTorch, notamment nn.CrossEntropyLoss() qui fait le softmax
# et le -log(p)
def softmax(logits: torch.Tensor) -> torch.Tensor:
    e = torch.exp(logits - logits.max())  # On décale pour éviter les débordements
    return e / e.sum()


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


def cmd_loss() -> None:
    print_title("Calculer la perte sur du texte")
    print_new_line()

    logits = torch.tensor([1.0, 3.0, 0.5])
    print_indented(
        f"logits: {logits.tolist()} → la 2ème classe a le plus haut score, "
        f"c'est ce que le modèle prédirait",
        2,
    )

    probs = softmax(logits)
    print_indented(
        "Distribution des probabilités selon le modèle (softmax): "
        f"{[round(p, 3) for p in probs.tolist()]}",
        2,
    )

    # La perte ne dépend que de la classe attendue
    target_class = 1  # La 2ème classe est la bonne réponse
    loss = -torch.log(probs[target_class])
    print_indented(f"Perte si la bonne classe est la 2ème : {loss.item():.3f}", 2)

    target_class = 2  # La 3ème classe est la bonne réponse
    loss = -torch.log(probs[target_class])
    print_indented(f"Perte si la bonne classe est la 3ème : {loss.item():.3f}", 2)


def cmd_v1(*, embedding_dim: int, lr: float, epochs: int, filename: str) -> None:
    print_title("Entraîner et tester le modèle v1")
    print_indented("Un modèle qui prédit le prochain caractère", 1)
    print_new_line()

    print_indented("Création du tokenizer", 1)
    tokenizer = CharTokenizer.train("".join(MOTS))
    print_indented(f"vocab_size = {len(tokenizer.vocab)}", 2)
    print_new_line()

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    char_model = CharacterModel(len(tokenizer.vocab), embedding_dim)
    print_indented(
        f"Nombre de paramètres = {sum(p.numel() for p in char_model.parameters())}", 2
    )
    print_new_line()

    print_indented("Préparation des données d'entraînement", 1)

    xs: list[int] = []
    ys: list[int] = []
    # Pour chaque mot du corpus
    for mot in MOTS:
        token_ids = tokenizer.encode(mot, add_special=False)
        xs += token_ids[:-1]  # Tous sauf le dernier
        ys += token_ids[1:]  # Tous sauf le premier (décalage de 1)

    print_indented(f"Nombre de paires (id token, id token suivant): {len(xs)}", 2)
    pairs = [
        (tokenizer.decode([i]), tokenizer.decode([j]))
        for i, j in zip(xs[:8], ys[:8], strict=True)
    ]
    print_indented(f"Premières paires: {pairs}", 2)

    # On construit nos données d'entraînement
    x = torch.tensor(xs)
    y = torch.tensor(ys)

    print_new_line()

    print_indented("Entraînement ...", 1)
    loss_history = train(
        char_model,
        x,
        y,
        lr=lr,
        epochs=epochs,
        logger=lambda s: print_indented(s, 2),
    )
    print_indented("Entraînement terminé", 1)
    print_indented(f"Perte finale = {loss_history[-1]:.2f}", 2)
    print_new_line()

    print_indented("Création de la visualisation de la courbe d'apprentissage", 1)
    filepath = OUTPUT_DIR / filename
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Époque")
    plt.ylabel("Entropie croisée")
    set_title(
        "Courbe d'apprentissage du modèle v1",
        f"taille des embeddings = {embedding_dim}, {epochs} époques, lr={lr}",
    )
    plt.grid(True)
    save_figure(filepath)
    print_indented(f"Courbe d'apprentissage créée sous {filepath}", 2)
    print_new_line()

    print_indented("Prédictions du modèle entraîné:", 1)

    for c in ["c", "s", "t"]:
        tops = top_next_tokens(char_model, tokenizer.encode(c, add_special=False)[0])
        print_indented(
            f"Après '{c}': {[(tokenizer.decode([i]), prob) for i, prob in tops]}", 2
        )
