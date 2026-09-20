from pathlib import Path

import torch
from torch import nn

from learn_gpt.commons.plotting import plt, save_figure, set_title
from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title
from learn_gpt.text.models import CharacterModel, ContextCharacterModel, train
from learn_gpt.text.tokenizer import BOS_ID, EOS_ID, CharTokenizer

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
        token_ids = tokenizer.encode(mot)
        xs += token_ids[:-1]  # Tous sauf le dernier
        ys += token_ids[1:]  # Tous sauf le premier (décalage de 1)

    print_indented(f"Nombre de paires (id token, id token suivant): {len(xs)}", 2)
    pairs = [
        (
            tokenizer.decode([i], keep_special=True),
            tokenizer.decode([j], keep_special=True),
        )
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
        tops = char_model.top_next_tokens(tokenizer.encode(c, add_special=False)[0])
        next_tokens = [
            (tokenizer.decode([i], keep_special=True), prob) for i, prob in tops
        ]
        print_indented(f"Après '{c}': {next_tokens}", 2)
    print_new_line()

    print_indented("Génération avec le modèle entraîné:", 1)
    # Pour que les générations soient reproductibles
    torch.manual_seed(42)  # Parce que 42 est la réponse à la grande question sur la vie

    print_indented("Avec température T=0.8:", 2)
    for _ in range(10):
        tokens = char_model.generate(BOS_ID, EOS_ID, temperature=0.8)
        print_indented(tokenizer.decode(tokens), 3)

    print_indented("Avec température T=0.4:", 2)
    for _ in range(10):
        tokens = char_model.generate(BOS_ID, EOS_ID, temperature=0.4)
        print_indented(tokenizer.decode(tokens), 3)


def cmd_v2(
    *,
    embedding_dim: int,
    block_size: int,
    layer_size: int,
    lr: float,
    epochs: int,
    filename: str,
) -> None:
    if block_size < 1:
        raise ValueError("block_size doit être au moins 1")

    print_title("Entraîner et tester le modèle v2")
    print_indented("Un modèle avec une fenêtre de contexte", 1)
    print_new_line()

    print_indented("Création du tokenizer", 1)
    tokenizer = CharTokenizer.train("".join(MOTS))
    print_indented(f"vocab_size = {len(tokenizer.vocab)}", 2)
    print_new_line()

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    context_char_model = ContextCharacterModel(
        len(tokenizer.vocab),
        embedding_dim,
        block_size,
        layer_size,
    )
    n_parameters = sum(p.numel() for p in context_char_model.parameters())
    print_indented(f"Nombre de paramètres = {n_parameters}", 2)
    print_new_line()

    print_indented("Préparation des données d'entraînement", 1)
    print_indented(f"block_size = {block_size}", 2)

    # Cette fois, chaque exemple est une liste de block_size tokens
    xs: list[list[int]] = []
    ys: list[int] = []

    # Pour chaque mot du corpus
    for mot in MOTS:
        # Mot encadré par <bos> et <eos>
        token_ids = tokenizer.encode(mot)

        # On ajoute (block_size - 1) <bos> à gauche
        # Par exemple avec block_size = 3, ça donne
        # [<bos>, <bos>, <bos>, c, h, a, t, <eos>]
        padded = [BOS_ID] * (block_size - 1) + token_ids

        for i in range(len(padded) - block_size):
            # x contient les blocs successifs de block_size tokens
            xs.append(padded[i : i + block_size])
            # y est le token qui vient juste après
            ys.append(padded[i + block_size])

    print_indented("Extraits:", 2)
    for i in range(5):
        x_sample = [tokenizer.inv_vocab[j] for j in xs[i]]
        y_sample = tokenizer.inv_vocab[ys[i]]
        print_indented(f"{x_sample} -> {y_sample}", 3)

    # On construit nos données d'entraînement
    x = torch.tensor(xs)
    y = torch.tensor(ys)

    print_new_line()

    print_indented("Entraînement ...", 1)
    loss_history = train(
        context_char_model,
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
        "Courbe d'apprentissage du modèle v2",
        f"embeddings = {embedding_dim}, contexte = {block_size}, "
        f"{epochs} époques, lr={lr}",
    )
    plt.grid(True)
    save_figure(filepath)
    print_indented(f"Courbe d'apprentissage créée sous {filepath}", 2)
    print_new_line()

    print_indented("Génération avec le modèle entraîné:", 1)
    # Pour que les générations soient reproductibles
    torch.manual_seed(42)  # Parce que 42 est la réponse à la grande question sur la vie

    print_indented("Avec température T=0.8:", 2)
    for _ in range(10):
        tokens = context_char_model.generate(BOS_ID, EOS_ID, temperature=0.8)
        print_indented(tokenizer.decode(tokens), 3)

    print_indented("Avec température T=0.4:", 2)
    for _ in range(10):
        tokens = context_char_model.generate(BOS_ID, EOS_ID, temperature=0.4)
        print_indented(tokenizer.decode(tokens), 3)
