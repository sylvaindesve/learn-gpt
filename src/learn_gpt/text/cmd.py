from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from learn_gpt.commons.plotting import plot_learning_curves
from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title
from learn_gpt.text.data import (
    MOTS,
    download_cefr,
    load_cefr,
    loss_floor,
    stream_pairs,
    to_stream,
    to_train_data,
)
from learn_gpt.text.models.v1 import CharacterModel
from learn_gpt.text.models.v2 import ContextCharacterModel
from learn_gpt.text.models.v3 import AttentionCharacterModel
from learn_gpt.text.models.v4 import MultiHeadAttentionCharacterModel
from learn_gpt.text.models.v5 import StreamMultiHeadAttentionCharacterModel
from learn_gpt.text.models.v6 import MicroGPTModel
from learn_gpt.text.tokenizer import BOS_ID, EOS_ID, CharTokenizer
from learn_gpt.text.train import train, train_stream, train_stream_with_validation

OUTPUT_DIR = Path.cwd() / "output" / "text"


# Ce que toutes les versions du modèle savent faire : générer une séquence
class GenerativeModel(Protocol):
    def generate(
        self,
        bos_id: int,
        eos_id: int,
        temperature: float = 0.8,
        max_tokens: int = 20,
    ) -> list[int]: ...


# Crée le tokenizer d'un corpus et affiche la taille du vocabulaire
def create_tokenizer(corpus: list[str], label: str = "") -> CharTokenizer:
    suffix = f" ({label})" if label else ""
    print_indented(f"Création du tokenizer{suffix}", 1)
    tokenizer = CharTokenizer.train("".join(corpus))
    print_indented(f"vocab_size = {len(tokenizer.vocab)}", 2)
    print_new_line()
    return tokenizer


# Affiche le nombre de paramètres du modèle
def print_parameters(model: nn.Module) -> None:
    n_parameters = sum(p.numel() for p in model.parameters())
    print_indented(f"Nombre de paramètres = {n_parameters}", 2)
    print_new_line()


# Affiche les premiers exemples du jeu de données
def print_examples(
    xs: list[list[int]], ys: list[int], tokenizer: CharTokenizer, n: int = 5
) -> None:
    print_indented("Extraits:", 2)
    for i in range(n):
        context = [tokenizer.inv_vocab[j] for j in xs[i]]
        print_indented(f"{context} -> {tokenizer.inv_vocab[ys[i]]}", 3)


# Trace la courbe d'apprentissage et l'enregistre.
# La courbe de validation est optionnelle : `val_steps` donne alors l'étape de
# chaque point, puisque la validation n'est pas mesurée à chaque étape.
def plot_learning_curve(
    loss_history: list[float],
    *,
    title: str,
    context: str,
    filename: str,
    xlabel: str = "Époque",
    val_loss_history: list[float] | None = None,
    val_steps: list[int] | None = None,
) -> None:
    print_indented("Création de la visualisation de la courbe d'apprentissage", 1)
    filepath = OUTPUT_DIR / filename
    plot_learning_curves(
        loss_history,
        val_loss_history,
        title=title,
        context=context,
        filepath=filepath,
        xlabel=xlabel,
        ylabel="Entropie croisée",
        val_steps=val_steps,
    )
    print_indented(f"Courbe d'apprentissage créée sous {filepath}", 2)
    print_new_line()


# Génère et affiche quelques séquences, à plusieurs températures
def print_generations(
    model: GenerativeModel,
    tokenizer: CharTokenizer,
    *,
    temperatures: tuple[float, ...] = (0.8, 0.4),
    n_samples: int = 10,
    max_tokens: int = 20,
) -> None:
    print_indented("Génération avec le modèle entraîné:", 1)
    # Pour que les générations soient reproductibles
    torch.manual_seed(42)

    for temperature in temperatures:
        print_indented(f"Avec température T={temperature}:", 2)
        for _ in range(n_samples):
            tokens = model.generate(
                BOS_ID, EOS_ID, temperature=temperature, max_tokens=max_tokens
            )
            print_indented(tokenizer.decode(tokens), 3)


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


# Softmax pour obtenir une distribution de probabilités
# à partir de logits
# Pour la suite, nous utiliserons plutôt les fonctions proposées
# par PyTorch, notamment nn.CrossEntropyLoss() qui fait le softmax
# et le -log(p)
def softmax(logits: torch.Tensor) -> torch.Tensor:
    e = torch.exp(logits - logits.max())  # On décale pour éviter les débordements
    return e / e.sum()


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

    tokenizer = create_tokenizer(MOTS)

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    char_model = CharacterModel(len(tokenizer.vocab), embedding_dim)
    print_parameters(char_model)

    print_indented("Préparation des données d'entraînement", 1)

    # Le contexte est réduit à un seul token : chaque exemple est donc
    # simplement une paire (token, token suivant)
    xs: list[int] = []
    ys: list[int] = []
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
    print_new_line()

    print_indented("Entraînement ...", 1)
    loss_history = train(
        char_model,
        torch.tensor(xs),
        torch.tensor(ys),
        lr=lr,
        epochs=epochs,
        logger=lambda s: print_indented(s, 2),
    )
    print_indented("Entraînement terminé", 1)
    print_indented(f"Perte finale = {loss_history[-1]:.2f}", 2)
    print_new_line()

    plot_learning_curve(
        loss_history,
        title="Courbe d'apprentissage du modèle v1",
        context=f"taille des embeddings = {embedding_dim}, {epochs} époques, lr={lr}",
        filename=filename,
    )

    print_indented("Prédictions du modèle entraîné:", 1)

    for c in ["c", "s", "t"]:
        tops = char_model.top_next_tokens(tokenizer.encode(c, add_special=False)[0])
        next_tokens = [
            (tokenizer.decode([i], keep_special=True), prob) for i, prob in tops
        ]
        print_indented(f"Après '{c}': {next_tokens}", 2)
    print_new_line()

    print_generations(char_model, tokenizer)


def cmd_v2(
    *,
    embedding_dim: int,
    block_size: int,
    layer_size: int,
    lr: float,
    epochs: int,
    filename: str,
) -> None:
    print_title("Entraîner et tester le modèle v2")
    print_indented("Un modèle avec une fenêtre de contexte", 1)
    print_new_line()

    tokenizer = create_tokenizer(MOTS)

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    context_char_model = ContextCharacterModel(
        len(tokenizer.vocab), embedding_dim, block_size, layer_size
    )
    print_parameters(context_char_model)

    print_indented("Préparation des données d'entraînement", 1)
    print_indented(f"block_size = {block_size}", 2)

    # Cette fois, chaque exemple est une liste de block_size tokens
    # et les cibles sont les tokens qui viennent après
    xs, ys = to_train_data(MOTS, block_size, tokenizer, BOS_ID)
    print_examples(xs, ys, tokenizer)
    print_new_line()

    print_indented("Entraînement ...", 1)
    loss_history = train(
        context_char_model,
        torch.tensor(xs),
        torch.tensor(ys),
        lr=lr,
        epochs=epochs,
        logger=lambda s: print_indented(s, 2),
    )
    print_indented("Entraînement terminé", 1)
    print_indented(f"Perte finale = {loss_history[-1]:.2f}", 2)
    print_new_line()

    plot_learning_curve(
        loss_history,
        title="Courbe d'apprentissage du modèle v2",
        context=f"embeddings = {embedding_dim}, contexte = {block_size}, "
        f"{epochs} époques, lr={lr}",
        filename=filename,
    )

    print_generations(context_char_model, tokenizer)


def cmd_v3(
    *,
    embedding_dim: int,
    block_size: int,
    lr: float,
    epochs: int,
    filename: str,
) -> None:
    print_title("Entraîner et tester le modèle v3")
    print_indented("Un modèle avec calcul d'attention", 1)
    print_new_line()

    tokenizer = create_tokenizer(MOTS)

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    attn_char_model = AttentionCharacterModel(
        len(tokenizer.vocab), embedding_dim, block_size
    )
    print_parameters(attn_char_model)

    print_indented("Préparation des données d'entraînement", 1)
    print_indented(f"block_size = {block_size}", 2)

    xs, ys = to_train_data(MOTS, block_size, tokenizer, BOS_ID)
    print_new_line()

    print_indented("Entraînement ...", 1)
    loss_history = train(
        attn_char_model,
        torch.tensor(xs),
        torch.tensor(ys),
        lr=lr,
        epochs=epochs,
        logger=lambda s: print_indented(s, 2),
    )
    print_indented("Entraînement terminé", 1)
    print_indented(f"Perte finale = {loss_history[-1]:.2f}", 2)
    print_new_line()

    plot_learning_curve(
        loss_history,
        title="Courbe d'apprentissage du modèle v3",
        context=f"embeddings = {embedding_dim}, contexte = {block_size}, "
        f"{epochs} époques, lr={lr}",
        filename=filename,
    )

    print_generations(attn_char_model, tokenizer)


def cmd_v4(
    *,
    embedding_dim: int,
    block_size: int,
    n_head: int,
    lr: float,
    epochs: int,
    filename: str,
) -> None:
    print_title("Entraîner et tester le modèle v4")
    print_indented("Un modèle avec plusieurs têtes d'attention", 1)
    print_new_line()

    print_indented("Dimensionnement du modèle", 1)
    print_indented(f"embedding_dim = {embedding_dim}", 2)
    print_indented(f"n_head = {n_head}", 2)
    print_indented(f"block_size = {block_size}", 2)
    print_new_line()

    tokenizer = create_tokenizer(MOTS)

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    multihead_char_model = MultiHeadAttentionCharacterModel(
        len(tokenizer.vocab), embedding_dim, block_size, n_head
    )
    print_parameters(multihead_char_model)

    print_indented("Préparation des données d'entraînement", 1)

    xs, ys = to_train_data(MOTS, block_size, tokenizer, BOS_ID)
    print_new_line()

    print_indented("Entraînement ...", 1)
    loss_history = train(
        multihead_char_model,
        torch.tensor(xs),
        torch.tensor(ys),
        lr=lr,
        epochs=epochs,
        logger=lambda s: print_indented(s, 2),
    )
    print_indented("Entraînement terminé", 1)
    print_indented(f"Perte finale = {loss_history[-1]:.2f}", 2)
    print_new_line()

    plot_learning_curve(
        loss_history,
        title="Courbe d'apprentissage du modèle v4",
        context=f"embeddings = {embedding_dim}, contexte = {block_size}, "
        f"têtes = {n_head}, {epochs} époques, lr={lr}",
        filename=filename,
    )

    print_generations(multihead_char_model, tokenizer)


def cmd_v5(
    *,
    embedding_dim: int,
    block_size: int,
    n_head: int,
    with_mask: bool,
    lr: float,
    batch_size: int,
    steps: int,
    filename: str,
) -> None:
    print_title("Entraîner et tester le modèle v5")
    print_indented("Calcul des pertes sur toutes les positions", 1)
    print_new_line()

    print_indented("Dimensionnement du modèle", 1)
    print_indented(f"embedding_dim = {embedding_dim}", 2)
    print_indented(f"n_head = {n_head}", 2)
    print_indented(f"block_size = {block_size}", 2)
    print_indented(f"masque causal = {'oui' if with_mask else 'non'}", 2)
    print_new_line()

    if not with_mask:
        print_indented(
            "Attention : sans masque causal, chaque position voit la réponse "
            "et la perte n'a plus de sens.",
            1,
        )
        print_new_line()

    tokenizer = create_tokenizer(MOTS)

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    stream_multihead_char_model = StreamMultiHeadAttentionCharacterModel(
        len(tokenizer.vocab), embedding_dim, block_size, n_head, with_mask
    )
    print_parameters(stream_multihead_char_model)

    print_indented("Préparation des données d'entraînement (flux)", 1)

    stream = to_stream(MOTS, tokenizer)
    print_indented(f"Taille du flux = {len(stream)} tokens", 2)

    # La perte porte maintenant sur toutes les positions : le plancher se
    # calcule donc sur les paires vues à chaque position d'une fenêtre
    contexts, targets = stream_pairs(stream, block_size)
    floor = loss_floor(contexts, targets)
    print_indented(f"Plancher = {floor:.2f}", 2)

    print_new_line()

    print_indented("Entraînement ...", 1)
    loss_history = train_stream(
        stream_multihead_char_model,
        stream,
        len(tokenizer.vocab),
        block_size,
        lr=lr,
        batch_size=batch_size,
        steps=steps,
        logger=lambda s: print_indented(s, 2),
    )
    print_indented("Entraînement terminé", 1)
    print_indented(f"Perte finale = {loss_history[-1]:.2f} (plancher = {floor:.2f})", 2)
    print_new_line()

    plot_learning_curve(
        loss_history,
        title="Courbe d'apprentissage du modèle v5",
        context=f"embeddings = {embedding_dim}, contexte = {block_size}, "
        f"têtes = {n_head}, masque = {'oui' if with_mask else 'non'}, "
        f"{steps} étapes, lr={lr}",
        filename=filename,
        xlabel="Étape",
    )

    print_generations(stream_multihead_char_model, tokenizer)


def cmd_v6(
    *,
    embedding_dim: int,
    block_size: int,
    n_head: int,
    n_layers: int,
    lr: float,
    batch_size: int,
    steps: int,
    eval_every: int,
    max_tokens: int,
    filename: str,
) -> None:
    print_title("Entraîner et tester le modèle v6")
    print_indented("Un vrai corpus de phrases françaises", 1)
    print_new_line()

    print_indented("Téléchargement du corpus", 1)
    downloaded = download_cefr()
    if downloaded:
        print_indented(f"Corpus absent : téléchargement de {', '.join(downloaded)}", 2)
    else:
        print_indented("Corpus déjà présent sur le disque", 2)
    print_new_line()

    print_indented("Chargement du corpus", 1)
    phrases_train, phrases_val, phrases_test = load_cefr()
    print_indented(
        f"train : {len(phrases_train)} phrases, "
        f"{sum(len(phrase) for phrase in phrases_train)} caractères",
        2,
    )
    print_indented(
        f"val   : {len(phrases_val)} phrases, "
        f"{sum(len(phrase) for phrase in phrases_val)} caractères",
        2,
    )
    print_indented(
        f"test  : {len(phrases_test)} phrases (gardées de côté pour la fin)", 2
    )
    print_new_line()

    # Le tokenizer est entraîné sur le jeu d'entraînement uniquement : la
    # validation doit rester inconnue du modèle
    tokenizer = create_tokenizer(phrases_train, label="sur train uniquement")

    print_indented("Dimensionnement du modèle", 1)
    print_indented(f"embedding_dim = {embedding_dim}", 2)
    print_indented(f"n_head = {n_head}", 2)
    print_indented(f"block_size = {block_size}", 2)
    print_indented(f"n_layers = {n_layers}", 2)
    print_new_line()

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    microGPT = MicroGPTModel(
        len(tokenizer.vocab), embedding_dim, block_size, n_head, n_layers
    )
    print_parameters(microGPT)

    print_indented("Préparation des données d'entraînement", 1)

    train_stream_data = to_stream(phrases_train, tokenizer)
    val_stream_data = to_stream(phrases_val, tokenizer)
    windows_per_epoch = len(train_stream_data) - block_size
    epochs = steps * batch_size / windows_per_epoch

    print_indented(f"Flux d'entraînement = {len(train_stream_data)} tokens", 2)
    print_indented(f"Flux de validation = {len(val_stream_data)} tokens", 2)
    print_indented(
        f"{steps} étapes de {batch_size} fenêtres = {epochs:.2f} époque(s), "
        f"une époque valant {windows_per_epoch} fenêtres",
        2,
    )
    print_new_line()

    print_indented("Entraînement ...", 1)
    train_loss_history, val_loss_history, eval_steps = train_stream_with_validation(
        microGPT,
        train_stream_data,
        val_stream_data,
        len(tokenizer.vocab),
        block_size,
        lr=lr,
        batch_size=batch_size,
        steps=steps,
        eval_every=eval_every,
        logger=lambda s: print_indented(s, 2),
    )
    print_indented("Entraînement terminé", 1)
    print_indented(f"Perte d'entraînement = {train_loss_history[-1]:.2f}", 2)
    print_indented(f"Perte de validation = {val_loss_history[-1]:.2f}", 2)
    print_new_line()

    plot_learning_curve(
        train_loss_history,
        title="Courbe d'apprentissage du modèle v6",
        context=f"embeddings = {embedding_dim}, contexte = {block_size}, "
        f"têtes = {n_head}, couches = {n_layers}, {steps} étapes, lr={lr}",
        filename=filename,
        xlabel="Étape",
        val_loss_history=val_loss_history,
        val_steps=eval_steps,
    )

    print_generations(microGPT, tokenizer, max_tokens=max_tokens)


def cmd_v2_v3(*, lr: float, epochs: int) -> None:
    print_title("Comparaison des modèles v2 et v3")
    print_indented("Ce que l'attention change", 1)
    print_new_line()

    tokenizer = create_tokenizer(MOTS)

    # Résultats de l'entraînement des modèles
    results: list[tuple[int, int, str, int, float, float]] = []

    print_indented("Entraînement des modèles ...", 1)

    for block_size in [3, 5]:
        xs, ys = to_train_data(MOTS, block_size, tokenizer, BOS_ID)
        x = torch.tensor(xs)
        y = torch.tensor(ys)
        floor = loss_floor(xs, ys)

        for embedding_dim in [8, 32]:
            print_indented(f"block_size={block_size}, embedding_dim={embedding_dim}", 2)

            # On pose la graine avant chaque modèle pour qu'une configuration
            # ne dépende pas de celles qui ont été entraînées avant elle
            torch.manual_seed(0)
            ctx_char_model = ContextCharacterModel(
                len(tokenizer.vocab), embedding_dim, block_size, 64
            )
            ctx_loss_history = train(ctx_char_model, x, y, lr=lr, epochs=epochs)

            torch.manual_seed(0)
            attn_char_model = AttentionCharacterModel(
                len(tokenizer.vocab), embedding_dim, block_size
            )
            attn_loss_history = train(attn_char_model, x, y, lr=lr, epochs=epochs)

            # La v2 puis la v3, côte à côte pour la même configuration
            results.append(
                (
                    block_size,
                    embedding_dim,
                    "v2",
                    sum(p.numel() for p in ctx_char_model.parameters()),
                    ctx_loss_history[-1],
                    floor,
                )
            )
            results.append(
                (
                    block_size,
                    embedding_dim,
                    "v3",
                    sum(p.numel() for p in attn_char_model.parameters()),
                    attn_loss_history[-1],
                    floor,
                )
            )

    # Colonnes de largeur fixe pour que le tableau reste aligné
    print_new_line()
    header = (
        f"{'Contexte':<9}{'Emb.':<7}{'Modèle':<8}"
        f"{'Paramètres':>11}{'Perte':>9}{'Plancher':>10}"
    )
    print_indented(header, 1)
    print_indented("-" * len(header), 1)

    for block_size, embedding_dim, name, n_parameters, loss, floor in results:
        print_indented(
            f"{'b=' + str(block_size):<9}{'d=' + str(embedding_dim):<7}"
            f"{name:<8}{n_parameters:>11}{loss:>9.3f}{floor:>10.3f}",
            1,
        )

    print_new_line()
    print_indented("(b = block_size, d = embedding_dim)", 1)
    print_indented("La v2 a une couche cachée de 64 neurones, la v3 n'en a pas.", 1)
    print_indented(
        "Le plancher est la perte d'un modèle qui prédirait exactement la", 1
    )
    print_indented(
        "distribution observée dans le corpus : on ne peut pas faire mieux.", 1
    )
