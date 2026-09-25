from math import log
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from learn_gpt.commons.plotting import plot_learning_curves
from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title
from learn_gpt.gpt.checkpoint import load_model, save_model
from learn_gpt.gpt.data import (
    download_cefr,
    load_cefr,
    to_stream,
    token_stats,
    tokenizer_path,
)
from learn_gpt.gpt.model import MicroGPTModel
from learn_gpt.gpt.tokenizer import BOS_ID, EOS_ID, BPETokenizer
from learn_gpt.gpt.train import train_stream_with_validation

OUTPUT_DIR = Path.cwd() / "output" / "gpt"


# Crée le tokenizer BPE du corpus, ou le recharge s'il a déjà été appris.
# Apprendre les fusions prend des minutes sur un vrai corpus : on ne le refait
# pas à chaque exécution, on garde le résultat sur le disque.
def create_tokenizer(corpus: list[str], vocab_size: int) -> BPETokenizer:
    path = tokenizer_path(vocab_size)

    print_indented("Tokenizer BPE", 1)

    if path.is_file():
        print_indented(f"Tokenizer rechargé depuis {path}", 2)
        tokenizer = BPETokenizer.load(path)
    else:
        print_indented(f"Apprentissage de {vocab_size} tokens ...", 2)
        tokenizer = BPETokenizer.train(
            "".join(corpus),
            vocab_size,
            logger=lambda s: print_indented(s, 3),
        )
        tokenizer.save(path)
        print_indented(f"Tokenizer sauvegardé sous {path}", 2)

    print_indented(f"vocab_size = {len(tokenizer.vocab)}", 2)
    print_indented(f"{len(tokenizer.merges)} fusions apprises", 2)

    return tokenizer


# Affiche le nombre de paramètres du modèle
def print_parameters(model: nn.Module) -> None:
    n_parameters = sum(p.numel() for p in model.parameters())
    print_indented(f"Nombre de paramètres = {n_parameters}", 2)
    print_new_line()


# La perte par token ne suffit pas à comparer deux tokenizers : elle dépend de
# la taille du vocabulaire. La ramener en bits par caractère rend les chiffres
# comparables d'un tokenizer à l'autre.
def bits_per_character(loss: float, characters_per_token: float) -> float:
    return loss / (characters_per_token * log(2))


# Met une durée en secondes sous une forme lisible
def format_duration(seconds: float) -> str:
    minutes, secondes = divmod(round(seconds), 60)
    if minutes:
        return f"{minutes} min {secondes} s"
    return f"{secondes} s"


# Trace la courbe d'apprentissage (entraînement et validation) et l'enregistre
def plot_learning_curve(
    train_loss_history: list[float],
    val_loss_history: list[float],
    *,
    title: str,
    context: str,
    filename: str,
    val_steps: list[int],
) -> None:
    print_indented("Création de la visualisation de la courbe d'apprentissage", 1)
    filepath = OUTPUT_DIR / filename
    plot_learning_curves(
        train_loss_history,
        val_loss_history,
        title=title,
        context=context,
        filepath=filepath,
        xlabel="Étape",
        ylabel="Entropie croisée",
        val_steps=val_steps,
    )
    print_indented(f"Courbe d'apprentissage créée sous {filepath}", 2)
    print_new_line()


# Génère et affiche quelques séquences, à plusieurs températures
def print_generations(
    model: MicroGPTModel,
    tokenizer: BPETokenizer,
    *,
    temperatures: tuple[float, ...] = (0.8, 0.4),
    n_samples: int = 10,
    max_tokens: int = 100,
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


def cmd_gpt(
    *,
    vocab_size: int,
    embedding_dim: int,
    block_size: int,
    n_head: int,
    n_layers: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    steps: int,
    eval_every: int,
    max_tokens: int,
    model_filename: str,
    filename: str,
) -> None:
    print_title("Entraîner et tester le modèle GPT")
    print_indented("Un GPT avec un tokenizer BPE", 1)
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

    # Le tokenizer est appris sur le jeu d'entraînement uniquement : la
    # validation doit rester inconnue du modèle, tokenizer compris
    tokenizer = create_tokenizer(phrases_train, vocab_size)

    train_ratio, train_unknown = token_stats(phrases_train, tokenizer)
    val_ratio, val_unknown = token_stats(phrases_val, tokenizer)
    print_indented(
        f"train : {train_ratio:.2f} caractères/token, "
        f"{100 * train_unknown:.2f} % d'inconnus",
        2,
    )
    print_indented(
        f"val   : {val_ratio:.2f} caractères/token, "
        f"{100 * val_unknown:.2f} % d'inconnus",
        2,
    )
    print_new_line()

    print_indented("Dimensionnement du modèle", 1)
    print_indented(f"embedding_dim = {embedding_dim}", 2)
    print_indented(f"n_head = {n_head}", 2)
    print_indented(f"block_size = {block_size}", 2)
    print_indented(f"n_layers = {n_layers}", 2)
    print_new_line()

    print_indented("Instanciation du modèle", 1)
    torch.manual_seed(0)
    model = MicroGPTModel(
        len(tokenizer.vocab),
        embedding_dim,
        block_size,
        n_head,
        n_layers,
        dropout=dropout,
    )
    print_parameters(model)

    print_indented("Préparation des données d'entraînement", 1)

    train_data = to_stream(phrases_train, tokenizer)
    val_data = to_stream(phrases_val, tokenizer)
    windows_per_epoch = len(train_data) - block_size
    epochs = steps * batch_size / windows_per_epoch

    print_indented(f"Flux d'entraînement = {len(train_data)} tokens", 2)
    print_indented(f"Flux de validation = {len(val_data)} tokens", 2)
    print_indented(
        f"{steps} étapes de {batch_size} fenêtres = {epochs:.2f} époque(s), "
        f"une époque valant {windows_per_epoch} fenêtres",
        2,
    )
    print_new_line()

    print_indented("Entraînement ...", 1)
    start = perf_counter()
    train_loss_history, val_loss_history, eval_steps = train_stream_with_validation(
        model,
        train_data,
        val_data,
        len(tokenizer.vocab),
        block_size,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        steps=steps,
        eval_every=eval_every,
        logger=lambda s: print_indented(s, 2),
    )
    duration = perf_counter() - start

    print_indented("Entraînement terminé", 1)
    print_indented(f"Durée de l'entraînement = {format_duration(duration)}", 2)
    print_indented(f"Perte d'entraînement = {train_loss_history[-1]:.2f}", 2)
    print_indented(
        f"Perte de validation = {val_loss_history[-1]:.2f}, "
        f"soit {bits_per_character(val_loss_history[-1], val_ratio):.2f} "
        f"bits par caractère",
        2,
    )
    print_new_line()

    plot_learning_curve(
        train_loss_history,
        val_loss_history,
        title="Courbe d'apprentissage du modèle GPT",
        context=f"embeddings = {embedding_dim}, contexte = {block_size}, "
        f"têtes = {n_head}, couches = {n_layers}, {steps} étapes, lr={lr}",
        filename=filename,
        val_steps=eval_steps,
    )

    print_generations(model, tokenizer, max_tokens=max_tokens)

    print_indented("Sauvegarde du modèle", 1)
    filepath = OUTPUT_DIR / model_filename
    save_model(
        filepath,
        model,
        tokenizer,
        embedding_dim=embedding_dim,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layers,
    )
    size_ko = filepath.stat().st_size / 1024
    print_indented(f"Modèle sauvegardé sous {filepath} ({size_ko:.0f} Ko)", 2)
    print_indented(
        f"Pour générer à nouveau : uv run learn-gpt gpt-gen --model {model_filename}",
        2,
    )


def cmd_gpt_gen(
    *,
    model_filename: str,
    max_tokens: int,
    temperature: float,
) -> None:
    print_title("Générer avec un modèle GPT sauvegardé")
    print_new_line()

    print_indented("Rechargement du modèle", 1)
    filepath = OUTPUT_DIR / model_filename

    if not filepath.is_file():
        raise SystemExit(
            f"Aucun modèle sauvegardé sous {filepath} : "
            "lancez d'abord `uv run learn-gpt gpt`"
        )

    model, tokenizer = load_model(filepath)
    print_indented(f"Modèle rechargé depuis {filepath}", 2)
    print_parameters(model)

    print_generations(
        model,
        tokenizer,
        temperatures=(temperature,),
        max_tokens=max_tokens,
    )
