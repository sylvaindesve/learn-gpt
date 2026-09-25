from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from learn_gpt import __version__
from learn_gpt.gpt.cmd import (
    cmd_gpt as gpt_cmd_gpt,
    cmd_gpt_gen as gpt_cmd_gpt_gen,
)
from learn_gpt.linear.cmd import (
    cmd_data as linear_cmd_data,
    cmd_train as linear_cmd_train,
)
from learn_gpt.neuron.cmd import (
    cmd_co2 as neuron_cmd_co2,
    cmd_neuron as neuron_cmd_neuron,
    cmd_parabola as neuron_cmd_parabola,
)
from learn_gpt.text.cmd import (
    cmd_loss as text_cmd_loss,
    cmd_show as text_cmd_show,
    cmd_v1 as text_cmd_v1,
    cmd_v2 as text_cmd_v2,
    cmd_v2_v3 as text_cmd_v2_v3,
    cmd_v3 as text_cmd_v3,
    cmd_v4 as text_cmd_v4,
    cmd_v5 as text_cmd_v5,
    cmd_v6 as text_cmd_v6,
    cmd_v6_gen as text_cmd_v6_gen,
)

Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    # Parseur de la commande `learn-gpt`
    parser = argparse.ArgumentParser(
        prog="learn-gpt",
        description="Apprendre le fonctionnement d'un modèle GPT",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Parseur des sous-commandes
    subparsers = parser.add_subparsers(
        title="commandes",
        dest="command",
        metavar="COMMANDE",
    )

    # Sous-commande pour la section sur la régression linéaire
    linear = subparsers.add_parser("linear", help="section régression linéaire")
    linear.set_defaults(handler=_help_handler(linear))
    linear_sub = linear.add_subparsers(
        title="sous-commandes",
        dest="linear_command",
        metavar="SOUS-COMMANDE",
    )

    # Sous-sous-commande pour la visualisaton des données
    linear_data = linear_sub.add_parser("data", help="visualiser les données")
    linear_data.set_defaults(handler=cmd_linear_data)

    # Sous-sous-commande pour l'entraînement du modèle de régression linéaire
    linear_train = linear_sub.add_parser("train", help="entraîner le modèle")
    _add_lr_argument(linear_train, default=0.01)
    _add_epochs_argument(linear_train, default=300)
    linear_train.set_defaults(handler=cmd_linear_train)

    # Sous-commande pour la section sur les réseaux de neurones
    neuron = subparsers.add_parser("neuron", help="section réseau de neurones")
    neuron.set_defaults(handler=_help_handler(neuron))
    neuron_sub = neuron.add_subparsers(
        title="sous-commandes",
        dest="neuron_command",
        metavar="SOUS-COMMANDE",
    )

    # Sous-sous-commande pour voir la sortie d'un neurone
    neuron_show = neuron_sub.add_parser(
        "show", help="visualiser la sortie d'un neurone"
    )
    neuron_show.set_defaults(handler=cmd_neuron_show)

    # Sous-sous-commande pour entraîner un petit réseau sur la parabole x²
    neuron_parabola = neuron_sub.add_parser(
        "parabola", help="entraîner un petit réseau de neurones sur la parabole"
    )
    _add_lr_argument(neuron_parabola, default=0.01)
    _add_epochs_argument(neuron_parabola, default=3000)
    neuron_parabola.set_defaults(handler=cmd_neuron_parabola)

    # Sous-sous-commande pour entraîner un réseau sur les données de CO₂
    neuron_co2 = neuron_sub.add_parser(
        "co2", help="entraîner un réseau sur les données de CO₂"
    )
    _add_layer_size_argument(neuron_co2, default=64)
    _add_layers_argument(neuron_co2, default=1)
    _add_batch_size_argument(neuron_co2, default=1024)
    _add_adam_argument(neuron_co2)
    _add_lr_argument(neuron_co2, default=0.01)
    _add_weight_decay_argument(neuron_co2, default=0.0)
    _add_dropout_argument(neuron_co2, default=0.0)
    _add_epochs_argument(neuron_co2, default=300)
    _add_learning_curve_filename_argument(neuron_co2, default="co2_learn.png")
    neuron_co2.set_defaults(handler=cmd_neuron_co2)

    # Sous-commande pour la section sur le texte
    text = subparsers.add_parser("text", help="section texte")
    text.set_defaults(handler=_help_handler(text))
    text_sub = text.add_subparsers(
        title="sous-commandes",
        dest="text_command",
        metavar="SOUS-COMMANDE",
    )

    # Sous-sous-commande pour voir la représentation de texte avec des nombres
    text_show = text_sub.add_parser("show", help="transformer du texte en nombres")
    text_show.set_defaults(handler=cmd_text_show)

    # Sous-sous-commande pour voir le calcul de perte sur du texte
    text_loss = text_sub.add_parser("loss", help="calculer la perte sur du texte")
    text_loss.set_defaults(handler=cmd_text_loss)

    # Sous-sous-commande pour entraîner un modèle v1 de prédiction du caractère suivant
    text_v1 = text_sub.add_parser("v1", help="entraîner et tester un modèle v1")
    _add_embedding_dim_argument(text_v1, default=8)
    _add_lr_argument(text_v1, default=0.02)
    _add_epochs_argument(text_v1, default=400)
    _add_learning_curve_filename_argument(text_v1, default="v1_learn.png")
    text_v1.set_defaults(handler=cmd_text_v1)

    # Sous-sous-commande pour entraîner un modèle v2 avec fenêtre de contexte
    text_v2 = text_sub.add_parser(
        "v2", help="entraîner et tester un modèle v2 (contexte)"
    )
    _add_embedding_dim_argument(text_v2, default=8)
    _add_block_size_argument(text_v2, default=3)
    _add_layer_size_argument(text_v2, default=64)
    _add_lr_argument(text_v2, default=0.02)
    _add_epochs_argument(text_v2, default=100)
    _add_learning_curve_filename_argument(text_v2, default="v2_learn.png")
    text_v2.set_defaults(handler=cmd_text_v2)

    # Sous-sous-commande pour entraîner un modèle v3 avec calcul d'attention
    text_v3 = text_sub.add_parser(
        "v3", help="entraîner et tester un modèle v3 (attention)"
    )
    _add_embedding_dim_argument(text_v3, default=8)
    _add_block_size_argument(text_v3, default=3)
    _add_lr_argument(text_v3, default=0.01)
    _add_epochs_argument(text_v3, default=600)
    _add_learning_curve_filename_argument(text_v3, default="v3_learn.png")
    text_v3.set_defaults(handler=cmd_text_v3)

    # Sous-sous-commande pour comparer les performances des modèles v2 et v3
    text_v2_v3 = text_sub.add_parser("v2v3", help="comparer les modèles v2 et v3")
    _add_lr_argument(text_v2_v3, default=0.01)
    _add_epochs_argument(text_v2_v3, default=600)
    text_v2_v3.set_defaults(handler=cmd_text_v2_v3)

    # Sous-sous-commande pour entraîner un modèle v4 avec d'attention multi-têtes
    text_v4 = text_sub.add_parser(
        "v4", help="entraîner et tester un modèle v4 (multi-têtes)"
    )
    _add_embedding_dim_argument(text_v4, default=8)
    _add_block_size_argument(text_v4, default=3)
    _add_n_head_argument(text_v4, default=4)
    _add_lr_argument(text_v4, default=0.01)
    _add_epochs_argument(text_v4, default=300)
    _add_learning_curve_filename_argument(text_v4, default="v4_learn.png")
    text_v4.set_defaults(handler=cmd_text_v4)

    # Sous-sous-commande pour entraîner un modèle v5 avec perte calculée
    # sur toutes les positions
    text_v5 = text_sub.add_parser(
        "v5", help="entraîner et tester un modèle v5 (perte sur toutes les positions)"
    )
    _add_embedding_dim_argument(text_v5, default=8)
    _add_block_size_argument(text_v5, default=3)
    _add_n_head_argument(text_v5, default=4)
    _add_mask_argument(text_v5)
    _add_lr_argument(text_v5, default=0.01)
    _add_batch_size_argument(text_v5, default=16)
    _add_steps_argument(text_v5, default=3000)
    _add_learning_curve_filename_argument(text_v5, default="v5_learn.png")
    text_v5.set_defaults(handler=cmd_text_v5)

    # Sous-sous-commande pour entraîner un modèle v6 GPT complet
    text_v6 = text_sub.add_parser("v6", help="entraîner et tester un modèle v6 (GPT)")
    _add_embedding_dim_argument(text_v6, default=64)
    _add_block_size_argument(text_v6, default=32)
    _add_n_head_argument(text_v6, default=4)
    _add_layers_argument(text_v6, default=2)
    _add_lr_argument(text_v6, default=0.003)
    _add_batch_size_argument(text_v6, default=32)
    _add_steps_argument(text_v6, default=3000)
    _add_eval_every_argument(text_v6, default=200)
    _add_max_tokens_argument(text_v6, default=100)
    _add_model_argument(text_v6, default="v6_model.pt")
    _add_learning_curve_filename_argument(text_v6, default="v6_learn.png")
    text_v6.set_defaults(handler=cmd_text_v6)

    # Sous-sous-commande pour générer à partir d'un modèle v6 sauvegardé
    text_v6_gen = text_sub.add_parser(
        "v6-gen", help="générer avec un modèle v6 sauvegardé"
    )
    _add_model_argument(text_v6_gen, default="v6_model.pt")
    _add_max_tokens_argument(text_v6_gen, default=100)
    _add_temperature_argument(text_v6_gen, default=0.8)
    text_v6_gen.set_defaults(handler=cmd_text_v6_gen)

    # Commande pour entraîner un GPT avec un tokenizer BPE
    gpt = subparsers.add_parser("gpt", help="entraîner un GPT avec un tokenizer BPE")
    _add_vocab_size_argument(gpt, default=512)
    _add_embedding_dim_argument(gpt, default=64)
    _add_block_size_argument(gpt, default=64)
    _add_n_head_argument(gpt, default=4)
    _add_layers_argument(gpt, default=4)
    _add_lr_argument(gpt, default=0.003)
    _add_weight_decay_argument(gpt, default=0.0)
    _add_dropout_argument(gpt, default=0.0)
    _add_batch_size_argument(gpt, default=32)
    _add_steps_argument(gpt, default=3000)
    _add_eval_every_argument(gpt, default=200)
    _add_max_tokens_argument(gpt, default=100)
    _add_model_argument(gpt, default="gpt_model.pt")
    _add_learning_curve_filename_argument(gpt, default="gpt_learn.png")
    gpt.set_defaults(handler=cmd_gpt)

    # Commande pour générer à partir d'un modèle GPT sauvegardé
    gpt_gen = subparsers.add_parser(
        "gpt-gen", help="générer avec un modèle GPT sauvegardé"
    )
    _add_model_argument(gpt_gen, default="gpt_model.pt")
    _add_max_tokens_argument(gpt_gen, default=100)
    _add_temperature_argument(gpt_gen, default=0.8)
    gpt_gen.set_defaults(handler=cmd_gpt_gen)

    return parser


def _help_handler(target: argparse.ArgumentParser) -> Handler:
    # Retourne un handler qui affiche l'aide du parseur donné
    def handler(_args: argparse.Namespace) -> int:
        target.print_help()
        return 0

    return handler


# Ajoute un argument sur la taille de la couche cachée de neurones
def _add_layer_size_argument(
    parser: argparse.ArgumentParser, default: int = 32
) -> None:
    parser.add_argument(
        "--layersize",
        type=int,
        default=default,
        metavar="N",
        help="nombre de neurones dans la couche cachée (défaut : %(default)s)",
    )


# Ajoute un argument sur la taille du vocabulaire du tokenizer BPE
def _add_vocab_size_argument(
    parser: argparse.ArgumentParser, default: int = 512
) -> None:
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=default,
        metavar="N",
        help="taille du vocabulaire du tokenizer BPE (défaut : %(default)s)",
    )


# Ajoute un argument sur la taille des embeddings
def _add_embedding_dim_argument(
    parser: argparse.ArgumentParser, default: int = 8
) -> None:
    parser.add_argument(
        "--embedding",
        type=int,
        default=default,
        metavar="N",
        help="taille des embeddings (défaut : %(default)s)",
    )


# Contrôle sur --block
def _block_size(value: str) -> int:
    block_size = int(value)
    if block_size < 1:
        raise argparse.ArgumentTypeError("block_size doit être au moins 1")
    return block_size


# Ajoute un argument sur la taille de la fenêtre de contexte
def _add_block_size_argument(parser: argparse.ArgumentParser, default: int = 3) -> None:
    parser.add_argument(
        "--block",
        type=_block_size,
        default=default,
        metavar="N",
        help="taille de la fenêtre de contexte (défaut : %(default)s)",
    )


# Ajoute un argument sur le nombre de têtes d'attention
def _add_n_head_argument(parser: argparse.ArgumentParser, default: int = 4) -> None:
    parser.add_argument(
        "--head",
        type=int,
        default=default,
        metavar="N",
        help=(
            "nombre de têtes d'attention, diviseur de embedding_dim "
            "(défaut : %(default)s)"
        ),
    )


# Ajoute un argument pour désactiver le masque causal
def _add_mask_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-mask",
        action="store_false",
        dest="mask",
        default=True,
        help="désactive le masque causal : le modèle peut alors voir la réponse",
    )


# Ajoute un argument sur le nombre de couches cachées
def _add_layers_argument(parser: argparse.ArgumentParser, default: int = 1) -> None:
    parser.add_argument(
        "--layers",
        type=int,
        default=default,
        metavar="N",
        help="nombre de couches cachées (défaut : %(default)s)",
    )


# Ajoute un argument pour le contrôle du taux d'apprentissage
def _add_lr_argument(parser: argparse.ArgumentParser, default: float = 0.01) -> None:
    parser.add_argument(
        "--lr",
        type=float,
        default=default,
        metavar="TAUX",
        help="taux d'apprentissage (défaut : %(default)s)",
    )


# Ajoute un argument pour le contrôle du nombre d'époques
def _add_epochs_argument(parser: argparse.ArgumentParser, default: int = 3000) -> None:
    parser.add_argument(
        "--epochs",
        type=int,
        default=default,
        metavar="N",
        help="nombre d'époques (défaut : %(default)s)",
    )


# Ajoute un argument pour le contrôle du nombre d'étapes
def _add_steps_argument(parser: argparse.ArgumentParser, default: int = 3000) -> None:
    parser.add_argument(
        "--steps",
        type=int,
        default=default,
        metavar="N",
        help="nombre d'étapes (défaut : %(default)s)",
    )


# Ajoute un argument pour la fréquence d'évaluation sur la validation
def _add_eval_every_argument(
    parser: argparse.ArgumentParser, default: int = 200
) -> None:
    parser.add_argument(
        "--eval-every",
        type=int,
        default=default,
        metavar="N",
        help=(
            "mesure la perte de validation toutes les N étapes (défaut : %(default)s)"
        ),
    )


# Ajoute un argument pour le nombre maximum de tokens générés
def _add_max_tokens_argument(
    parser: argparse.ArgumentParser, default: int = 20
) -> None:
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=default,
        metavar="N",
        help="nombre maximum de tokens générés (défaut : %(default)s)",
    )


# Ajoute un argument pour le fichier de sauvegarde du modèle
def _add_model_argument(
    parser: argparse.ArgumentParser, default: str = "v6_model.pt"
) -> None:
    parser.add_argument(
        "--model",
        type=str,
        default=default,
        metavar="FILENAME",
        help="nom du fichier du modèle (défaut : %(default)s)",
    )


# Ajoute un argument pour la température de génération
def _add_temperature_argument(
    parser: argparse.ArgumentParser, default: float = 0.8
) -> None:
    parser.add_argument(
        "--temperature",
        type=float,
        default=default,
        metavar="T",
        help="température de génération (défaut : %(default)s)",
    )


# Ajoute un argument pour le contrôle de la taille des lots
def _add_batch_size_argument(
    parser: argparse.ArgumentParser, default: int = 1024
) -> None:
    parser.add_argument(
        "--batch",
        type=int,
        default=default,
        metavar="N",
        help="taille des lots (défaut : %(default)s)",
    )


# Ajoute un argument pour le contrôle du weight decay
def _add_weight_decay_argument(
    parser: argparse.ArgumentParser, default: float = 0.0
) -> None:
    parser.add_argument(
        "--wd",
        type=float,
        default=default,
        metavar="TAUX",
        help="weight decay (défaut : %(default)s)",
    )


# Ajoute un argument pour le contrôle du dropout
def _add_dropout_argument(
    parser: argparse.ArgumentParser, default: float = 0.0
) -> None:
    parser.add_argument(
        "--dropout",
        type=float,
        default=default,
        metavar="TAUX",
        help="dropout (défaut : %(default)s)",
    )


# Ajoute un argument pour choisir Adam à la place de la descente de gradient
def _add_adam_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adam",
        action="store_true",
        help=(
            "utilise Adam au lieu de la descente de gradient classique (SGD), "
            "penser à baisser le lr"
        ),
    )


# Ajoute un argument pour indiquer le fichier pour la courbe d'apprentissage
def _add_learning_curve_filename_argument(
    parser: argparse.ArgumentParser, default: str = "curve.png"
) -> None:
    parser.add_argument(
        "--filename",
        type=str,
        default=default,
        metavar="FILENAME",
        help=(
            "nom du fichier pour le tracé de la courbe d'apprentissage "
            "(défaut : %(default)s)"
        ),
    )


def cmd_linear_data(_args: argparse.Namespace) -> int:
    linear_cmd_data()
    return 0


def cmd_linear_train(args: argparse.Namespace) -> int:
    linear_cmd_train(lr=args.lr, epochs=args.epochs)
    return 0


def cmd_neuron_show(_args: argparse.Namespace) -> int:
    neuron_cmd_neuron()
    return 0


def cmd_neuron_parabola(args: argparse.Namespace) -> int:
    neuron_cmd_parabola(lr=args.lr, epochs=args.epochs)
    return 0


def cmd_neuron_co2(args: argparse.Namespace) -> int:
    neuron_cmd_co2(
        layer_size=args.layersize,
        n_layers=args.layers,
        batch_size=args.batch,
        adam=args.adam,
        lr=args.lr,
        weight_decay=args.wd,
        dropout=args.dropout,
        epochs=args.epochs,
        filename=args.filename,
    )
    return 0


def cmd_text_show(_args: argparse.Namespace) -> int:
    text_cmd_show()
    return 0


def cmd_text_loss(_args: argparse.Namespace) -> int:
    text_cmd_loss()
    return 0


def cmd_text_v1(args: argparse.Namespace) -> int:
    text_cmd_v1(
        embedding_dim=args.embedding,
        lr=args.lr,
        epochs=args.epochs,
        filename=args.filename,
    )
    return 0


def cmd_text_v2(args: argparse.Namespace) -> int:
    text_cmd_v2(
        embedding_dim=args.embedding,
        block_size=args.block,
        layer_size=args.layersize,
        lr=args.lr,
        epochs=args.epochs,
        filename=args.filename,
    )
    return 0


def cmd_text_v3(args: argparse.Namespace) -> int:
    text_cmd_v3(
        embedding_dim=args.embedding,
        block_size=args.block,
        lr=args.lr,
        epochs=args.epochs,
        filename=args.filename,
    )
    return 0


def cmd_text_v2_v3(args: argparse.Namespace) -> int:
    text_cmd_v2_v3(lr=args.lr, epochs=args.epochs)
    return 0


def cmd_text_v4(args: argparse.Namespace) -> int:
    text_cmd_v4(
        embedding_dim=args.embedding,
        block_size=args.block,
        n_head=args.head,
        lr=args.lr,
        epochs=args.epochs,
        filename=args.filename,
    )
    return 0


def cmd_text_v5(args: argparse.Namespace) -> int:
    text_cmd_v5(
        embedding_dim=args.embedding,
        block_size=args.block,
        n_head=args.head,
        with_mask=args.mask,
        lr=args.lr,
        batch_size=args.batch,
        steps=args.steps,
        filename=args.filename,
    )
    return 0


def cmd_text_v6(args: argparse.Namespace) -> int:
    text_cmd_v6(
        embedding_dim=args.embedding,
        block_size=args.block,
        n_head=args.head,
        n_layers=args.layers,
        lr=args.lr,
        batch_size=args.batch,
        steps=args.steps,
        eval_every=args.eval_every,
        max_tokens=args.max_tokens,
        model_filename=args.model,
        filename=args.filename,
    )
    return 0


def cmd_text_v6_gen(args: argparse.Namespace) -> int:
    text_cmd_v6_gen(
        model_filename=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    return 0


def cmd_gpt(args: argparse.Namespace) -> int:
    gpt_cmd_gpt(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding,
        block_size=args.block,
        n_head=args.head,
        n_layers=args.layers,
        lr=args.lr,
        weight_decay=args.wd,
        dropout=args.dropout,
        batch_size=args.batch,
        steps=args.steps,
        eval_every=args.eval_every,
        max_tokens=args.max_tokens,
        model_filename=args.model,
        filename=args.filename,
    )
    return 0


def cmd_gpt_gen(args: argparse.Namespace) -> int:
    gpt_cmd_gpt_gen(
        model_filename=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Lance le programme et retourne le code de sortie."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Sans sous-commande, aucune fonction n'a été attachée : on affiche l'aide.
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    return handler(args)
