from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from learn_gpt import __version__
from learn_gpt.linear.cmd import cmd_data, cmd_train
from learn_gpt.neuron.cmd import cmd_co2, cmd_neuron, cmd_parabola

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
    cmd_data()
    return 0


def cmd_linear_train(args: argparse.Namespace) -> int:
    cmd_train(lr=args.lr, epochs=args.epochs)
    return 0


def cmd_neuron_show(_args: argparse.Namespace) -> int:
    cmd_neuron()
    return 0


def cmd_neuron_parabola(args: argparse.Namespace) -> int:
    cmd_parabola(lr=args.lr, epochs=args.epochs)
    return 0


def cmd_neuron_co2(args: argparse.Namespace) -> int:
    cmd_co2(
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
