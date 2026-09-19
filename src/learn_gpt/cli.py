from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from learn_gpt import __version__
from learn_gpt.linear.cmd import cmd_data, cmd_train
from learn_gpt.neuron.cmd import cmd_neuron, cmd_parabola

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
    _add_training_arguments(linear_train, lr_default=0.01, epochs_default=300)
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
    _add_training_arguments(neuron_parabola, lr_default=0.01, epochs_default=3000)
    neuron_parabola.set_defaults(handler=cmd_neuron_parabola)

    return parser


def _help_handler(target: argparse.ArgumentParser) -> Handler:
    # Retourne un handler qui affiche l'aide du parseur donné
    def handler(_args: argparse.Namespace) -> int:
        target.print_help()
        return 0

    return handler


# Options communes aux commandes d'entraînement. `%(default)s` fait
# afficher la valeur par défaut par argparse, sans la répéter dans le texte
def _add_training_arguments(
    parser: argparse.ArgumentParser, lr_default: float = 0.01, epochs_default=3000
) -> None:
    parser.add_argument(
        "--lr",
        type=float,
        default=lr_default,
        metavar="TAUX",
        help="taux d'apprentissage (défaut : %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs_default,
        metavar="N",
        help="nombre d'époques (défaut : %(default)s)",
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
