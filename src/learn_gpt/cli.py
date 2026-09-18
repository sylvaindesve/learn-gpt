from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from learn_gpt import __version__
from learn_gpt.linear.cmd import cmd_data

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

    return parser


def _help_handler(target: argparse.ArgumentParser) -> Handler:
    # Retourne un handler qui affiche l'aide du parseur donné
    def handler(_args: argparse.Namespace) -> int:
        target.print_help()
        return 0

    return handler


def cmd_linear_data(_args: argparse.Namespace) -> int:
    cmd_data()
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
