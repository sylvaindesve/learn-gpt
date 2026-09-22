"""Point d'entrée unique vers matplotlib, configuré pour un rendu sans écran.

``matplotlib.use("Agg")`` doit être appelé avant le premier import de
``pyplot``. En centralisant les deux ici, les modules appelants n'ont plus à se
soucier de cet ordre : ils importent simplement ``plt`` depuis ce module.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

__all__ = ["plt", "save_figure", "set_title"]


# Enregistre la figure courante, en créant les dossiers parents au besoin
def save_figure(path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # `bbox_inches="tight"` adapte le cadrage à ce qui est réellement dessiné
    # (titre, légende, libellés) : sans cela, tout ce qui dépasse la figure
    # est rogné silencieusement.
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


# Affiche le titre du graphique et, juste en dessous en plus petit, son
# contexte (hyperparamètres de l'entraînement, résultat final, ...).
# C'est ce qui permet de garder un titre court plutôt qu'une longue ligne
# qui déborde de la figure.
def set_title(title: str, context: str = "") -> None:
    # On réserve de la place au-dessus du graphique quand il y a un contexte
    plt.title(title, pad=22 if context else 6)

    if not context:
        return

    # Le contexte est centré sur la zone de tracé et non sur la figure :
    # les marges gauche et droite ne sont pas toujours symétriques.
    ax = plt.gca()
    ax.text(
        0.5,
        1.015,
        context,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.4",
    )


# Trace et enregistre les courbes d'apprentissage (entraînement et, si elle est
# fournie, validation).
# La validation n'est pas mesurée à chaque étape : `val_steps` donne donc
# l'abscisse de chaque point de validation.
def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    *,
    title: str,
    context: str,
    filepath: Path,
    xlabel: str = "Époque",
    ylabel: str = "Perte",
    val_steps: list[int] | None = None,
) -> None:
    plt.figure()
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Perte sur les données d'entraînement",
    )

    if val_losses:
        plt.plot(
            val_steps if val_steps else range(1, len(val_losses) + 1),
            val_losses,
            label="Perte sur les données de validation",
        )
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    set_title(title, context)
    plt.grid(True)

    save_figure(filepath)
