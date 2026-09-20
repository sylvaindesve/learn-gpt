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


# Trace les courbes d'apprentissage
def trace_learning_curve(
    # Historique de pertes sur les données d'entraînement
    train_loss_history: list[float],
    # Historique de pertes sur les données de validation
    val_loss_history: list[float],
    filepath: Path,
    rmse_unit: str = "",  # Unité de la RMSE
):
    plt.figure()
    plt.plot(
        train_loss_history,
        label="Perte sur les données d'entraînement",
    )
    plt.plot(val_loss_history, label="Perte sur les données de validation")
    plt.xlabel("Époque")
    plt.ylabel(f"RMSE {rmse_unit}")
    plt.legend()
    set_title(
        "Courbe d'apprentissage",
        f"RMSE finale : {val_loss_history[-1]:.2f} {rmse_unit} "
        f"(entraînement : {train_loss_history[-1]:.2f} {rmse_unit})",
    )
    plt.grid(True)

    save_figure(filepath, dpi=150)
