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

__all__ = ["plt", "save_figure"]


# Enregistre la figure courante, en créant les dossiers parents au besoin
def save_figure(path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)


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
    plt.title(
        f"RMSE finale: {val_loss_history[-1]:.2f} {rmse_unit} "
        f"(entraînement: {train_loss_history[-1]:.2f} {rmse_unit} )"
    )
    plt.grid(True)

    save_figure(filepath, dpi=150)
