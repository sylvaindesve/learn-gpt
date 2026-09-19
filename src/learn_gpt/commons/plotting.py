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
