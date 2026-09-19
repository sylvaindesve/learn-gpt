from pathlib import Path

import torch

from learn_gpt.commons.plotting import plt, save_figure
from learn_gpt.commons.print_helpers import print_indented, print_title
from learn_gpt.neuron.neuron import neuron

OUTPUT_DIR = Path.cwd() / "output" / "neuron"


# Commande pour visualiser la sortie d'un neurone
def cmd_neuron() -> None:
    print_title("Réseau de neurones : visualisation de la sortie d'un neurone")

    print_indented("Création de la visualisation graphique", 1)

    x = torch.arange(-6.0, 6.0, 0.1)
    plt.plot(
        x.numpy(),
        neuron(x, torch.tensor(0.7), torch.tensor(0.0)).numpy(),
        label="w=0.7, b=0",
    )
    plt.plot(
        x.numpy(),
        neuron(x, torch.tensor(0.7), torch.tensor(2.0)).numpy(),
        label="w=0.7, b=2  (décalé)",
    )
    plt.plot(
        x.numpy(),
        neuron(x, torch.tensor(-1.2), torch.tensor(0.0)).numpy(),
        label="w=-1.2, b=0 (inversé)",
    )
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.legend()
    plt.title("Un neurone : une courbe en S contrôlée par (w, b)")
    save_figure(OUTPUT_DIR / "neuron.png")

    print_indented(f"Visualisation graphique créée sous {OUTPUT_DIR / 'neuron.png'}", 2)
