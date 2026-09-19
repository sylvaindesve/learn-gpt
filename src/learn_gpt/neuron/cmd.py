from pathlib import Path

import torch

from learn_gpt.commons.plotting import plt, save_figure
from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title
from learn_gpt.neuron.neuron import create_network_parameters, network, neuron, train

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


def cmd_parabola(*, lr: float, epochs: int) -> None:
    print_title(
        "Réseau de neurones : entraînement d'un petit réseau de neurones "
        "sur la parabole"
    )

    # Valeurs et cible pour la parabole x²
    x = torch.arange(-3.0, 3.0, 0.1)
    y = x**2

    wh, bh, wo, bo = create_network_parameters(6, seed=42)

    print_indented(f"Entraînement ({epochs} époques, lr={lr})...", 1)
    loss_history = train(
        x,
        y,
        wh,
        bh,
        wo,
        bo,
        lr=lr,
        epochs=epochs,
        logger=lambda s: print_indented(s, 2),
    )
    print_indented("Entraînement terminé", 1)
    print_indented(
        f"Perte: initiale ≈ {loss_history[0]:.1f}, finale ≈ {loss_history[-1]:.1f}",
        1,
    )
    print_new_line()

    print_indented("Création de la visualisation de la courbe d'apprentissage", 1)
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoque")
    plt.ylabel("Perte (RMSE)")
    plt.title(f"Courbe d'apprentissage ({epochs} époques, lr={lr})")
    save_figure(OUTPUT_DIR / "learn_simple.png")
    print_indented(
        f"Courbe d'apprentissage créée sous {OUTPUT_DIR / 'learn_simple.png'}", 2
    )
    print_new_line()

    pred = network(x, wh, bh, wo, bo)

    print_indented("Création de la visualisation de la courbe apprise", 1)
    plt.figure()
    plt.scatter(x.numpy(), y.numpy(), s=8, label="parabole y = x²")
    plt.plot(
        x.numpy(),
        pred.detach().numpy(),
        color="tab:orange",
        label="réseau maison",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Prédiction du réseau après apprentissage ({epochs} époques, lr={lr})")
    save_figure(OUTPUT_DIR / "parabola.png")
    print_indented(f"Courbe apprise créée sous {OUTPUT_DIR / 'parabola.png'}", 2)
