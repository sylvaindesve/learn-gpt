from pathlib import Path

import torch

from learn_gpt.commons.data import ensure_file, extract_data, read_csv, split
from learn_gpt.commons.plotting import plt, save_figure, set_title, trace_learning_curve
from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title
from learn_gpt.linear.train import normalize
from learn_gpt.neuron.mlp import MultiLayerPerceptron, train as train_mlp
from learn_gpt.neuron.neuron import create_network_parameters, network, neuron, train

OUTPUT_DIR = Path.cwd() / "output" / "neuron"

DATASET_URL = (
    "https://www.data.gouv.fr/api/1/datasets/r/bc42c2e3-d24c-4499-a966-d35656c6cfc1"
)
DATA_PATH = Path.cwd() / "data" / "co2" / "fic_etiq_edition_40-mars-2015.csv"


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
    set_title("Un neurone : une courbe en S contrôlée par (w, b)")
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
    set_title("Courbe d'apprentissage", f"{epochs} époques, lr={lr}")
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
    set_title("Prédiction du réseau après apprentissage", f"{epochs} époques, lr={lr}")
    save_figure(OUTPUT_DIR / "parabola.png")
    print_indented(f"Courbe apprise créée sous {OUTPUT_DIR / 'parabola.png'}", 2)


def cmd_co2(
    *,
    layer_size: int,
    n_layers: int,
    batch_size: int,
    adam: bool,
    lr: float,
    weight_decay: float,
    dropout: float,
    epochs: int,
    filename: str,
) -> None:
    print_title(
        "Réseau de neurones : entraînement d'un réseau de neurones "
        "sur les données de CO₂"
    )
    target, features = _load_data()

    print_indented("Préparation des données pour l'entraînement", 1)

    # On passe en tenseurs

    # La cible
    y_raw = torch.tensor(target).reshape(-1, 1)
    y, _, sigma = normalize(y_raw)
    print_indented(f"Forme cible: {tuple(y.shape)}", 2)

    # Les features numériques qu'il faut normaliser
    x_num_raw = torch.tensor(features[0:3]).t()
    x_num, _, _ = normalize(x_num_raw)

    # Les features binaires
    x_bin = torch.tensor(features[3]).reshape(-1, 1)

    # Toutes les features
    x = torch.cat([x_num, x_bin], dim=1)
    print_indented(f"Forme features: {tuple(x.shape)}", 2)

    # Jeu d'entraînement et jeu de test
    x_train, y_train, x_val, y_val = split(x, y)

    print_indented(
        f"Séparation des données: {len(x_train)} train / {len(x_val)} val", 2
    )
    print_new_line()

    print_indented("Instanciation du modèle", 1)
    print_indented(f"Nombre de couches: {n_layers}", 2)
    print_indented(f"Nombre de neurones dans une couche: {layer_size}", 2)
    co2_model = MultiLayerPerceptron(x.shape[1], layer_size, n_layers, dropout)
    print_indented(
        f"Nombre de paramètres: {sum(p.numel() for p in co2_model.parameters())}", 2
    )
    print_new_line()

    print_indented("Paramètres d'entraînement:", 1)
    print_indented(f"Optimiseur: {'Adam' if adam else 'SGD'}", 2)
    print_indented(f"Taille des lots: {batch_size}", 2)
    print_indented(f"Taux d'apprentissage: {lr}", 2)
    print_indented(f"Weight decay: {weight_decay}", 2)
    print_indented(f"Dropout: {dropout}", 2)
    print_indented(f"Epoques: {epochs}", 2)
    print_new_line()

    print_indented("Entraînement ...", 1)
    train_loss_history, val_loss_history, _ = train_mlp(
        co2_model,
        x_train,
        y_train,
        x_val,
        y_val,
        sigma=sigma,
        adam=adam,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        logger=lambda s: print_indented(s, 2),
        log_every=5,
    )
    print_indented("Entraînement terminé", 1)
    print_indented(
        f"Perte: initiale ≈ {val_loss_history[0]:.1f} g/km, "
        f"finale ≈ {val_loss_history[-1]:.1f} g/km",
        1,
    )
    print_new_line()

    print_indented("Création de la visualisation de la courbe d'apprentissage", 1)
    trace_learning_curve(
        train_loss_history, val_loss_history, OUTPUT_DIR / filename, rmse_unit="g/km"
    )
    print_indented(f"Courbe d'apprentissage créée sous {OUTPUT_DIR / filename}", 2)


def _load_data() -> tuple[list[float], list[list[float]]]:
    print_indented(f"Chargement du fichier {DATA_PATH}", 1)

    if ensure_file(DATA_PATH, DATASET_URL):
        print_indented("Corpus absent : téléchargement depuis data.gouv.fr", 2)

    data = read_csv(DATA_PATH, encoding="latin-1", delimiter=";")
    print_indented("Données chargées", 2)
    print_indented(f"Nombre de lignes: {len(data['co2_mixte'])}", 2)
    print_new_line()

    # Ajout colonne `is_diesel` basée sur la colonne `energ`
    data["is_diesel"] = ["1.0" if e == "GO" else "0.0" for e in data["energ"]]

    print_indented(
        "Extraction des émissions, des puissances, de la masse et du type de carburant",
        1,
    )
    target, features, skipped = extract_data(
        data, "co2_mixte", ["puiss_max", "puiss_admin", "masse_ordma_min", "is_diesel"]
    )
    print_indented(f"Nombre de lignes chargées: {len(target)}", 2)
    print_indented(f"Nombre de lignes incomplètes ignorées: {skipped}", 2)
    print_new_line()

    return target, features
