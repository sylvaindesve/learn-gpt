from pathlib import Path

import torch

from learn_gpt.commons.data import ensure_file, extract_data, read_csv
from learn_gpt.commons.plotting import plt, save_figure
from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title
from learn_gpt.linear.train import normalize, train

DATASET_URL = (
    "https://www.data.gouv.fr/api/1/datasets/r/bc42c2e3-d24c-4499-a966-d35656c6cfc1"
)
DATA_PATH = Path.cwd() / "data" / "co2" / "fic_etiq_edition_40-mars-2015.csv"

OUTPUT_DIR = Path.cwd() / "output" / "linear"


# Commande pour visualiser les données
def cmd_data() -> None:
    print_title("Régression linéaire : visualisation des données")
    y_co2, x_conso_mixte = _load_data()

    print_indented("Création de la visualisation graphique", 1)
    plt.figure()
    plt.scatter(x_conso_mixte, y_co2, s=1, alpha=0.3, label="données")
    plt.xlabel("Consommation (L/100km)")
    plt.ylabel("Emissions de CO₂ (g/km)")
    plt.title("Emissions de CO₂ en fonction de la consommation")
    save_figure(OUTPUT_DIR / "data.png")
    print_indented(f"Visualisation graphique créée sous {OUTPUT_DIR / 'data.png'}", 2)


# Commande pour entraîner le modèle de régression linéaire
def cmd_train() -> None:
    print_title("Régression linéaire : entraînement")
    y_co2, x_conso_mixte = _load_data()

    # On passe en tenseurs
    y = torch.tensor(y_co2)
    x_raw = torch.tensor(x_conso_mixte)

    # On normalise les caractéristiques
    x, mu, sigma = normalize(x_raw)

    print_indented("Entraînement...", 1)
    w, b, loss_history = train(x, y, epochs=300, logger=lambda s: print_indented(s, 2))
    print_indented("Entraînement terminé", 1)
    print_indented(f"Valeurs finales: w={w.item():.2f}, b={b.item():.2f}", 1)
    print_indented(
        f"Perte: initiale ≈ {loss_history[0]:.1f} g/km, "
        f"finale ≈ {loss_history[-1]:.1f} g/km",
        1,
    )
    print_new_line()

    print_indented("Création de la visualisation de la courbe d'apprentissage", 1)
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoque")
    plt.ylabel("Perte (RMSE)")
    plt.title("Courbe d'apprentissage")
    save_figure(OUTPUT_DIR / "learn.png")
    print_indented(f"Courbe d'apprentissage créée sous {OUTPUT_DIR / 'learn.png'}", 2)
    print_new_line()

    print_indented("Création de la visualisation de la régression linéaire", 1)
    plt.figure()

    # On trace les données
    plt.scatter(x_conso_mixte, y_co2, s=1, alpha=0.3, label="données")

    # Pour tracer la droite on prend le points extrêmes de x
    x_raw_min_max = torch.tensor([x_raw.min(), x_raw.max()])

    # Comme le modèle a été entraîné sur les valeurs normalisées
    # il faut appliquer cette même normalisation avant d'appliquer
    # l'équation
    x_min_max = (x_raw_min_max - mu) / sigma
    y_min_max = x_min_max * w + b

    plt.plot(
        x_raw_min_max.detach().numpy(),
        y_min_max.detach().numpy(),
        color="tab:orange",
        label="droite apprise",
    )
    plt.xlabel("consommation mixte (L/100 km)")
    plt.ylabel("CO₂ (g/km)")
    plt.legend()
    plt.title("Le modèle sur de vraies données")
    save_figure(OUTPUT_DIR / "regression.png")
    print_indented(f"Régression linéaire créée sous {OUTPUT_DIR / 'regression.png'}", 2)


def _load_data() -> tuple[list[float], list[float]]:
    print_indented(f"Chargement du fichier {DATA_PATH}", 1)

    if ensure_file(DATA_PATH, DATASET_URL):
        print_indented("Corpus absent : téléchargement depuis data.gouv.fr", 2)

    data = read_csv(DATA_PATH, encoding="latin-1", delimiter=";")
    print_indented("Données chargées", 2)
    print_indented(f"Nombre de lignes: {len(data['co2_mixte'])}", 2)
    print_new_line()

    print_indented("Extraction des émissions et de la consommation", 1)
    y_co2, [x_conso_mixte], skipped = extract_data(data, "co2_mixte", ["conso_mixte"])
    print_indented(f"Nombre de lignes incomplètes ignorées: {skipped}", 2)
    print_new_line()

    return y_co2, x_conso_mixte
