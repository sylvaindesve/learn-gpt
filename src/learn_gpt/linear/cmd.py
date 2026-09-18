from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")

from learn_gpt.commons.data import ensure_file, extract_data, read_csv
from learn_gpt.commons.print_helpers import print_indented, print_new_line, print_title

DATASET_URL = (
    "https://www.data.gouv.fr/api/1/datasets/r/bc42c2e3-d24c-4499-a966-d35656c6cfc1"
)
DATA_PATH = Path.cwd() / "data" / "co2" / "fic_etiq_edition_40-mars-2015.csv"


# Commande pour visualiser les données
def cmd_data() -> None:
    print_title("Régression linéaire : visualisation des données")
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

    OUTPUT_DIR = Path.cwd() / "output" / "linear"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_indented("Création de la visualisation graphique", 1)
    plt.figure()
    plt.scatter(x_conso_mixte, y_co2, s=1, alpha=0.3, label="données")
    plt.xlabel("Consommation (L/100km)")
    plt.ylabel("Emissions de CO₂ (g/km)")
    plt.title("Emissions de CO₂ en fonction de la consommation")
    plt.savefig(OUTPUT_DIR / "data.png", dpi=150)
    print_indented(f"Visualisation graphique créée sous {OUTPUT_DIR / 'data.png'}", 2)
