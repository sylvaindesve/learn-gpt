import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch


# Lit un CSV et retourne ses colonnes sous la forme
# { header1: [values, ...], header2: [values, ...] }
def read_csv(
    dataset_path: Path,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> dict[str, list[str]]:
    with open(dataset_path, encoding=encoding, newline="") as f:
        data = csv.reader(f, delimiter=delimiter)
        header = next(data)

        columns: dict[str, list[str]] = {name: [] for name in header}
        for ligne in data:
            for i, name in enumerate(header):
                columns[name].append(ligne[i].strip())

    return columns


# Extrait d'un dictionnaire issu de `read_csv` la donnée cibles et les données
# de caractéristiques. Ne garde que les lignes où toutes les valeurs peuvent
# être castées en `float`
def extract_data(
    data_dict: dict[str, list[str]],
    target_column_name: str,
    feature_column_names: list[str],
) -> tuple[list[float], list[list[float]], int]:
    target, features, skipped = [], [[] for _ in feature_column_names], 0
    size = len(data_dict[target_column_name])
    for i in range(size):
        try:
            row_target = float(data_dict[target_column_name][i])
            row_features = [float(data_dict[name][i]) for name in feature_column_names]
        except ValueError:
            skipped += 1
            continue
        target.append(row_target)
        for j, value in enumerate(row_features):
            features[j].append(value)
    return target, features, skipped


# Sépare les données en données d'entraînement et données de validation
def split(x: torch.Tensor, y: torch.Tensor, ratio: float = 0.2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    indices = torch.randperm(n, generator=g)
    n_val = int(ratio * n)
    idx_val, idx_train = indices[:n_val], indices[n_val:]
    return x[idx_train], y[idx_train], x[idx_val], y[idx_val]


# Télécharge `url` vers `destination`, en créant les dossiers parents.
# Si la réponse est une archive ZIP, le premier fichier `.csv` qu'elle
# contient est extrait vers `destination`. C'est le cas des ressources
# data.gouv.fr, qui renvoient une archive et non le CSV directement.
# L'écriture passe par un fichier `.part` renommé à la fin : si le
# téléchargement échoue en cours de route, aucun fichier tronqué ne reste
# sur le disque.
def download_file(url: str, destination: Path, *, timeout: float = 60.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")

    try:
        with (
            urllib.request.urlopen(url, timeout=timeout) as response,
            temporary.open("wb") as f,
        ):
            shutil.copyfileobj(response, f)

        if zipfile.is_zipfile(temporary):
            extract_first_csv(temporary, destination)
        else:
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


# Extrait le premier fichier `.csv` de `archive` vers `destination`
def extract_first_csv(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"Aucun fichier .csv dans l'archive {archive}")

        with zf.open(names[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


# Télécharge `url` vers `path` s'il est absent
# Retourne `True` si le fichier a été téléchargé, `False` s'il était déjà là
def ensure_file(path: Path, url: str) -> bool:
    if path.is_file():
        return False

    download_file(url, path)
    return True
