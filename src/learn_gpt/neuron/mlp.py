from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Un réseau de neurones MLP (Multilayer Perceptron) avec plusieurs couches
# cachée de neurones et un mécanisme de dropout pour éteindre certains
# neurones durant l'entraînement.
# Ce n'est fondamentalement pas différent du réseau qu'on a écrit dans
# `neuron.py` mais ici on s'appuie un peu plus sur PyTorch.
#
# Un réseau de neurones 1 → h → 1 tel qu'on l'a définit dans `neuron.py`
# peut s'écrire de la façon suivante avec `nn` :
#
# ```python
# network = nn.Sequential(nn.Linear(1, h), nn.Tanh(), nn.Linear(h, 1))
# ```
class MultiLayerPerceptron(nn.Module):
    def __init__(
        self, n_features: int, layer_size: int, n_layers: int, dropout: float = 0.0
    ):
        super().__init__()
        network = []

        # La première couche cachée de layer_size neurones acceptant en entrée
        # des vecteurs de n_features caractéristiques
        network += [nn.Linear(n_features, layer_size), nn.Tanh()]
        if dropout > 0:
            # On ajoute un dropout pour éteindre certains neurones à l'entraînement
            network.append(nn.Dropout(dropout))

        # On ajoute d'autres couches cachées sur n_layers > 1
        for _ in range(n_layers - 1):
            network += [nn.Linear(layer_size, layer_size), nn.Tanh()]
            if dropout > 0:
                network.append(nn.Dropout(dropout))

        # Et enfin le neurone de sortie
        network += [nn.Linear(layer_size, 1)]

        self.network = nn.Sequential(*network)

    # x représente n vecteurs exemple de n_features caractéristiques
    # Il est de forme (n, n_features)
    # La sortie est de forme (n, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Entraînement du modèle
def train(
    model: nn.Module,  # Le modèle à entraîner
    x_train: torch.Tensor,  # Les données d'entrées pour l'entraînement
    y_train: torch.Tensor,  # Les sorties correspondant aux données d'entraînement
    x_val: torch.Tensor,  # Les données d'entrées pour la validation
    y_val: torch.Tensor,  # Les sorties correspondant aux données de validation
    # Si les données de sortie ont été normalisées en divisant par l'écart-type,
    # on fournit cet écart-type pour l'appliquer à la RMSE enregistrée
    sigma: torch.Tensor | None = None,
    batch_size: int = 1024,  # La taille de lots
    lr: float = 0.01,  # Le taux d'apprentissage
    epochs: int = 300,  # Le nombre d'époques
    weight_decay: float = 0.0,
    adam: bool = False,  # S'il faut utiliser l'optimiseur Adam
    seed: int = 0,
    logger: Callable[[str], None] = lambda _: None,
    log_every: int = 50,
) -> tuple[list[float], list[float], list[float]]:
    torch.manual_seed(seed)

    # On initialise l'optimiseur.
    # Ce dernier va gérer la mise à jour des paramètres

    if adam:
        # Un optimiseur ADAM utilisant un momentum pour adapter sa descente
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Sinon, on choisit la descente de gradient classique (SGD)
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Le loader nous aide à créer des lots aléatoires parmi les données d'entraînement
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )

    batch_loss_history: list[float] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    for epoch in range(epochs):
        for x, y in loader:
            # Remise à zéro des gradients
            opt.zero_grad()

            # Calcul de perte (MSE)
            loss = ((model(x) - y) ** 2).mean()

            # Calcul des gradients
            loss.backward()

            # Mise à jour des paramètres
            opt.step()

            batch_loss_history.append(_denormalize_rmse(loss.item() ** 0.5, sigma))

        # On calcule ensuite la perte sur les données de validation
        # et sur l'ensemble des données d'entraînement mais seulement
        # une fois par époque
        with torch.no_grad():
            # Le mode `eval` désactive les mécanismes d'entraînement
            # comme le dropout
            model.eval()

            # MSE sur l'ensemble de données d'entraînement
            train_loss = ((model(x_train) - y_train) ** 2).mean()

            # MSE sur les données de validation
            val_loss = ((model(x_val) - y_val) ** 2).mean()

            # On remet en mode entraînement
            model.train()

            train_loss_history.append(
                _denormalize_rmse(train_loss.item() ** 0.5, sigma)
            )
            val_loss_history.append(_denormalize_rmse(val_loss.item() ** 0.5, sigma))

        if epoch % log_every == 0:
            logger(
                f"Epoque {epoch + 1}/{epochs}, pertes (RMSE): "
                f"train = {_denormalize_rmse(train_loss.item() ** 0.5, sigma):.2f}, "
                f"eval = {_denormalize_rmse(val_loss.item() ** 0.5, sigma):.2f}"
            )

    # En fin d'entraînement on se remet en mode évaluation
    model.eval()

    return train_loss_history, val_loss_history, batch_loss_history


# Dénormalise la perte en multipliant par l'écart-type utilisé pour
# normaliser les cibles.
def _denormalize_rmse(loss: float, sigma: torch.Tensor | None) -> float:
    if sigma is not None:
        return float(sigma * loss)
    return loss
