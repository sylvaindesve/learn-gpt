from collections.abc import Callable

import torch


# Un neurone artificiel dans sa forme la plus simple
# Ici, x est une liste de n exemples de forme (n,)
# w et b sont des scalaires
def neuron(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ici la fonction d'activation est `tanh` (tangente hyperbolique)
    return torch.tanh(w * x + b)


# Une couche de neurones
# x est une liste de n exemples de forme (n,)
# w est une liste de h poids, un par neurone
#   le nombre de neurones de cette couche est donc h
# b est une liste de h biais, un par neurone
def layer(
    x: torch.Tensor, w: list[torch.Tensor], b: list[torch.Tensor]
) -> list[torch.Tensor]:
    out = []
    for i in range(len(w)):
        out.append(neuron(x, w[i], b[i]))
    return out


# Un réseau de neurones 1 → h → 1
def network(
    x: torch.Tensor,  # liste de n exemples de forme (n,)
    wh: list[torch.Tensor],  # poids de la couche cachée de h = len(wh) neurones
    bh: list[torch.Tensor],  # biais de la couche cachée
    wo: list[torch.Tensor],  # poids du neurone de sortie
    bo: torch.Tensor,  # biais du neurone de sortie
) -> torch.Tensor:
    # Sorties de la couche cachées
    hidden = layer(x, wh, bh)

    # On multiplie chacune des sorties de la couches cachée par les
    # poids du neurone de sortie, on somme et ajoute le biais de sortie
    return sum([wo[i] * hidden[i] for i in range(len(wo))]) + bo


# Initialise les paramètres pour un réseau avec
# hidden_layer_size neurones cachés
def create_network_parameters(
    hidden_layer_size: int,
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    wh = [
        torch.tensor(float(v), requires_grad=True)
        for v in torch.randn(hidden_layer_size, generator=g)
    ]
    bh = [
        torch.tensor(float(v), requires_grad=True)
        for v in torch.randn(hidden_layer_size, generator=g)
    ]
    wo = [
        torch.tensor(float(v), requires_grad=True)
        for v in torch.randn(hidden_layer_size, generator=g)
    ]
    bo = torch.tensor(0.0, requires_grad=True)
    return wh, bh, wo, bo


def train(
    x: torch.Tensor,  # liste de n exemples de forme (n,)
    y: torch.Tensor,  # les valeurs cibles pour les n exemples, de forme (n,)
    wh: list[torch.Tensor],  # poids de la couche cachée de h = len(wh) neurones
    bh: list[torch.Tensor],  # biais de la couche cachée
    wo: list[torch.Tensor],  # poids du neurone de sortie
    bo: torch.Tensor,  # biais du neurone de sortie
    lr: float = 0.01,  # taux d'apprentissage
    epochs: int = 3000,  # nombre d'itérations sur l'ensemble des exemples
    logger: Callable[[str], None] = lambda _: None,
    log_every: int = 50,
) -> list[float]:
    loss_history: list[float] = []
    for epoch in range(epochs):
        # Prédiction du modèle
        y_pred = network(x, wh, bh, wo, bo)

        # Perte (MSE) moyenne sur l'ensemble des données
        loss = ((y_pred - y) ** 2).mean()

        # Backpropagation pour calcul des gradients
        loss.backward()

        # Mise à jour des paramètres
        with torch.no_grad():
            for j in range(len(wh)):
                wh[j] -= lr * wh[j].grad  # pyright: ignore[reportOperatorIssue]
                bh[j] -= lr * bh[j].grad  # pyright: ignore[reportOperatorIssue]
                wo[j] -= lr * wo[j].grad  # pyright: ignore[reportOperatorIssue]
            bo -= lr * bo.grad  # pyright: ignore[reportOperatorIssue]

        # Remise à zéro des gradients
        for j in range(len(wh)):
            wh[j].grad.zero_()  # pyright: ignore[reportOptionalMemberAccess]
            bh[j].grad.zero_()  # pyright: ignore[reportOptionalMemberAccess]
            wo[j].grad.zero_()  # pyright: ignore[reportOptionalMemberAccess]
        bo.grad.zero_()  # pyright: ignore[reportOptionalMemberAccess]

        loss_history.append(loss.item() ** 0.5)

        if epoch % log_every == 0:
            logger(
                f"Etape {epoch + 1}/{epochs}, perte (RMSE) = {loss.item() ** 0.5:.2f}"
            )
    return loss_history
