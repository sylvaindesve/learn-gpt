from collections.abc import Callable

import torch


# Normaliser un tenseur en centrant sur la moyenne et en divisant
# par l'écart-type (ce qui ramène les données à un écart-type de 1)
def normalize(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = data.mean()  # μ = moyenne
    sigma = (((data - mu) ** 2).mean()) ** 0.5  # σ = écart-type  # noqa: RUF003
    return (data - mu) / sigma, mu, sigma


# Entraînement d'un modèle de régression linéaire.
def train(
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    epochs: int = 300,
    logger: Callable[[str], None] = lambda _: None,
    log_every: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    # Initialisation du poids
    w = torch.tensor(0.0, requires_grad=True)
    # Initialisation du biais
    b = torch.tensor(0.0, requires_grad=True)

    loss_history: list[float] = []
    for epoch in range(epochs):
        # Prédiction du modèle
        y_pred = x * w + b

        # Perte (MSE) moyenne sur l'ensemble des données
        loss = ((y_pred - y) ** 2).mean()

        # Backpropagation pour calcul des gradients
        loss.backward()

        # Mise à jour des paramètres
        with torch.no_grad():
            w -= lr * w.grad  # pyright: ignore[reportOperatorIssue]
            b -= lr * b.grad  # pyright: ignore[reportOperatorIssue]

        # Remise à zéro des gradients
        w.grad.zero_()  # pyright: ignore[reportOptionalMemberAccess]
        b.grad.zero_()  # pyright: ignore[reportOptionalMemberAccess]

        loss_history.append(loss.item() ** 0.5)

        if epoch % log_every == 0:
            logger(
                f"Etape {epoch + 1}/{epochs}, perte (RMSE) = {loss.item() ** 0.5:.2f}"
            )
    return w, b, loss_history
