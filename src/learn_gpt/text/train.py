from collections.abc import Callable

import torch
import torch.nn as nn


# Un entraînement sans données de validation
def train(
    model: nn.Module,  # le modèle
    # x est de forme (n,) pour la v1 : n exemples d'ID de token
    # x est de forme (n, block_size) pour la v2 : n blocs de block_size ID de token
    x: torch.Tensor,
    y: torch.Tensor,  # forme (n,) : les ID de token attendus pour chaque exemple
    lr: float,  # taux d'apprentissage
    epochs: int,  # nombre d'époques
    seed: int = 0,
    logger: Callable[[str], None] = lambda _: None,
    log_every: int = 50,
) -> list[float]:
    torch.manual_seed(seed)

    # Optimiseur
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Fonction de calcul de perte : entropie croisée
    loss_fn = nn.CrossEntropyLoss()

    loss_history: list[float] = []

    for epoch in range(epochs):
        # Remise à zéro des gradients
        opt.zero_grad()

        # Sortie du modèle, des logits (score)
        # forme (n, vocab_size)
        logits = model(x)

        # Calcul de la perte pour l'ensemble des exemples en regardant
        # la probabilité du token attendu
        loss = loss_fn(logits, y)

        # Calcul des gradients
        loss.backward()

        # Mise à jour des poids
        opt.step()

        loss_history.append(loss.item())

        if epoch % log_every == 0:
            logger(f"Epoque {epoch + 1}/{epochs}, perte = {loss_history[-1]:.2f}")
    return loss_history
