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


# Choisit au hasard n_samples échantillons de block_size tokens dans le stream
def sample(
    stream: list[int], n_samples: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(stream) - block_size, (n_samples,))
    x = torch.tensor([stream[i : i + block_size] for i in starts])
    y = torch.tensor([stream[i + 1 : i + block_size + 1] for i in starts])
    return x, y


# Un entraînement avec calcul de perte sur toutes les positions
def train_stream(
    model: nn.Module,  # le modèle
    stream: list[int],  # un flux continu d'ID de tokens
    vocab_size: int,  # taille du vocabulaire
    block_size: int,  # taille de la fenêtre de contexte
    lr: float,  # taux d'apprentissage
    batch_size: int,  # taille des lots
    steps: int,  # nombre d'étapes
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

    for step in range(steps):
        # On tire au hasard batch_size échantillons de block_size tokens dans le flux
        x, y = sample(stream, batch_size, block_size)

        # Remise à zéro des gradients
        opt.zero_grad()

        # La sortie du modèle est de forme (batch_size, block_size, vocab_size)
        #   - pour chaque batch
        #   - pour chaque position dans le bloc
        #   - le score de la prochaine position
        logits = model(x)

        # Et donc on calcule la perte pour chaque position
        #
        #   .reshape(-1, vocab_size) permet de passer à
        #   (batch_size * block_size, vocab_size)
        #   -> On "enlève" la notion de batch, on présente tous les blocs
        #
        #   .reshape(-1) fait la même chose sur y : (batch_size, block_size)
        #   -> (batch_size * block_size,)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

        # Calcul des gradients
        loss.backward()

        # Mise à jour des poids
        opt.step()

        loss_history.append(loss.item())

        if step % log_every == 0:
            logger(f"Etape {step + 1}/{steps}, perte = {loss_history[-1]:.2f}")
    return loss_history
