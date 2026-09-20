from collections.abc import Callable

import torch
import torch.nn as nn


# Un modèle très simple de prédiction du caractère suivant
#   entrée de forme (n,) : n exemples (des IDs de token)
#   sortie de forme (n, vocab_size) : pour chaque exemple, les scores (logits)
#                                     des tokens du vocabulaire
class CharacterModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.wte(x))


# Un entraînement sans données de validation
def train(
    model: nn.Module,  # le modèle
    x: torch.Tensor,  # forme (n,) : n exemples d'ID de token
    y: torch.Tensor,  # forme(n,) : les ID de token attendus pour chaque exemple
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

        # Calcul de la perte pour l'ensemble de exemples en regardant
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


# Les K tokens les plus probables selon le modèle après le token fourni
def top_next_tokens(
    model: nn.Module, token_id: int, k: int = 3
) -> list[tuple[int, float]]:
    model.eval()
    t = torch.tensor([token_id])
    logits = model(t)[0]
    probs = torch.softmax(logits, dim=0)
    top = torch.topk(probs, k)
    return [
        (tok, round(float(prob), 3))
        for tok, prob in zip(top.indices.tolist(), top.values.tolist(), strict=True)
    ]
