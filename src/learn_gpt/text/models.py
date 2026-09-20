from collections.abc import Callable

import torch
import torch.nn as nn


# Un modèle v1 très simple de prédiction du caractère suivant
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

    # Les K tokens les plus probables selon le modèle après le token fourni
    def top_next_tokens(self, token_id: int, k: int = 3) -> list[tuple[int, float]]:
        self.eval()

        with torch.no_grad():
            t = torch.tensor([token_id])
            logits = self.forward(t)[0]
            probs = torch.softmax(logits, dim=0)
            top = torch.topk(probs, k)

        return [
            (tok, round(float(prob), 3))
            for tok, prob in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        ]

    # Générer des séquences
    def generate(
        self,
        bos_id: int,
        eos_id: int,
        temperature: float = 0.8,
        max_tokens: int = 20,
    ) -> list[int]:
        self.eval()

        output: list[int] = []

        # On commence avec le début de séquence
        token = bos_id

        # On ne dépasse pas la longueur maximum
        with torch.no_grad():
            for _ in range(max_tokens):
                # Les logits en sortie du modèle
                logits = self.forward(torch.tensor([token]))[0]

                # Distribution des probabilités contrôlée par la température
                probs = torch.softmax(logits / temperature, dim=0)

                # Tirage aléatoire
                token = int(torch.multinomial(probs, 1).item())

                # Si la prédiction est la fin de séquence, on s'arrête
                if token == eos_id:
                    break

                output.append(token)

        return output


# Un modèle v2 qui intègre une fenêtre de contexte et une couche cachée pour
# plus de calculs
class ContextCharacterModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,  # taille de vocabulaire
        embedding_dim: int,  # taille des embeddings
        block_size: int,  # taille de la fenêtre de contexte
        layer_size: int,  # nombre de neurones dans la couche de calcul
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(block_size * embedding_dim, layer_size)
        self.act = nn.Tanh()
        self.output = nn.Linear(layer_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x contient n séquences de block_size tokens : (n, block_size)

        # Chaque token est transformé en un embedding de taille embedding_dim
        te = self.wte(x)  # (n, block_size, embedding_dim)

        # On aplatit le tout, de (n, block_size, embedding_dimension)
        # on passe à (N, block_size * embedding_dimension)
        te = te.reshape(te.shape[0], -1)

        # Il ne reste plus qu'à passer à travers la couche cachée
        # La sortie est de forme (n, vocab_size)
        return self.output(self.act(self.hidden(te)))

    # Générer des séquences
    def generate(
        self,
        bos_id: int,  # ID du token special <bos>
        eos_id: int,  # ID du token special <eos>
        temperature: float = 0.8,  # température pour contrôler la distribution
        max_tokens: int = 20,  # longueur maximale de séquence
    ) -> list[int]:
        self.eval()

        output: list[int] = []

        # On commence avec un contexte rempli de <bos>
        ctx = [bos_id] * self.block_size

        with torch.no_grad():
            # On ne dépasse pas la longueur maximum
            for _ in range(max_tokens):
                # Les logits en sortie du modèle
                logits = self.forward(torch.tensor([ctx]))[0]

                # Distribution des probabilités contrôlée par la température
                probs = torch.softmax(logits / temperature, dim=0)

                # Tirage aléatoire
                next_token = int(torch.multinomial(probs, 1).item())

                # Si la prédiction est la fin de séquence, on s'arrête
                if next_token == eos_id:
                    break

                # On rajoute le token prédit à la sortie
                output.append(next_token)

                # On décale le contexte
                ctx = [*ctx[1:], next_token]

        return output


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
