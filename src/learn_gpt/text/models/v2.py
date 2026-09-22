import torch
import torch.nn as nn


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

        # On aplatit le tout, de (n, block_size, embedding_dim)
        # on passe à (N, block_size * embedding_dim)
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
