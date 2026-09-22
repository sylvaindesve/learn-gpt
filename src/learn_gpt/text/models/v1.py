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
