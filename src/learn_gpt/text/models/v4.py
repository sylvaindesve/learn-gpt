import torch
import torch.nn as nn


# Une couche d'attention à plusieurs têtes
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_head: int):
        super().__init__()

        if embedding_dim % n_head != 0:
            raise ValueError(
                f"n_head ({n_head}) doit être un diviseur de "
                f"embedding_dim ({embedding_dim})"
            )

        self.n_head = n_head
        self.head_dim = embedding_dim // n_head

        # 3 matrices de dimensions (embedding_dim, embedding_dim)
        # Query : ce que je cherche
        self.wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Key : ce que je contiens
        self.wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Value : ce que j'offre
        self.wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Pour recoller les têtes ensemble
        self.wo = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forme de x = (n, block_size, embedding_dim)
        # -> n séquences de block_size embeddings
        n, block_size, embedding_dim = x.shape
        n_head, head_dim = self.n_head, self.head_dim

        # On commence par créer Query, Key et Value
        #   Chacun est de dimensions (n, n_head, block_size, head_dim)
        #   -> un sous-ensemble de l'embedding, pour chaque position, pour chaque tête,
        #   pour chaque séquence
        #
        #   .reshape(n, block_size, n_head, head_dim) ne fait qu'éclater
        #   la dernière dimension
        #   Ainsi, embedding_dim devient (n_head, head_dim) dans la forme.
        #   .transpose(1, 2) échange les axes 1 et 2, i.e. block_size et n_head
        #   Cela est nécessaire pour le produit matriciel qui va venir
        q = self.wq(x).reshape(n, block_size, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).reshape(n, block_size, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).reshape(n, block_size, n_head, head_dim).transpose(1, 2)

        # Calcul des scores par produit matriciel de q par k,
        # puis division par la racine carrée de la taille des têtes pour éviter
        # que cela n'explose
        #   k.transpose(2, 3) est de dimensions (n, n_head, head_dim, block_size)
        #   et donc scores est de dimensions (n, n_head, block_size, block_size) :
        #   chaque position regarde chaque position
        scores = q @ k.transpose(2, 3) / (head_dim**0.5)

        # Un softmax pour que les poids soient entre 0 et 1 et aient une somme de 1
        #   dim=-1 pour que le softmax s'applique sur la dernière dimension,
        #   i.e. la position regardée
        #   weights est de dimensions (n, n_head, block_size, block_size)
        weights = torch.softmax(scores, dim=-1)

        # On calcule la valeur "renforcée" par le calcul d'attention
        #   (n, n_head, block_size, block_size) @ (n, n_head, block_size, head_dim)
        #   -> (n, n_head, block_size, head_dim)
        #   Dans un produit matriciel, il faut regarder les 2 dernières dimensions,
        #   les autres sont broadcastées
        #   Ici, on a bien (block_size, block_size) @ (block_size, head_dim)
        #   -> (block_size, head_dim)
        output_per_head = weights @ v

        # On recolle
        #   .transpose(1, 2) inverse n_head et block_size
        #   -> (n, block_size, n_head, head_dim)
        #   .reshape(n, block_size, embedding_dim) recolle les 2 dernières dimensions
        output = output_per_head.transpose(1, 2).reshape(n, block_size, embedding_dim)

        # Une dernière opération linéaire pour mélanger (avec des poids appris)
        # les têtes
        return self.wo(output)


# Un modèle v4 avec attention multi-têtes
class MultiHeadAttentionCharacterModel(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, block_size: int, n_head: int
    ):
        super().__init__()
        self.block_size = block_size

        # Poids des tokens
        self.wte = nn.Embedding(vocab_size, embedding_dim)

        # Poids de la position des tokens dans le contexte
        self.wpe = nn.Embedding(block_size, embedding_dim)

        # Couche d'attention, cette fois multi-têtes
        self.attn = MultiHeadAttentionLayer(embedding_dim, n_head)

        # Sortie : on ramène au vocabulaire
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forme de x = (n, block_size) -> n séquences de block_size tokens

        # Codage des tokens et de leurs positions
        e = self.wte(x) + self.wpe(torch.arange(x.shape[1]))

        # Calcul d'attention
        #   Forme (n, block_size, embedding_dim)
        a = self.attn(e)

        # Sortie de forme (n, vocab_size)
        #   -> pour chaque séquence, le score de chaque token du vocabulaire
        return self.output(a[:, -1, :])

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
