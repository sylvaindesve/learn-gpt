import torch
import torch.nn as nn


# Une couche d'attention pour le modèle v3
class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        # 3 matrices de dimensions (embedding_dim, embedding_dim)
        # Query : ce que je cherche
        self.wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Key : ce que je contiens
        self.wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Value : ce que j'offre
        self.wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forme de x = (n, block_size, embedding_dim)
        # -> N séquences de block_size embeddings

        # On commence par créer Query, Key et Value
        # Chacun est de dimensions (N, block_size, embedding_dim) :
        # un vecteur pour chaque position de chaque séquence
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        embedding_dim = x.shape[-1]

        # Calcul des scores par produit matriciel de q par k,
        # puis division par la racine carrée de la taille des embeddings pour éviter
        # que cela n'explose
        #   k.transpose(1, 2) est de dimensions (n, embedding, block_size)
        #   et donc scores est de dimensions (n, block_size, block_size) :
        #   chaque position regarde chaque position
        scores = q @ k.transpose(1, 2) / (embedding_dim**0.5)

        # Un softmax pour que les poids soient entre 0 et 1 et aient une somme de 1
        #   dim=-1 pour que le softmax s'applique sur la dernière dimension, i.e. la
        #   position regardée
        #   weights est de dimensions (n, block_size, block_size)
        weights = torch.softmax(scores, dim=-1)

        # On renvoie la valeur "renforcée" par le calcul d'attention
        #   (n, block_size, block_size) @ (n, block_size, embedding_dim)
        #   -> (n, block_size, embedding_dim)
        #   Dans un produit matriciel, il faut regarder les 2 dernières dimensions,
        #   les autres sont broadcastées
        #   Ici, on a bien (block_size, block_size) @ (block_size, embedding_dim)
        #   -> (block_size, embedding_dim)
        return weights @ v


# Un modèle v3 qui intègre le calcul d'attention.
# Par rapport à la v2, on retire la couche cachée : l'attention suffit à
# mélanger l'information du contexte. Le nombre de paramètres de l'attention
# ne dépend pas de block_size (contrairement à l'entrée de la couche cachée
# de la v2, block_size * embedding_dim), donc le modèle est bien plus petit.
class AttentionCharacterModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int):
        super().__init__()
        self.block_size = block_size

        # Poids des tokens
        self.wte = nn.Embedding(vocab_size, embedding_dim)

        # Poids de la position des tokens dans le contexte
        #   Dans le modèle v2 ces poids n'étaient pas nécessaires.
        #   Ils le deviennent dans la v3 car le calcul d'attention
        #   multiplie chaque token par les mêmes poids et par conséquent
        #   sans ces poids de position l'attention de 'cha' serait la
        #   même que celle de 'hca'.
        self.wpe = nn.Embedding(block_size, embedding_dim)

        # Couche d'attention
        self.attn = AttentionLayer(embedding_dim)

        # Sortie : il faut se ramener au vocabulaire (car on produit un score
        # du prochain token)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forme de x = (N, block_size) -> N séquences de block_size tokens

        # L'embedding code à la fois le token et sa position dans le contexte
        #   Forme (N, block_size, embedding_dim) -> N séquences de
        #   block_size embeddings
        e = self.wte(x) + self.wpe(torch.arange(x.shape[1]))

        # Calcul d'attention
        #   Forme (N, block_size, embedding_dim)
        # Sans connexion résiduelle, la seule information qui arrive à la sortie
        # est un mélange des vecteurs values : le modèle perd l'identité du
        # dernier token. D'où une perte plus élevée que la v2 (~1.1 contre ~0.4) ;
        # avec `a = e + self.attn(e)` on retombe à ~0.5 à paramètres identiques.
        a = self.attn(e)

        # Pour la sortie finale, on ne regarde que le dernier embedding du bloc
        #   on fournit donc bien à la sortie un tenseur de forme
        #   (N, embedding_dim) afin d'avoir (N, vocab_size)
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
