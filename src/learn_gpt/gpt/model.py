import torch
import torch.nn as nn


# Normalisation par la moyenne quadratique (RMSNorm)
#   On divise chaque vecteur par sa moyenne quadratique, ce qui ramène tous les
#   vecteurs à la même échelle. Contrairement à LayerNorm, on ne soustrait pas la
#   moyenne : moins de calculs, et les GPT modernes s'en contentent.
#   Version simplifiée : le gain appris de la RMSNorm originale n'est pas implémenté.
def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    ms = (x**2).mean(dim=-1, keepdim=True)
    return x * (ms + 1e-5) ** -0.5


# Une couche d'attention à plusieurs têtes
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_head: int, dropout: float = 0.0):
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

        # Dropout sur l'attention
        self.dropout_attention = nn.Dropout(dropout)

        # Pour recoller les têtes ensemble
        self.wo = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forme de x = (n, block_size, embedding_dim) -> n séquences de block_size
        # embeddings
        n, block_size, embedding_dim = x.shape
        n_head, head_dim = self.n_head, self.head_dim

        # On commence par créer Query, Key et Value
        #   Chacun est de dimensions (n, n_head, block_size, head_dim).
        #   -> un sous-ensemble de l'embedding, pour chaque position,
        #   pour chaque tête, pour chaque séquence
        #
        #   .reshape(n, block_size, n_head, head_dim) ne fait qu'éclater la
        #   dernière dimension: embedding_dim devient (n_head, head_dim)
        #
        #   .transpose(1, 2) échange les axes 1 et 2, i.e. block_size et n_head
        #   Cela est nécessaire pour le produit matriciel qui va venir
        q = self.wq(x).reshape(n, block_size, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).reshape(n, block_size, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).reshape(n, block_size, n_head, head_dim).transpose(1, 2)

        # Calcul des scores par produit matriciel de q par k,
        # puis division par la racine carrée de la taille des têtes pour
        # éviter que cela n'explose
        #
        # k.transpose(2, 3) est de dimensions (n, n_head, head_dim, block_size)
        # et donc scores est de dimensions (n, n_head, block_size, block_size) :
        # chaque position regarde chaque position
        scores = q @ k.transpose(2, 3) / (head_dim**0.5)

        # Masque du triangle haut droit de la matrice -> interdiction pour une position
        # de regarder une position future
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        # Un softmax pour que les poids soient entre 0 et 1 et aient une somme de 1
        #   dim=-1 pour que le softmax s'applique sur la dernière dimension,
        #   i.e. la position regardée
        #   weights est de dimensions (n, n_head, block_size, block_size)
        weights = torch.softmax(scores, dim=-1)

        # Dropout appliqués aux scores d'attention
        weights = self.dropout_attention(weights)

        # On calcule la valeur "renforcée" par le calcul d'attention
        #   (n, n_head, block_size, block_size) @ (n, n_head, block_size, head_dim)
        #   -> (n, n_head, block_size, head_dim)
        #
        # Dans un produit matriciel, il faut regarder les 2 dernières dimensions,
        # les autres sont broadcastées
        # Ici, on a bien (block_size, block_size) @ (block_size, head_dim)
        # -> (block_size, head_dim)
        output_per_head = weights @ v

        # On recolle
        #
        # .transpose(1, 2) inverse n_head et block_size
        #   -> (n, block_size, n_head, head_dim)
        # .reshape(n, block_size, embedding_dim) recolle les 2 dernières dimensions
        output = output_per_head.transpose(1, 2).reshape(n, block_size, embedding_dim)

        # Une dernière opération linéaire pour mélanger (avec des poids appris)
        # les têtes
        return self.wo(output)


# Un bloc Transformer (enfin) : deux sous-couches, chacune précédée d'une
# normalisation et suivie d'une connexion résiduelle.
#   - normalisation -> attention multi-têtes -> résidu
#   - normalisation -> MLP avec ReLU          -> résidu
# On normalise l'entrée de chaque sous-couche plutôt que sa sortie : c'est la
# convention des GPT modernes, plus stable quand on empile les blocs.
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_head: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Attention multi-têtes : la couche qui communique
        #   Chaque position échange de l'information avec les autres
        self.attn = MultiHeadAttentionLayer(embedding_dim, n_head, dropout)

        # Dropout sur l'attention
        self.dropout_attention = nn.Dropout(dropout)

        # Couche MLP : la couche qui calcule
        #   Chaque position transforme l'information qu'elle a reçue
        #   Sans biais
        self.mlp = nn.Sequential(
            # On projette d'abord vers un espace plus large (expansion_factor
            # fois embedding_dim, 4 par défaut) : c'est là que se fait
            # l'essentiel du calcul, et c'est ce qui donne au bloc la majorité
            # de ses paramètres.
            nn.Linear(embedding_dim, expansion_factor * embedding_dim, bias=False),
            nn.ReLU(),
            nn.Linear(expansion_factor * embedding_dim, embedding_dim, bias=False),
        )

        # Dropout en sortie du MLP
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forme de x = (n, block_size, embedding_dim)
        # -> n séquences de block_size embeddings

        # Normalisation avant l'attention
        normalized_before = rmsnorm(x)

        # Calcul d'attention sur les entrées normalisées
        attention = self.attn(normalized_before)

        # Dropout sur l'attention
        attention = self.dropout_attention(attention)

        # Connexion résiduelle : ce qui permet d'entraîner un réseau profond
        # Lorsque les poids sont à zéro, on se retrouve avec x
        # Cela permet au gradient de "trouver un chemin"
        attention_added = x + attention

        # Normalisation avant le MLP
        normalized_after = rmsnorm(attention_added)

        # MLP
        f = self.mlp(normalized_after)

        # Dropout sur la sortie du MLP
        f = self.dropout_mlp(f)

        # Connexion résiduelle (encore)
        result = attention_added + f

        # result est toujours de forme (n, block_size, embedding_dim)
        return result


# Notre modèle utilisant le bloc Transformer
class MicroGPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        n_head: int = 4,
        n_layer: int = 2,
        expansion_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.block_size = block_size

        # Poids des tokens
        self.wte = nn.Embedding(vocab_size, embedding_dim)

        # Poids de la position des tokens dans le contexte
        self.wpe = nn.Embedding(block_size, embedding_dim)

        # Dropout sur les embeddings
        self.dropout_embeddings = nn.Dropout(dropout)

        # Blocs Transformer
        self.transformers = nn.Sequential(
            *[
                TransformerBlock(embedding_dim, n_head, expansion_factor, dropout)
                for _ in range(n_layer)
            ]
        )

        # Sortie : il faut se ramener au vocabulaire
        #   sans biais
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forme de x = (n, block_size) -> n séquences de block_size tokens

        # L'embedding code à la fois le token et sa position dans le contexte
        #   Forme (n, block_size, embedding_dim) -> n séquences de block_size embeddings
        e = self.wte(x) + self.wpe(torch.arange(x.shape[1]))

        # Dropout appliqué aux embeddings
        e = self.dropout_embeddings(e)

        # On traverse les blocs Transformer (attention + MLP)
        #   Forme (n, block_size, embedding_dim)
        h = self.transformers(e)

        # Dernière normalisation avant la sortie : en traversant les blocs, les
        # valeurs peuvent changer d'échelle. C'est le rôle du `ln_f` d'un GPT.
        h = rmsnorm(h)

        # En sortie, des logits pour chaque position
        #   Forme (n, block_size, vocab_size)
        return self.output(h)

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

        # On commence avec un seul <bos> : c'est un début de contexte que le
        # modèle a vu à l'entraînement (le flux ne contient jamais deux <bos>
        # consécutifs). Le contexte s'allonge ensuite au fil de la génération.
        ctx = [bos_id]

        with torch.no_grad():
            # On ne dépasse pas la longueur maximum
            for _ in range(max_tokens):
                # Fenêtre glissante des block_size derniers tokens
                #   si ctx est plus court que block_size, on prend tout
                x = torch.tensor([ctx[-self.block_size :]])

                # La sortie du modèle est de forme (batch_size, block_size, vocab_size)
                # Ici, on a envoyé qu'un seul batch et on veut les scores de la dernière
                # position du bloc
                #   -> donc [0, -1]
                # logits est donc de forme (vocab_size,)
                logits = self.forward(x)[0, -1]

                # Distribution des probabilités contrôlée par la température
                probs = torch.softmax(logits / temperature, dim=0)

                # Tirage aléatoire
                next_token = int(torch.multinomial(probs, 1).item())

                # Si la prédiction est la fin de séquence, on s'arrête
                if next_token == eos_id:
                    break

                # On rajoute le token prédit à la sortie
                output.append(next_token)

                # On ajoute au context pour reprendre la boucle
                ctx.append(next_token)

        return output
