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

        block_size = x.shape[1]
        embedding_dim = x.shape[-1]

        # Calcul des scores par produit matriciel de q par k,
        # puis division par la racine carrée de la taille des embeddings pour éviter
        # que cela n'explose
        #   k.transpose(1, 2) est de dimensions (n, embedding, block_size)
        #   et donc scores est de dimensions (n, block_size, block_size) :
        #   chaque position regarde chaque position
        scores = q @ k.transpose(1, 2) / (embedding_dim**0.5)

        # Masque du triangle strictement supérieur : une position n'a pas le droit
        # de regarder une position future. C'est la règle du jeu de la prédiction :
        # à la génération, les tokens suivants n'existent pas encore, donc le modèle
        # doit apprendre à prédire avec le passé seul (sinon il lirait la réponse).
        # Note : sans effet tant que l'on ne renvoie que la dernière position
        # (elle voit déjà tout le bloc) ; deviendra indispensable quand la perte
        # sera calculée à toutes les positions, comme dans un vrai GPT.
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

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
