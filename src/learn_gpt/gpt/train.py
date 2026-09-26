from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn


# Construit les fenêtres (contexte, cible) qui commencent aux positions données
def windows_at(
    stream: list[int], starts: list[int], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([stream[i : i + block_size] for i in starts])
    y = torch.tensor([stream[i + 1 : i + block_size + 1] for i in starts])
    return x, y


# Choisit au hasard n_samples fenêtres de block_size tokens dans le flux
def sample(
    stream: list[int], n_samples: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(stream) - block_size, (n_samples,)).tolist()
    return windows_at(stream, starts, block_size)


# Calcule la perte du modèle sur des fenêtres du flux.
#
#   - l'échantillon est tiré une seule fois, avec une graine fixe : on mesure
#     donc toujours sur les mêmes fenêtres, ce qui rend les mesures comparables
#     entre elles. Un lot tiré au hasard à chaque fois donnerait une valeur
#     bruitée, qui peut même passer sous le plancher du corpus.
#   - le calcul se fait par lots : évaluer d'un coup les dizaines de milliers
#     de fenêtres d'un vrai corpus demanderait des centaines de Mo de logits.
#
# Si n_windows est None, ou dépasse le nombre de fenêtres disponibles, on
# évalue sur tout le flux.
def evaluate_stream(
    model: nn.Module,
    stream: list[int],
    vocab_size: int,
    block_size: int,
    n_windows: int | None = None,
    batch_size: int = 256,
    seed: int = 0,
) -> float:
    n_available = len(stream) - block_size

    if n_windows is None or n_windows >= n_available:
        starts = list(range(n_available))
    else:
        generator = torch.Generator().manual_seed(seed)
        starts = torch.randperm(n_available, generator=generator)[:n_windows].tolist()

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss, total_windows = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            x, y = windows_at(stream, starts[i : i + batch_size], block_size)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            # On pondère par le nombre de fenêtres du lot pour que le dernier
            # lot, souvent plus petit, ne pèse pas autant que les autres
            total_loss += float(loss.item()) * x.shape[0]
            total_windows += x.shape[0]

    return total_loss / total_windows


# Données que l'entraînement envoie à chaque étape
@dataclass(frozen=True)
class Progress:
    step: int  # l'étape qui vient de se terminer
    train_loss: float  # la perte sur le lot de cette étape
    val_loss: float | None  # la perte de validation, si calculée à cette étape


# Entraînement en mesurant régulièrement la perte sur un jeu de
# validation que le modèle ne voit jamais pendant l'apprentissage.
# C'est cette perte-là qui dit si le modèle apprend vraiment, ou s'il
# surapprend les données d'entraînement.
#
# Renvoie un générateur qui émet des objets Progress à chaque étape.
# La perte de validation est évaluée seulement toutes les `eval_every`
# étapes.
def train_stream_with_validation(
    model: nn.Module,  # le modèle
    train_stream: list[int],  # le flux d'entraînement
    val_stream: list[int],  # le flux de validation
    vocab_size: int,  # taille du vocabulaire
    block_size: int,  # taille de la fenêtre de contexte
    lr: float,  # taux d'apprentissage
    weight_decay: float,  # dégradation des pondérations
    batch_size: int,  # taille des lots
    steps: int,  # nombre d'étapes
    eval_every: int = 200,  # fréquence d'évaluation de la validation
    val_windows: int = 1024,  # nombre de fenêtres de validation évaluées
    seed: int = 0,
) -> Iterator[Progress]:
    torch.manual_seed(seed)

    # Optimiseur
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Fonction de calcul de perte : entropie croisée
    loss_fn = nn.CrossEntropyLoss()

    for step in range(steps):
        # On tire au hasard batch_size fenêtres de block_size tokens dans le flux
        x, y = sample(train_stream, batch_size, block_size)

        # Remise à zéro des gradients
        opt.zero_grad()

        # La sortie du modèle est de forme (batch_size, block_size, vocab_size)
        logits = model(x)

        # Et donc on calcule la perte pour chaque position
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

        # Calcul des gradients
        loss.backward()

        # On applique une décroissance linéaire du lr
        #   Cela permet d'affiner sur la fin de l'entraînement
        #   Il faut le faire pour chaque groupe de paramètres géré par l'optimiseur
        for param_group in opt.param_groups:
            param_group["lr"] = lr * (1 - step / steps)

        # Mise à jour des poids
        opt.step()

        # Évaluation sur le jeu de validation
        val_loss = None
        if (step + 1) % eval_every == 0:
            val_loss = evaluate_stream(
                model,
                val_stream,
                vocab_size,
                block_size,
                n_windows=val_windows,
                seed=seed,
            )

            # evaluate_stream a passé le modèle en mode évaluation
            model.train()

        yield Progress(step + 1, loss.item(), val_loss)
