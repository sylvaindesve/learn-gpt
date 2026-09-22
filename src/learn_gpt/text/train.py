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


# Le même entraînement, mais on mesure régulièrement la perte sur un jeu de
# validation que le modèle ne voit jamais pendant l'apprentissage.
# C'est cette perte-là qui dit si le modèle apprend vraiment, ou s'il
# surapprend les données d'entraînement.
#
# Renvoie (pertes d'entraînement, pertes de validation, étapes d'évaluation) :
# la perte d'entraînement est mesurée à chaque étape, la perte de validation
# seulement toutes les `eval_every` étapes, d'où la liste des étapes.
def train_stream_with_validation(
    model: nn.Module,  # le modèle
    train_stream: list[int],  # le flux d'entraînement
    val_stream: list[int],  # le flux de validation
    vocab_size: int,  # taille du vocabulaire
    block_size: int,  # taille de la fenêtre de contexte
    lr: float,  # taux d'apprentissage
    batch_size: int,  # taille des lots
    steps: int,  # nombre d'étapes
    eval_every: int = 200,  # fréquence d'évaluation de la validation
    val_windows: int = 1024,  # nombre de fenêtres de validation évaluées
    seed: int = 0,
    logger: Callable[[str], None] = lambda _: None,
    log_every: int = 50,
) -> tuple[list[float], list[float], list[int]]:
    torch.manual_seed(seed)

    # Optimiseur
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Fonction de calcul de perte : entropie croisée
    loss_fn = nn.CrossEntropyLoss()

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    eval_steps: list[int] = []

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

        # Mise à jour des poids
        opt.step()

        train_loss_history.append(loss.item())

        # Évaluation sur le jeu de validation
        if (step + 1) % eval_every == 0:
            val_loss = evaluate_stream(
                model,
                val_stream,
                vocab_size,
                block_size,
                n_windows=val_windows,
                seed=seed,
            )
            val_loss_history.append(val_loss)
            eval_steps.append(step + 1)

            # evaluate_stream a passé le modèle en mode évaluation
            model.train()

            logger(
                f"Etape {step + 1}/{steps}, "
                f"perte train = {train_loss_history[-1]:.2f}, "
                f"perte val = {val_loss:.2f}"
            )
        elif step % log_every == 0:
            logger(f"Etape {step + 1}/{steps}, perte = {train_loss_history[-1]:.2f}")

    return train_loss_history, val_loss_history, eval_steps
