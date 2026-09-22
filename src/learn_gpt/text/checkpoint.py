from pathlib import Path

import torch

from learn_gpt.text.models.v6 import MicroGPTModel
from learn_gpt.text.tokenizer import CharTokenizer


# Sauvegarde un modèle entraîné, avec tout ce qu'il faut pour le recharger :
#   - ses hyperparamètres, pour pouvoir reconstruire l'architecture
#   - le vocabulaire du tokenizer, pour pouvoir décoder les tokens générés
#   - ses poids
#
# Sauvegarder les poids seuls ne suffirait pas : sans les hyperparamètres, on
# ne saurait pas quelle architecture reconstruire pour les accueillir, et sans
# le vocabulaire on ne saurait pas retransformer les tokens en caractères.
def save_model(
    path: Path,
    model: MicroGPTModel,
    tokenizer: CharTokenizer,
    *,
    embedding_dim: int,
    block_size: int,
    n_head: int,
    n_layer: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "hyperparameters": {
                "embedding_dim": embedding_dim,
                "block_size": block_size,
                "n_head": n_head,
                "n_layer": n_layer,
            },
            "vocab": tokenizer.vocab,
            "model_state": model.state_dict(),
        },
        path,
    )


# Recharge un modèle et son tokenizer depuis un fichier de sauvegarde
def load_model(path: Path) -> tuple[MicroGPTModel, CharTokenizer]:
    checkpoint = torch.load(path, weights_only=True)

    tokenizer = CharTokenizer(checkpoint["vocab"])
    model = MicroGPTModel(
        vocab_size=len(tokenizer.vocab), **checkpoint["hyperparameters"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer
