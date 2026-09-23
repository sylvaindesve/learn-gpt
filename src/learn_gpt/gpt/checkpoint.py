from pathlib import Path

import torch

from learn_gpt.gpt.model import MicroGPTModel
from learn_gpt.gpt.tokenizer import BPETokenizer


# Sauvegarde un modèle entraîné, avec tout ce qu'il faut pour le recharger :
#   - ses hyperparamètres, pour pouvoir reconstruire l'architecture
#   - les fusions et le vocabulaire du tokenizer, pour pouvoir encoder un texte
#     et décoder les tokens générés
#   - ses poids
#
# Sauvegarder les poids seuls ne suffirait pas : sans les hyperparamètres, on
# ne saurait pas quelle architecture reconstruire pour les accueillir.
# Et pour ce tokenizer-ci il faut les fusions autant que le vocabulaire : le
# vocabulaire sert à décoder les tokens, mais c'est la liste des fusions qui
# sert à encoder un texte.
def save_model(
    path: Path,
    model: MicroGPTModel,
    tokenizer: BPETokenizer,
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
            "merges": tokenizer.merges,
            "vocab": tokenizer.vocab,
            "model_state": model.state_dict(),
        },
        path,
    )


# Recharge un modèle et son tokenizer depuis un fichier de sauvegarde
def load_model(path: Path) -> tuple[MicroGPTModel, BPETokenizer]:
    checkpoint = torch.load(path, weights_only=True)

    # Les fusions reviennent sous forme de listes : on les remet en tuples,
    # car c'est sous cette forme que le tokenizer les utilise comme clés
    merges = [tuple(pair) for pair in checkpoint["merges"]]
    tokenizer = BPETokenizer(merges, checkpoint["vocab"])

    model = MicroGPTModel(
        vocab_size=len(tokenizer.vocab), **checkpoint["hyperparameters"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer
