import torch


# Un neurone artificiel dans sa forme la plus simple
def neuron(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ici la fonction d'activation est `tanh` (tangente hyperbolique)
    return torch.tanh(w * x + b)
