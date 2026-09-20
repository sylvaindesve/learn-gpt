Ceci est un projet pour apprendre. Par conséquent, ne modifie jamais un fichier sauf si je te le demande expressément.

## Conventions

- Tout est en français : commentaires, docstrings, sorties de la CLI, README.
- Le README est le fil pédagogique du projet. Il explique le *pourquoi* avant le *comment* et doit rester synchronisé avec le code.
- Certaines redondances sont volontaires : `neuron/neuron.py` réimplémente à la main ce que `neuron/mlp.py` fait avec PyTorch. Le but est de montrer la mécanique avant de s'appuyer sur le framework. Ne pas « nettoyer » en supprimant l'un des deux.

## Vérifications avant de conclure

```bash
uv run ruff check . && uv run ruff format --check .
uvx pyright --pythonpath .venv/bin/python
```

`pyright` n'est pas une dépendance du projet : il tourne via `uvx`, sans être installé dans le `.venv`.

## Piège d'environnement

Sous le bac à sable de cette session, `uv` ne peut pas écrire dans `~/.cache/uv` et échoue sur `Operation not permitted`. Rediriger `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` et `UV_TOOL_DIR` vers un dossier temporaire. Sans effet sur une machine normale.
