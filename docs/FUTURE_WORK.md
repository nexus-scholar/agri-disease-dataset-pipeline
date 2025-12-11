# Future Work

## Domain-Agnostic Support

- Generalize processors so they operate on declarative dataset schemas (folder layout, label rules) to support non-plant domains.
- Introduce plugin registry for new dataset modules (`pipeline/<domain>.py`) with discovery via entrypoints or config files.
- Provide metadata templates and validation to ensure compatibility with downstream ML pipelines regardless of source domain.
- Offer sample processors for other domains (e.g., materials, medical imaging) to demonstrate extensibility.

