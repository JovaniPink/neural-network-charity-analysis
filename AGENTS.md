# Repository guidance

## Purpose

This repository is a historical TensorFlow notebook, dataset, and SavedModel
archive. It is not a current training pipeline or deployable inference system.

## Canonical command

```sh
python3 scripts/check_repository.py
```

The validator uses only the Python standard library. It must not execute a
notebook, import TensorFlow, deserialize a model, or contact an external source.

## Working rules

- Preserve notebooks, saved outputs, and SavedModel directories as historical
  evidence unless a task explicitly authorizes artifact migration.
- Never use the archived models for consequential decisions or expose them as
  a production service.
- Treat all CSV files as unverified third-party snapshots with incomplete
  rights and lineage evidence.
- Do not add live data acquisition or model loading to tests or CI.
- A maintained successor needs a reproducible dependency lock, explicit data
  splits, leakage/fairness/calibration evaluation, lineage, and a model card.
- Stage explicit files, preserve unrelated artifacts, and run the canonical
  command before opening or updating a pull request.
