# Codebase Structure Guide

This note explains which parts of `fin3` are source code, which are reproducibility assets, and which are historical or generated directories.

## 1. Source Code

These directories are the actual maintainable codebase:

- `main.py`
- `scripts/`
- `src/kt_unlearn/`
- `src/kt_lib/`
- `configs/`

### Recommended reading order

If a reviewer or collaborator wants to understand the implementation quickly, use this order:

1. `README.md`
2. `main.py`
3. `scripts/kt_backend.py`
4. `scripts/run_batch.py`
5. `scripts/run_kt_paper_suite.py`
6. `src/kt_unlearn/unlearners/QEFUKT.py`
7. `src/kt_unlearn/unlearners/FisherForgetting.py`
8. `src/kt_unlearn/evaluations/erasure_bridge.py`
9. `src/kt_unlearn/model/KTModel.py`
10. `src/kt_lib/model/sequential_kt_model/`

## 2. Config Layers

### Canonical

- `configs/benchmark/kt/assist2009/`

This is the main benchmark config tree and should be the first place to add or maintain formal experiment configs.

### Compatibility Layer

- `configs/kt/`

This folder remains runnable, but it should be considered a compatibility alias for older paths. New configs should not be added here unless backward compatibility is necessary.

## 3. Data and Artifacts

### Input Data

- `data/processed_datasets/`
- `data/demo/`

These are source inputs for experiments.

### Runtime Outputs

- `output/`

This is the canonical output root and should be used for new runs.

### Legacy Runtime Outputs

- `outputs/`

This directory is kept only for compatibility with older local layouts.

## 4. Paper Material

- `paper/`

This directory is not part of the runnable framework itself. It contains manuscript sources, figures, response-letter drafts, and writing notes. It is useful for submission management, but it should not be confused with the code entry path.

## 5. Historical Notes

- `docs/OLD_FIN_README.md`
- `docs/OLD_RESULTS_SUMMARY.md`

These are archive notes, not the current repository guide.

## 6. Cleanup Rules

To keep the repository organized going forward:

1. put new runnable logic in `src/` or `scripts/`
2. put new formal configs in `configs/benchmark/`
3. put new runtime outputs in `output/`
4. keep `outputs/` only as a compatibility path
5. avoid adding generated caches, `.pyc`, or temporary logs to source directories
6. avoid moving legacy artifact directories unless every config and script path is updated consistently
