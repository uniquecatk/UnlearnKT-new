# ASSIST2009 Benchmarks

This directory contains the repository's main KT unlearning benchmark configs.

Groups:

- `class1_*`, `class2_*`, `class3_*`: class-based forgetting benchmarks
- `uid_*`: explicit user-list forgetting benchmarks
- `stirkt_*`: STIRKT variants and tuned ablations
- `sweeps/`: generated STIRKT sweep candidates

Minimal examples:

```bash
python main.py configs/benchmark/kt/assist2009/class1_baselines.jsonc
python main.py configs/benchmark/kt/assist2009/class1_stirkt.jsonc
```

Notes:

- Benchmark CSV outputs default to `globals.results_root`.
- Split artifacts default to `globals.split_root`.
- Defaults remain `output/runs/erasure` and `${results_root}/splits`.
- Batch-generated reproducibility manifests are written by `scripts/run_batch.py`.
