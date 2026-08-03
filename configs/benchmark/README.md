# Benchmark Configs

This tree is the canonical home for benchmark-ready experiment configs.

Current convention:

- `benchmark/kt/assist2009/`: hand-maintained KT benchmark configs
- `benchmark/kt/assist2009/sweeps/`: generated or tuned STIRKT sweep variants
- `resource/`: reusable data and model templates
- `snippets/`: shared evaluator snippets and composition fragments

Minimal run:

```bash
python main.py configs/benchmark/kt/assist2009/class1_baselines.jsonc
```

Batch benchmark:

```bash
python scripts/run_batch.py --benchmark-name assist2009_quick --datasets assist2009_raw --models DKT AKT --methods GoldModel FisherForgetting --profile quick --full-eval
```

Output path convention:

- benchmark configs now use `globals.results_root` and `globals.split_root`
- default values remain `output/runs/erasure` and `${results_root}/splits`
- you can override them in config or via `KT_RESULTS_ROOT` and `KT_SPLIT_ROOT`
