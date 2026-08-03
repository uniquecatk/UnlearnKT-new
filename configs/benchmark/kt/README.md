# KT Benchmark Configs

This directory mirrors the benchmark-oriented organization used by `ERASURE-main`.

Current layout:

- `kt/assist2009/`: canonical KT benchmark configs
- `kt/assist2009/sweeps/`: generated or tuned STIRKT sweep variants

Recommended entrypoints:

```bash
python main.py configs/benchmark/kt/assist2009/class1_baselines.jsonc
python main.py configs/benchmark/kt/assist2009/class1_stirkt.jsonc
```

Legacy compatibility notes:

- `configs/kt/` is still runnable, but it is now a compatibility layer.
- New configs and sweep outputs should land under this benchmark tree.
