# Experiment Results Summary

## 1. Current Scope

This summary collects the main experiment lines completed so far:

1. assist2009 baseline forgetting experiments under different user split strategies
2. assist2009 `STIRKT` design and hyperparameter sweep
3. multi-dataset `DKT + STIRKT`
4. started multi-model extension with `AKT` and `SimpleKT`

## 2. Split Strategy Baselines

Main file:

- `unlearning-kt\outputs\kt_erasure_baselines_summary.csv`

Summary by strategy:

| strategy | method | test AUC | forget AUC | retain AUC |
| --- | --- | ---: | ---: | ---: |
| class1 | SelectiveSynapticDampening | 0.7719 | 0.7250 | 0.7570 |
| class1 | FisherForgetting | 0.6387 | 0.5768 | 0.6139 |
| class2 | SelectiveSynapticDampening | 0.7616 | 0.7547 | 0.7529 |
| class2 | FisherForgetting | 0.5985 | 0.6140 | 0.6156 |
| class3 | SelectiveSynapticDampening | 0.7586 | 0.7887 | 0.7396 |
| class3 | FisherForgetting | 0.6001 | 0.6337 | 0.6068 |
| uid | SelectiveSynapticDampening | 0.7443 | 0.8534 | 0.7590 |
| uid | FisherForgetting | 0.6056 | 0.7032 | 0.6271 |

Interpretation:

- `SelectiveSynapticDampening` is generally more stable on utility
- `FisherForgetting` usually forgets harder than SSD but damages utility more
- these baselines motivated the move toward a selective-and-repair style method

## 3. assist2009 STIRKT Main Variants

Main files:

- `unlearning-kt\outputs\assist2009_class1_stirkt_interleave.csv`
- `unlearning-kt\outputs\assist2009_class1_stirkt_interleave2.csv`

Key results:

| variant | test AUC | forget AUC | retain AUC |
| --- | ---: | ---: | ---: |
| interleave | 0.6820 | 0.0072 | 0.7009 |
| interleave2 | 0.7228 | 0.0154 | 0.7489 |

Interpretation:

- `interleave` forgets most aggressively
- `interleave2` gives the best overall balance and became the main working candidate

## 4. STIRKT Sweep

Main file:

- `unlearning-kt\outputs\assist2009_class1_stirkt_sweep_summary.csv`

Top sweep results:

| variant | alpha | mask_ratio | forget_steps | repair | test AUC | forget AUC | retain AUC | balance_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sweep_b_more_repair | 0.0025 | 0.025 | 3 | 3 | 0.7232 | 0.0155 | 0.7620 | 1.3333 |
| sweep_f_stronger_mask_weaker_step | 0.0022 | 0.030 | 3 | 2 | 0.7372 | 0.0339 | 0.7601 | 1.3283 |
| sweep_c_stronger_alpha | 0.0030 | 0.025 | 3 | 2 | 0.7262 | 0.0177 | 0.7509 | 1.3280 |

Interpretation:

- `sweep_b_more_repair` is the current best balanced result
- stronger repair is important for KT utility retention
- stronger masking can improve utility but may slightly weaken forgetting

## 5. Multi-Dataset DKT + STIRKT

Main file:

- `outputs\pyedmine_stirkt_dkt\batch_summary.csv`

Summary:

| dataset | test AUC | forget AUC | retain AUC |
| --- | ---: | ---: | ---: |
| assistments12 | 0.7313 | 0.6770 | 0.7540 |
| assistments17 | 0.6522 | 0.5993 | 0.6529 |
| algebra05 | 0.6733 | 0.2478 | 0.6983 |
| statics | 0.6244 | 0.5363 | 0.6422 |

Interpretation:

- `STIRKT` already shows cross-dataset portability
- `algebra05` is currently the strongest cross-dataset forgetting case
- the method is not yet uniformly optimal with one shared hyperparameter set across all datasets

## 6. Multi-Model Extension

Main files:

- `outputs\pyedmine_stirkt_akt_simplekt\batch_summary.csv`
- `outputs\pyedmine_stirkt_multimodel\batch_summary.csv`

Confirmed example:

| dataset | model | test AUC | forget AUC | retain AUC |
| --- | --- | ---: | ---: | ---: |
| assistments12 | AKT | 0.7280 | 0.6902 | 0.7296 |

Interpretation:

- `STIRKT` is no longer limited to `DKT`
- `AKT / SimpleKT` runs have been started and some results are already stored
- this branch still needs more tuning before claiming stable superiority

## 7. Best Current Conclusion

The strongest current claim is:

- on `assist2009 + class1`, `STIRKT` provides a better forgetting-utility balance than the current `SSD/Fisher` baselines

The best current configuration is:

- `assist2009 + class1 + STIRKT + sweep_b_more_repair`

The main unfinished part is:

- validating the same advantage consistently across more splits, datasets, and model architectures

## 8. Important Result Files

ERASURE-side:

- `unlearning-kt\outputs\kt_erasure_baselines_summary.csv`
- `unlearning-kt\outputs\assist2009_class1_stirkt_interleave2.csv`
- `unlearning-kt\outputs\assist2009_class1_stirkt_sweep_summary.csv`

pyedmine cross-dataset side:

- `outputs\pyedmine_stirkt_dkt\batch_summary.csv`
- `outputs\pyedmine_stirkt_akt_simplekt\batch_summary.csv`
- `outputs\pyedmine_stirkt_multimodel\batch_summary.csv`

Combined summaries:

- `outputs\pyedmine_batch_runs_combined.csv`
- `outputs\pyedmine_batch_runs_combined_v2.csv`
- `outputs\pyedmine_formal_core_combined.csv`
