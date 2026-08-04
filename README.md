# UnlearnKT

Machine unlearning framework for privacy-preserving knowledge tracing, built on top of `ERASURE-main`.

---

## Environment Requirements

- **Python**: 3.10+ (developed and tested on Python 3.12.3)
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **CUDA**: 12.1+ recommended (CPU-only mode also supported)
- **Hardware** (for reference; the development machine): Intel Core i5-12600KF (16 logical cores), 32 GB RAM, NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)

---

## Installation

```bash
pip install -r requirements.txt
```

All package versions in `requirements.txt` are pinned to ensure reproducibility. For PyTorch, install the CUDA-compatible build if you have a GPU:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

The framework is driven by JSONC configuration files. A single config defines everything: dataset, model, forgetting strategy, and evaluation metrics.

```bash
python main.py configs/example.jsonc
python main.py configs/proof_of_concept.jsonc
```

The core paper method (`QEFU-KT`) is implemented in `src/kt_unlearn/unlearners/QEFUKT.py`.

### Dataset Preparation

Dataset sources can be referenced here: [https://uniquecatk.github.io/UnlearnKT-new/](https://uniquecatk.github.io/UnlearnKT-new/)

This repository now uses one unified KT data chain:

`official raw data -> PyEdmine-style preprocessing in this repo -> data/processed_datasets -> kt_dataset_bridge.py -> sequence csv -> experiments`

Choose one setup path only.

#### Path A. Directly Use Prepared Dataset Zip Packages

Recommended for `assist2009`, `assist2012`, `assistments15`, `assistments17`, and `statics2011`.

- Place the zip files under `data/processed_datasets_zip/`
- Extract them into `data/processed_datasets/`
- Run `scripts/kt_dataset_bridge.py`
- Start experiments

Before extraction, the folder layout under `data/` should look like this:

```text
data/
└── processed_datasets_zip/
    ├── assist2009-*.zip
    ├── assist2012-*.zip
    ├── assist2017-*.zip
    ├── assistments15.zip
    └── statics2011-*.zip
```

After extraction, `data/processed_datasets/` should look like this:

```text
data/
└── processed_datasets/
    ├── assist2009/
    ├── assist2012/
    ├── assist2015/
    ├── assist2017/
    ├── statics2011/
    ├── ...
    └── <dataset-folder>/
        ├── data.txt
        ├── Q_table.npy
        ├── statics_raw.json
        ├── statics_preprocessed.json
        └── *_map.csv
```

For `ednet-kt1`, keep using [Google Drive](https://drive.google.com/drive/folders/14ZLY7B_Tgs8k82qW3eQD7ufcHh0Bq50W) or a GitHub Release asset, then extract it into `data/processed_datasets/ednet-kt1/`.

Commands for Path A:

```bash
python scripts/kt_dataset_bridge.py convert --dataset-name assist2009
python scripts/kt_dataset_bridge.py convert --dataset-name assist2012
python scripts/kt_dataset_bridge.py convert --dataset-name assistments15
python scripts/kt_dataset_bridge.py convert --dataset-name assistments17
python scripts/kt_dataset_bridge.py convert --dataset-name statics2011
python scripts/kt_dataset_bridge.py convert --dataset-name ednet-kt1
```

#### Path B. Prepare from Official Raw Data

- Place the raw dataset files under `data/raw_datasets/`
- Run `scripts/prepare_pyedmine_dataset.py`
- Then run `scripts/kt_dataset_bridge.py`

The folder layout under `data/` should be:

```text
data/
├── raw_datasets/
│   ├── assist2009/
│   │   └── skill_builder_data.csv
│   ├── assist2012/
│   │   └── 2012-2013-data-with-predictions-4-final.csv
│   ├── assistments15/
│   │   └── 2015_100_skill_builders_main_problems.csv
│   ├── assistments17/
│   │   └── anonymized_full_release_competition_dataset.csv
│   ├── statics2011/
│   │   └── AllData_student_step_2011F.csv
│   └── ednet-kt1/
│       └── users_*.csv
```

Commands for Path B:

```bash
python scripts/prepare_pyedmine_dataset.py preprocess --dataset-name assist2009 --run-bridge
python scripts/prepare_pyedmine_dataset.py preprocess --dataset-name assist2012 --run-bridge
python scripts/prepare_pyedmine_dataset.py preprocess --dataset-name assistments15 --run-bridge
python scripts/prepare_pyedmine_dataset.py preprocess --dataset-name assistments17 --run-bridge
python scripts/prepare_pyedmine_dataset.py preprocess --dataset-name statics2011 --run-bridge
```

For EdNet-KT1, first aggregate the raw user logs and then run preprocessing:

```bash
python scripts/prepare_pyedmine_dataset.py prepare-ednet-raw --dataset-src-dir <EdNet-KT1-user-dir> --contents-dir <EdNet-contents-dir>
python scripts/prepare_pyedmine_dataset.py preprocess --dataset-name ednet-kt1 --run-bridge
```

The bridge outputs the final experiment inputs here:

```text
output/runs/kt/pyedmine_converted/
├── assist2009_sequences.csv
├── assistments15_sequences.csv
├── assistments17_sequences.csv
├── assist2012_sequences.csv
├── statics2011_sequences.csv
└── ednet-kt1_sequences.csv
```

---

## Dataset Format

The runtime input of this repository is the sequence CSV generated by `scripts/kt_dataset_bridge.py`.

Required columns are:
- `uid` — user ID
- `questions` — comma-separated question ID sequence
- `responses` — comma-separated correctness sequence
- `fold` — split indicator
- `concepts` — optional comma-separated concept ID sequence

The framework's `KTFileDataSource` in `src/kt_unlearn/data/data_sources/KTFileDataSource.py` reads this format directly. `data/processed_datasets/` stores the intermediate PyEdmine-style outputs, while `output/runs/kt/pyedmine_converted/` stores the final sequence CSV files consumed by the experiment runners.

---

## Configuration Files (JSONC)

After local dataset preparation, the main paper benchmark can be launched with:

```bash
python scripts/run_qefu_full_batch_suite.py \
  --datasets assist2009 assistments15 assistments17 assist2012 statics2011 ednet-kt1 \
  --models DKT SAKT AKT DKVMN \
  --strategies class1 class2 random20 \
  --methods GoldModel Finetuning FisherForgetting SelectiveSynapticDampening NegGrad AdvancedNegGrad QEFU-KT \
  --benchmark-name my_benchmark --overwrite
```

This generates every combination of dataset x model x strategy x method (6 x 4 x 3 x 7 = **504 experiments**). A full round takes approximately 30 hours on the reference machine (Intel i5-12600KF, 32 GB RAM, RTX 4060 Ti 16 GB). Each flag maps to a specific experimental dimension and a corresponding JSONC config:

### Dataset (`--datasets`)

Six public KT datasets of varying scale (from ~3K to ~900K interactions):

| Flag | Dataset | Interactions |
|---|---|---|
| `assist2009` | ASSIST2009 | ~400K |
| `assistments15` | ASSIST2015 | ~700K |
| `assistments17` | ASSIST2017 | ~900K |
| `assist2012` | ASSIST2012 | ~2.7M |
| `statics2011` | STATICS2011 | ~360K |
| `ednet-kt1` | EdNet-KT1 | ~100M |

Each dataset should be prepared locally under `data/processed_datasets/` and converted into sequence CSV files under `output/runs/kt/pyedmine_converted/` before launching the batch scripts.

### KT Model (`--models`)

Four representative KT backbones:

| Flag | Model | Description |
|---|---|---|
| `DKT` | Deep Knowledge Tracing | LSTM-based sequential prediction |
| `SAKT` | Self-Attentive KT | Transformer-based attention over past interactions |
| `AKT` | Attentive Knowledge Tracing | Attention-based KT with Rasch model embeddings |
| `DKVMN` | Dynamic Key-Value Memory Network | Memory-augmented sequential KT model |

Each model is declared in the JSONC `predictor` field. Implementations are in `src/kt_lib/`.

### Forgetting Strategy (`--strategies`)

Three **user-level** deletion scenarios, mirroring realistic learner-level deletion requests. Each strategy has a set of JSONC configs under `configs/benchmark/kt/assist2009/`:

| Flag | Strategy | What is forgotten | Baseline config | QEFU-KT config |
|---|---|---|---|---|
| `class1` | Class-1 (high-perf.) | All interactions of the **high-performance** user group | `class1_baselines.jsonc` | `class1_forget.jsonc` |
| `class2` | Class-2 (low-perf.) | All interactions of the **low-performance** user group | `class2_baselines.jsonc` | `class2_forget.jsonc` |
| `random20` | Random-20% | A **random 20%** of users | `class3_baselines.jsonc` | `class3_forget.jsonc` |

These correspond to the paper's three forgetting classes (Class-1, Class-2, Class-3), covering distinct deletion-request patterns: removing top-performing learners, struggling learners, or a random subset.

### Unlearning Method (`--methods`)

Seven compared methods. Each is declared in the JSONC `forget_unlearning` field:

| Flag | Method | Description |
|---|---|---|
| `GoldModel` | Gold Model | Retrain from scratch on only the retain set (retraining reference) |
| `Finetuning` | Fine-tuning | Continue training on the retain set |
| `NegGrad` | Negative Gradient | Gradient ascent on the forget set |
| `AdvancedNegGrad` | Advanced NegGrad | Gradient ascent on forget set + descent on retain set |
| `FisherForgetting` | Fisher Forgetting | Perturb weights according to Fisher information matrix |
| `SelectiveSynapticDampening` | SSD | Selectively dampen Fisher-important parameters |
| `QEFU-KT` | QEFU-KT (ours) | KT-adapted Fisher forgetting with question-aware weight modulation |

Baseline methods (`GoldModel`, `Finetuning`, `NegGrad`, `AdvancedNegGrad`, `FisherForgetting`, `SelectiveSynapticDampening`) are grouped in `*_baselines.jsonc` configs. `QEFU-KT` has its own `*_forget.jsonc` configs with method-specific hyperparameters (target_strength, exposure_power, ascent_steps, etc.).

### Config File Structure

Each JSONC config is a self-contained experiment specification with four top-level fields:

| Field | Purpose |
|---|---|
| `data` | Dataset source, split strategy, batch size (e.g., `KTFileDataSource` + `KTDataSplitterByFold`) |
| `predictor` | KT model type and hyperparameters (e.g., `DKT`, `SAKT`, `AKT`) |
| `forget_unlearning` | Forgetting method and its parameters |
| `compose_kt_eval` | Reference to an evaluation snippet under `configs/snippets/` that injects metrics (AUC, AUS, UMIA, AIN, RunTime) |

Configs also support a `globals` section for overriding output paths (`results_root`, `split_root`).

### Supporting Config Directories

| Directory | Purpose |
|---|---|
| `configs/examples/` | Quick-start examples: `assist2009_demo.jsonc` (lightweight) and `assist2009_full_eval.jsonc` (full metrics). Verify with `python main.py configs/examples/assist2009_demo.jsonc` |
| `configs/resource/kt/` | Reusable templates (e.g., `assist2009_dkt.jsonc` — shared data/model definition) |
| `configs/snippets/` | Evaluation snippets: `kt_demo_evaluation.json` (lightweight), `kt_full_evaluation.json` (all paper metrics), `kt_erasure_bridge_evaluation.json` (cross-framework bridge) |
| `configs/uid_lists/` | User ID lists for user-level forgetting experiments |

### Hyperparameter Sweep Configs

QEFU-KT hyperparameter sweeps use 6 pre-defined configs in `configs/benchmark/kt/assist2009/sweeps/`:

| File | Parameter varied |
|---|---|
| `sweep_a_conservative.jsonc` | Conservative defaults (baseline reference) |
| `sweep_b_more_repair.jsonc` | More repair epochs |
| `sweep_c_stronger_alpha.jsonc` | Higher forgetting learning rate (alpha) |
| `sweep_d_smaller_mask.jsonc` | Smaller mask ratio (sparser updates) |
| `sweep_e_more_anchor.jsonc` | Stronger anchor weight |
| `sweep_f_stronger_mask_weaker_step.jsonc` | Stronger masking + smaller step size |

Run with: `python scripts/run_qefukt_sweep.py`

---

## Scripts

### Paper-style Suite

```bash
python scripts/run_qefu_full_batch_suite.py \
  --datasets assist2009 assistments15 assistments17 assist2012 statics2011 ednet-kt1 \
  --models DKT SAKT AKT DKVMN \
  --strategies class1 class2 random20 \
  --methods GoldModel Finetuning FisherForgetting SelectiveSynapticDampening NegGrad AdvancedNegGrad QEFU-KT \
  --benchmark-name my_benchmark --overwrite
```

Runs the main paper benchmark (6 datasets x 4 models x 3 strategies x 7 methods = 504 combinations). Generates a summary CSV and Markdown table with all evaluation metrics. Before launching it, make sure the local dataset folders and converted sequence CSVs are prepared as described above.

For smaller legacy comparisons on four datasets, the repository also keeps:

```bash
python scripts/run_kt_paper_suite.py \
  --datasets assist2009 assistments12 assistments15 assistments17 \
  --models DKT SAKT DKVMN \
  --strategies class1 class2 random20
```

### Sweep Runner

```bash
python scripts/run_qefukt_sweep.py
```

Performs hyperparameter sweeps for the QEFU-KT method. Iterates over 6 predefined sweep configurations varying alpha (forgetting learning rate), mask_ratio (sparsity), forget_steps, repair_epochs, and anchor_weight. Outputs a ranked summary CSV sorted by composite balance score.

Sweep configurations are stored in `configs/benchmark/kt/assist2009/sweeps/`.

### Other Scripts

- `scripts/run_batch.py` — general-purpose batch experiment runner
- `scripts/run_experiment.py` — single experiment launcher
- `scripts/kt_backend.py` — backend logic for KT experiments
- `scripts/merge_kt_paper_suite.py` — merge and aggregate paper suite results

---

## Built-in Unlearners

| Method | Description |
|---|---|
| GoldModel | Retrain from scratch on retain set |
| Finetuning | Fine-tune on retain set |
| NegGrad / AdvancedNegGrad | Gradient ascent on forget set |
| FisherForgetting | Fisher-information-based weight perturbation |
| QEFU-KT | KT-adapted Fisher forgetting (proposed method) |
| SelectiveSynapticDampening | Selectively dampen Fisher-important parameters |
| Scrub | Teacher-student unlearning (SCRUB) |
| BadTeaching | Mismatched-label unlearning |

---

## Repository Structure

```
UnlearnKT-new/
├── src/kt_unlearn/      # KT unlearning framework (forgetters, evaluators, data pipeline)
├── src/kt_lib/          # KT model library (30+ KT models, trainers, evaluators)
├── configs/             # JSONC experiment configurations
├── scripts/             # Batch/sweep/paper-suite runners
├── data/demo/           # Small demo dataset tracked in Git
├── data/processed_datasets/  # Local processed datasets (not committed)
├── docs/                # Additional documentation
├── main.py              # Entry point
└── requirements.txt     # Pinned dependencies
```

## License

This project is released for research purposes. See the paper for details.
