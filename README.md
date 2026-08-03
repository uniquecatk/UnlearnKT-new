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

For reviewers and readers, the recommended setup path is to directly download the prepared dataset files from [Google Drive](https://drive.google.com/drive/folders/14ZLY7B_Tgs8k82qW3eQD7ufcHh0Bq50W) and extract them into:

`data/processed_datasets/`

The data directory layout used by the current repository is:

```text
data/
├── demo/
│   └── assist2009_demo_sequences.csv
└── processed_datasets/
    ├── ASSIST2009/
    │   └── skill_builder_data.csv
    ├── assist2012/
    │   ├── data.txt
    │   ├── Q_table.npy
    │   └── *_map.csv
    ├── assistments15/
    │   ├── preprocessed_data.csv
    │   ├── preprocessed_data_train.csv
    │   └── preprocessed_data_test.csv
    ├── assistments17/
    │   ├── preprocessed_data.csv
    │   ├── preprocessed_data_train.csv
    │   └── preprocessed_data_test.csv
    ├── statics2011/
    │   ├── data.txt
    │   ├── Q_table.npy
    │   └── *_map.csv
    └── ednet-kt1/
        ├── data.txt
        ├── Q_table.npy
        └── *_map.csv
```

For the paper experiments provided in this repository, no separate preprocessing or dataset splitting step is required at runtime. Download the prepared dataset files, place them under `data/processed_datasets/`, and then run the provided experiment scripts directly.

---

## Dataset Format

This repository expects the prepared dataset files to be placed directly in the local data directories before running experiments.

### Layer 1: Local Data Folders

These folders live under `data/processed_datasets/` and are read directly by the experiment scripts.

#### 1) Row-table format

Datasets: `ASSIST2009`, `assistments12`, `assistments15`, `assistments17`, `algebra05`, `bridge_algebra06`, `spanish`, `statics`

Common files:
- `skill_builder_data.csv` or `preprocessed_data.csv`
- `preprocessed_data_train.csv` / `preprocessed_data_test.csv`
- `q_mat.npz` for datasets that provide a Q-matrix

Common row-level columns in the prepared files:
- `user_id` or `uid`
- `item_id`, `problem_id`, or `question_id`
- `correct`, `response`, or `is_correct`
- optional: `skill_id`, `timestamp`

#### 2) Text-based processed format

Datasets: `assist2012`, `statics2011`, `ednet-kt1`

Each dataset directory contains:
- `data.txt` — tab-separated file with columns:
  - Column 0: number of interactions in the sequence
  - Column 1: space-separated question IDs
  - Column 2: space-separated correctness values (0/1)
- `Q_table.npy` — Q-matrix (questions × concepts)
- `*_map.csv` — ID mapping files (user, question, concept)

These text-based folders should be provided in their benchmark-ready form inside the prepared dataset package.

### Layer 2: Sequence CSV for Direct Runs

Direct JSONC runs use a compact sequence CSV such as `data/demo/assist2009_demo_sequences.csv`.

Required columns:
- `uid` — user ID
- `questions` — comma-separated question ID sequence
- `responses` — comma-separated correctness sequence
- `fold` — split indicator
- `concepts` — optional comma-separated concept ID sequence

The framework's `KTFileDataSource` in `src/kt_unlearn/data/data_sources/KTFileDataSource.py` reads this compact sequence format directly.

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

Each dataset should be prepared locally under `data/processed_datasets/`, and the corresponding JSONC config points to the required CSV or text file (see [Dataset Format](#dataset-format)).

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
