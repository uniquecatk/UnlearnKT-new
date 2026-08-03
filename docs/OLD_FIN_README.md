# KT Unlearning Experiment Framework

## 1. Purpose

This folder packages the current end-to-end experiment framework into one place so future runs can be organized from here.

The packaged workflow is:

1. preprocess and manage KT datasets in `fin-1`
2. reuse KT models from `fin-1`
3. apply KT-specific user split strategies in `unlearning-kt/fin-1`
4. run baseline unlearning methods and `STIRKT`
5. evaluate forget, retain, test, efficiency, and privacy-related metrics

After this packaging, the main execution path is:

```bash
cd c:\Users\ksm\Desktop\pyktmu-main\fin\unlearning-kt\fin-1
python main.py configs\kt\assist2009_class1_stirkt_interleave2.jsonc
```

You can also create or modify your own JSON/JSONC config in `configs\kt\` and run it the same way.

## 2. Folder Structure

### 2.1 `fin-1`

This part provides the KT data and model side.

- `fin-1\edmine\data\KTDataProcessor.py`
  - dataset preprocessing entry used to build KT-ready inputs
- `fin-1\dataset\`
  - processed datasets already included in this package
  - includes `ASSIST2009`, `assistments12`, `assistments17`, `algebra05`, `statics`, and others
- `fin-1\edmine\model\sequential_kt_model\`
  - reusable KT model implementations
  - includes `DKT`, `AKT`, `SimpleKT`, `DKVMN`, `DTransformer`, and more
- `fin-1\edmine\evaluator\`
  - original KT evaluation utilities from pyedmine

### 2.2 `unlearning-kt\fin-1`

This is the unlearning framework side.

- `main.py`
  - main ERASURE entry; runs a config file directly
- `erasure\data\kt_splitters.py`
  - KT-specific user-level splitters added for our experiments
- `erasure\unlearners\`
  - baseline unlearning methods plus our `STIRKT`
- `erasure\evaluations\`
  - evaluation manager and KT-specific metrics
- `configs\kt\`
  - ready-to-run KT configs
- `erasure_web_backend.py`
  - web/CLI backend that builds KT runs from CSV input
- `run_pyedmine_batch.py`
  - batch runner for experiments based on processed pyedmine datasets
- `run_stirkt_sweep.py`
  - hyperparameter sweep script for the `assist2009 + class1 + STIRKT` line

### 2.3 Results

- `unlearning-kt\outputs\`
  - ERASURE-side result tables for assist2009 baseline and STIRKT runs
- `outputs\`
  - pyedmine-based cross-dataset and cross-model result summaries

## 3. Included KT Split Strategies

The KT-specific split logic is implemented in:

- `unlearning-kt\fin-1\erasure\data\kt_splitters.py`

The main strategies are:

1. `KTDataSplitterByPerformance`
   - used for `high_performance`
   - forgets top-performing users
2. `KTDataSplitterByPerformance`
   - used for `low_performance`
   - forgets bottom-performing users
3. `KTDataSplitterByParticipation`
   - used for `low_participation`
   - forgets users with low interaction counts

Also included:

4. `KTDataSplitterByUserList`
   - used for `uid_list`
   - forgets an explicit list of users

These splitters save split artifacts such as:

- forget user ids
- retain user ids
- test user ids
- split summary CSV/JSON

## 4. Included Unlearning Methods

The unlearning methods are in:

- `unlearning-kt\fin-1\erasure\unlearners\`

Important methods included in this package:

- `GoldModel.py`
- `FisherForgetting.py`
- `SelectiveSynapticDampening.py`
- `Finetuning.py`
- `NegGrad.py`
- `BadTeaching.py`
- `Scrub.py`
- `UNSIR.py`
- `STIRKT.py`

`STIRKT.py` is our KT-specific method based on selective forget-region updates and retain-side repair.

## 5. Included Evaluation

The KT evaluation path is configured through `Evaluator` and KT measures in `configs\kt\*.jsonc`.

Common KT measures already wired:

- `KTMetrics`
  - reports test / forget / retain metrics for original and unlearned models
- `KTAUS`
  - efficacy-oriented forgetting score
- `KTRelearnTime`
  - relearning difficulty after unlearning
- `KTAIN`
  - KT adaptation/influence style score
- `KTUMIA`
  - membership inference related metric
- `RunTime`
  - runtime efficiency
- `SaveValues`
  - writes final CSV output

## 6. Ready-to-Use Configs

The most relevant configs are under:

- `unlearning-kt\fin-1\configs\kt\`

Important examples:

- `assist2009_class1_baselines.jsonc`
- `assist2009_class2_baselines.jsonc`
- `assist2009_class3_baselines.jsonc`
- `assist2009_uid_baselines.jsonc`
- `assist2009_class1_stirkt_interleave2.jsonc`
- `generated_stirkt_sweep\*.jsonc`

All copied KT configs in this package already point to paths inside `fin`.

## 7. How To Run

### 7.1 Run a KT JSON config directly

```bash
cd c:\Users\ksm\Desktop\pyktmu-main\fin\unlearning-kt\fin-1
python main.py configs\kt\assist2009_class1_stirkt_interleave2.jsonc
```

### 7.2 Run pyedmine-based multi-dataset experiments

```bash
cd c:\Users\ksm\Desktop\pyktmu-main\fin\unlearning-kt\fin-1
python run_pyedmine_batch.py --datasets assistments12 assistments17 algebra05 statics --models DKT --methods STIRKT --profile formal --full-eval --output-root c:\Users\ksm\Desktop\pyktmu-main\fin\outputs\pyedmine_stirkt_dkt
```

### 7.3 Run the STIRKT sweep

```bash
cd c:\Users\ksm\Desktop\pyktmu-main\fin\unlearning-kt\fin-1
python run_stirkt_sweep.py
```

## 8. Recommended Working Rule

For future experiments, keep the workflow as:

1. put or generate KT-ready data under `fin\fin-1\dataset`
2. write or edit JSON configs under `fin\unlearning-kt\fin-1\configs\kt`
3. run through `main.py` or `run_pyedmine_batch.py`
4. collect outputs only inside `fin\unlearning-kt\outputs` or `fin\outputs`

This keeps all future results self-contained inside `fin`.
