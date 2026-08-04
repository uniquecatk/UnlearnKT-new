Place the official raw dataset files in this directory before running `scripts/prepare_pyedmine_dataset.py`.

Expected examples:

- `data/raw_datasets/assist2009/skill_builder_data.csv`
- `data/raw_datasets/assist2012/2012-2013-data-with-predictions-4-final.csv`
- `data/raw_datasets/assistments15/2015_100_skill_builders_main_problems.csv`
- `data/raw_datasets/assistments17/anonymized_full_release_competition_dataset.csv`
- `data/raw_datasets/statics2011/AllData_student_step_2011F.csv`
- `data/raw_datasets/ednet-kt1/users_*.csv`

For EdNet-KT1, generate the `users_*.csv` chunks first with:

```bash
python scripts/prepare_pyedmine_dataset.py prepare-ednet-raw --dataset-src-dir <EdNet-KT1-user-dir> --contents-dir <EdNet-contents-dir>
```
