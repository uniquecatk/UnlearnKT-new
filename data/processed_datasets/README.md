Place the prepared PyEdmine-style outputs in this directory.

Expected examples:

- `data/processed_datasets/ASSIST2009/data.txt`
- `data/processed_datasets/assist2012/data.txt`
- `data/processed_datasets/assistments15/data.txt`
- `data/processed_datasets/assistments17/data.txt`
- `data/processed_datasets/statics2011/data.txt`
- `data/processed_datasets/ednet-kt1/data.txt`

These folders are the intermediate step. Convert them into runtime sequence CSV files with:

```bash
python scripts/kt_dataset_bridge.py convert --dataset-name assist2009
```

Keep large local datasets out of git. Only placeholder files should remain tracked.
