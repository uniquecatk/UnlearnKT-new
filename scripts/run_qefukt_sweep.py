import csv
import json
import subprocess
from copy import deepcopy
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent
SRC_ROOT = FRAMEWORK_ROOT / "src"
ERASURE_ROOT = FRAMEWORK_ROOT.parent / "ERASURE-main"
for import_root in (SRC_ROOT, ERASURE_ROOT):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

BASE_CONFIG = FRAMEWORK_ROOT / "configs" / "benchmark" / "kt" / "assist2009" / "qefukt_class1.jsonc"
SWEEP_DIR = FRAMEWORK_ROOT / "configs" / "benchmark" / "kt" / "assist2009" / "sweeps"


VARIANTS = [
    {
        "name": "sweep_a_conservative",
        "alpha": 0.0020,
        "mask_ratio": 0.022,
        "forget_steps": 3,
        "step_repair_epochs": 2,
        "anchor_weight": 0.0015,
        "max_update_norm": 0.015,
    },
    {
        "name": "sweep_b_more_repair",
        "alpha": 0.0025,
        "mask_ratio": 0.025,
        "forget_steps": 3,
        "step_repair_epochs": 3,
        "anchor_weight": 0.0015,
        "max_update_norm": 0.020,
    },
    {
        "name": "sweep_c_stronger_alpha",
        "alpha": 0.0030,
        "mask_ratio": 0.025,
        "forget_steps": 3,
        "step_repair_epochs": 2,
        "anchor_weight": 0.0015,
        "max_update_norm": 0.018,
    },
    {
        "name": "sweep_d_smaller_mask",
        "alpha": 0.0025,
        "mask_ratio": 0.020,
        "forget_steps": 3,
        "step_repair_epochs": 2,
        "anchor_weight": 0.0015,
        "max_update_norm": 0.020,
    },
    {
        "name": "sweep_e_more_anchor",
        "alpha": 0.0025,
        "mask_ratio": 0.025,
        "forget_steps": 3,
        "step_repair_epochs": 2,
        "anchor_weight": 0.0020,
        "max_update_norm": 0.020,
    },
    {
        "name": "sweep_f_stronger_mask_weaker_step",
        "alpha": 0.0022,
        "mask_ratio": 0.030,
        "forget_steps": 3,
        "step_repair_epochs": 2,
        "anchor_weight": 0.0018,
        "max_update_norm": 0.015,
    },
]


def build_variant(base_cfg, variant):
    cfg = deepcopy(base_cfg)
    globals_cfg = cfg.setdefault("globals", {})
    globals_cfg.setdefault("results_root", "output/runs/erasure")
    globals_cfg.setdefault("split_root", "${results_root}/splits")
    params = cfg["unlearners"][0]["parameters"]
    params["alpha"] = variant["alpha"]
    params["mask_ratio"] = variant["mask_ratio"]
    params["forget_steps"] = variant["forget_steps"]
    params["step_repair_epochs"] = variant["step_repair_epochs"]
    params["anchor_weight"] = variant["anchor_weight"]
    params["max_update_norm"] = variant["max_update_norm"]
    params["forget_optimizer"]["parameters"]["lr"] = variant["alpha"]

    split_base = f"assist2009_class1_{variant['name']}"
    cfg["data"]["parameters"]["partitions"][1]["parameters"]["artifact_dir"] = f"${{split_root}}/{split_base}"
    cfg["data"]["parameters"]["partitions"][2]["parameters"]["artifact_dir"] = f"${{split_root}}/{split_base}_retain_test"
    cfg["evaluator"]["parameters"]["measures"][-1]["parameters"]["path"] = f"${{results_root}}/{split_base}.csv"
    return cfg


def run_variant(config_path: Path):
    subprocess.run(
        [sys.executable, str(FRAMEWORK_ROOT / "main.py"), str(config_path)],
        cwd=FRAMEWORK_ROOT,
        check=True,
    )


def read_single_row(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def score_row(row):
    test_auc = float(row["kt.auc.test.unlearned"])
    retain_auc = float(row["kt.auc.retain.unlearned"])
    forget_auc = float(row["kt.auc.forget.unlearned"])
    utility = 0.5 * (test_auc + retain_auc)
    forget_gain = 1.0 - forget_auc
    balance = utility + 0.6 * forget_gain
    return utility, forget_gain, balance


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        base_cfg = json.load(f)
    results_root = FRAMEWORK_ROOT / base_cfg.get("globals", {}).get("results_root", "output/runs/erasure")
    summary_path = results_root / "assist2009_class1_qefukt_sweep_summary.csv"

    summary_rows = []

    for variant in VARIANTS:
        cfg = build_variant(base_cfg, variant)
        config_path = SWEEP_DIR / f"{variant['name']}.jsonc"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=True, indent=2)
        run_variant(config_path)

        result_csv = results_root / f"assist2009_class1_{variant['name']}.csv"
        row = read_single_row(result_csv)
        utility, forget_gain, balance = score_row(row)
        summary_rows.append(
            {
                "variant": variant["name"],
                "alpha": variant["alpha"],
                "mask_ratio": variant["mask_ratio"],
                "forget_steps": variant["forget_steps"],
                "step_repair_epochs": variant["step_repair_epochs"],
                "anchor_weight": variant["anchor_weight"],
                "max_update_norm": variant["max_update_norm"],
                "test_auc": row["kt.auc.test.unlearned"],
                "forget_auc": row["kt.auc.forget.unlearned"],
                "retain_auc": row["kt.auc.retain.unlearned"],
                "AUS": row["AUS"],
                "RelearnTime": row["RelearnTime"],
                "AIN": row["AIN"],
                "UMIA_AUC": row["UMIA_AUC"],
                "utility_score": f"{utility:.6f}",
                "forget_gain": f"{forget_gain:.6f}",
                "balance_score": f"{balance:.6f}",
            }
        )

    summary_rows.sort(key=lambda x: float(x["balance_score"]), reverse=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_path}")
    for row in summary_rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
