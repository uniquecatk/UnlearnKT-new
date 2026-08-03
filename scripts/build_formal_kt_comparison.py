from pathlib import Path
import math

import pandas as pd


ROOT = Path(r"c:\Users\ksm\Desktop\pyktmu-main\fin3")
OUT_DIR = ROOT / "output" / "runs" / "kt" / "formal_comparison_tables"

TARGET_COMBOS = [
    ("assist2009", "DKT", "class1"),
    ("assist2009", "DKT", "class2"),
    ("assist2009", "DKT", "random20"),
    ("assist2009", "SAKT", "class1"),
    ("assist2009", "SAKT", "class2"),
    ("assist2009", "SAKT", "random20"),
    ("assistments15", "DKT", "class1"),
    ("assistments15", "DKT", "class2"),
    ("assistments15", "DKT", "random20"),
    ("assistments17", "DKT", "class1"),
    ("assistments17", "DKT", "class2"),
    ("assistments17", "DKT", "random20"),
    ("assistments17", "SAKT", "class1"),
    ("assistments17", "SAKT", "class2"),
    ("assistments17", "SAKT", "random20"),
]

BASELINE_METHODS = ["GoldModel", "FisherForgetting", "SelectiveSynapticDampening"]
FINAL_METHODS = BASELINE_METHODS + ["STIRKT", "QEFU-KT"]


def load_csv(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source: {path}")
    df = pd.read_csv(path)
    df["source_name"] = source_name
    return df


def filter_target(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for dataset, model, strategy in TARGET_COMBOS:
        mask |= (
            (df["dataset"] == dataset)
            & (df["model"] == model)
            & (df["strategy"] == strategy)
        )
    return df[mask].copy()


def build_markdown(detail: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = ["# KT Unlearning Formal Comparison", ""]
    lines.append("## Mean Summary")
    lines.append("")
    lines.append("| Method | Mean AUS | Mean AIN | Mean UMIA Gap | Mean RunTime | AUS Rank | AIN Rank | UMIA Rank | RunTime Rank |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['AUS']:.4f} | {row['AIN']:.4f} | {row['UMIA_gap']:.4f} | {row['RunTime']:.2f} | "
            f"{row['rank_AUS']:.2f} | {row['rank_AIN']:.2f} | {row['rank_UMIA']:.2f} | {row['rank_RunTime']:.2f} |"
        )

    lines.append("")
    lines.append("## Detailed Comparison")
    lines.append("")
    lines.append("| Dataset | Model | Strategy | Method | AUS | AIN | UMIA | UMIA Gap | RunTime |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in detail.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['strategy']} | {row['method']} | "
            f"{row['AUS']:.4f} | {row['AIN']:.4f} | {row['UMIA']:.4f} | {row['UMIA_gap']:.4f} | {row['RunTime']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_csv(
        ROOT / "output" / "runs" / "kt" / "paper_suite" / "merged_completed_no2012" / "paper_suite_summary.csv",
        "paper_suite_merged",
    )
    baseline = filter_target(baseline)
    baseline = baseline[baseline["method"].isin(BASELINE_METHODS)].copy()

    kt_sources = [
        load_csv(ROOT / "output" / "runs" / "kt" / "user_verify_ktfisher_gpu" / "assist2009_selected" / "paper_suite_summary.csv", "ktfisher_gpu_assist2009"),
        load_csv(ROOT / "output" / "runs" / "kt" / "user_verify_ktfisher_gpu" / "assistments15_selected" / "paper_suite_summary.csv", "ktfisher_gpu_assistments15"),
        load_csv(ROOT / "output" / "runs" / "kt" / "user_verify_ktfisher_gpu" / "assistments17_selected" / "paper_suite_summary.csv", "ktfisher_gpu_assistments17"),
    ]
    kt_df = filter_target(pd.concat(kt_sources, ignore_index=True))
    kt_df = kt_df[kt_df["method"] == "QEFU-KT"].copy()

    stirkt_sources = [
        load_csv(ROOT / "output" / "runs" / "kt" / "formal_compare_runs" / "stirkt_assist2009" / "paper_suite_summary.csv", "stirkt_assist2009"),
        load_csv(ROOT / "output" / "runs" / "kt" / "formal_compare_runs" / "stirkt_assistments15" / "paper_suite_summary.csv", "stirkt_assistments15"),
        load_csv(ROOT / "output" / "runs" / "kt" / "formal_compare_runs" / "stirkt_assistments17" / "paper_suite_summary.csv", "stirkt_assistments17"),
    ]
    stirkt_df = filter_target(pd.concat(stirkt_sources, ignore_index=True))
    stirkt_df = stirkt_df[stirkt_df["method"] == "STIRKT"].copy()

    detail = pd.concat([baseline, stirkt_df, kt_df], ignore_index=True)
    detail = detail[detail["method"].isin(FINAL_METHODS)].copy()
    detail["combo"] = list(zip(detail["dataset"], detail["model"], detail["strategy"]))
    missing = []
    for combo in TARGET_COMBOS:
        found = set(detail.loc[detail["combo"] == combo, "method"])
        needed = set(FINAL_METHODS) - found
        if needed:
            missing.append((combo, sorted(needed)))
    if missing:
        lines = []
        for combo, methods in missing:
            lines.append(f"{combo}: missing {methods}")
        raise RuntimeError("Comparison sources incomplete:\n" + "\n".join(lines))

    detail["UMIA_gap"] = (detail["UMIA"] - 0.5).abs()
    detail = detail[
        [
            "dataset",
            "model",
            "strategy",
            "method",
            "AUS",
            "AIN",
            "UMIA",
            "UMIA_gap",
            "RunTime",
            "source_name",
        ]
    ].sort_values(["dataset", "model", "strategy", "method"])

    ranked_frames = []
    for _, group in detail.groupby(["dataset", "model", "strategy"], sort=False):
        g = group.copy()
        g["rank_AUS"] = g["AUS"].rank(ascending=False, method="min")
        g["rank_AIN"] = g["AIN"].rank(ascending=False, method="min")
        g["rank_UMIA"] = g["UMIA_gap"].rank(ascending=True, method="min")
        g["rank_RunTime"] = g["RunTime"].rank(ascending=True, method="min")
        ranked_frames.append(g)
    detail = pd.concat(ranked_frames, ignore_index=True)

    summary = (
        detail.groupby("method", as_index=False)[
            ["AUS", "AIN", "UMIA_gap", "RunTime", "rank_AUS", "rank_AIN", "rank_UMIA", "rank_RunTime"]
        ]
        .mean()
        .sort_values(["rank_AUS", "rank_UMIA", "rank_RunTime"])
    )

    detail.to_csv(OUT_DIR / "kt_formal_comparison_detail.csv", index=False)
    summary.to_csv(OUT_DIR / "kt_formal_comparison_summary.csv", index=False)
    (OUT_DIR / "kt_formal_comparison.md").write_text(build_markdown(detail, summary), encoding="utf-8")

    print(f"wrote: {OUT_DIR / 'kt_formal_comparison_detail.csv'}")
    print(f"wrote: {OUT_DIR / 'kt_formal_comparison_summary.csv'}")
    print(f"wrote: {OUT_DIR / 'kt_formal_comparison.md'}")


if __name__ == "__main__":
    main()

