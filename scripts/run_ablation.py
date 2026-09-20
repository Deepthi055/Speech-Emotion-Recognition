import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_cross_corpus import run_all_cross_corpus_experiments


VARIANT_ORDER = [
    "ce",
    "standard_supcon",
    "speaker_supcon",
    "corpus_supcon",
    "proposed_supcon",
]

VARIANT_TITLES = {
    "ce": "WavLM + CE Baseline",
    "standard_supcon": "Standard SupCon",
    "speaker_supcon": "Speaker-Aware SupCon",
    "corpus_supcon": "Corpus-Aware SupCon",
    "proposed_supcon": "Speaker + Corpus-Aware SupCon",
}


def load_rows(summary_path: Path) -> list[dict]:
    with summary_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"setup_key", "setup_title", "variant", "variant_title", "test_accuracy", "test_uar", "test_f1_macro"}
    missing_columns = required.difference(rows[0].keys() if rows else set())
    if missing_columns:
        raise ValueError(f"Missing columns in {summary_path}: {sorted(missing_columns)}")

    for row in rows:
        for metric in ("test_accuracy", "test_uar", "test_f1_macro", "test_f1_weighted"):
            row[metric] = float(row[metric])
    return rows


def build_summary(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    expected = {(setup, variant) for setup in {row["setup_key"] for row in rows} for variant in VARIANT_ORDER}
    actual = {(row["setup_key"], row["variant"]) for row in rows}
    missing = expected - actual
    if missing:
        missing_text = ", ".join(f"{setup}/{variant}" for setup, variant in sorted(missing))
        raise ValueError(f"Incomplete ablation results; missing: {missing_text}")

    aggregate = []
    for variant in VARIANT_ORDER:
        variant_rows = [row for row in rows if row["variant"] == variant]
        aggregate.append(
            {
                "variant": variant,
                "variant_title": variant_rows[0]["variant_title"],
                "mean_accuracy": statistics.mean(row["test_accuracy"] for row in variant_rows),
                "mean_uar": statistics.mean(row["test_uar"] for row in variant_rows),
                "mean_f1_macro": statistics.mean(row["test_f1_macro"] for row in variant_rows),
                "std_accuracy": statistics.pstdev(row["test_accuracy"] for row in variant_rows),
                "std_uar": statistics.pstdev(row["test_uar"] for row in variant_rows),
                "std_f1_macro": statistics.pstdev(row["test_f1_macro"] for row in variant_rows),
                "configurations": len(variant_rows),
            }
        )
    return rows, aggregate


def load_local_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for variant in VARIANT_ORDER:
        metrics_path = runs_dir / {
            "ce": "wavlm_ce_baseline",
            "standard_supcon": "wavlm_supcon",
            "speaker_supcon": "wavlm_speaker_supcon",
            "corpus_supcon": "wavlm_corpus_supcon",
            "proposed_supcon": "wavlm_proposed_supcon",
        }[variant] / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        with metrics_path.open() as file:
            metrics = json.load(file)
        rows.append(
            {
                "variant": variant,
                "variant_title": VARIANT_TITLES[variant],
                "best_val_uar": metrics["best_val_uar"],
                "test_accuracy": metrics["test_accuracy"],
                "test_uar": metrics["test_uar"],
                "test_f1_macro": metrics["test_f1_macro"],
                "test_f1_weighted": metrics["test_f1_weighted"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
    return rows


def build_local_summary(rows: list[dict]) -> dict:
    baseline = rows[0]
    comparisons = []
    for row in rows[1:]:
        comparisons.append(
            {
                "comparison": f"{row['variant_title']} vs {baseline['variant_title']}",
                "accuracy_difference": row["test_accuracy"] - baseline["test_accuracy"],
                "uar_difference": row["test_uar"] - baseline["test_uar"],
                "f1_macro_difference": row["test_f1_macro"] - baseline["test_f1_macro"],
                "f1_weighted_difference": row["test_f1_weighted"] - baseline["test_f1_weighted"],
            }
        )

    pairwise = [
        ("standard_supcon", "speaker_supcon"),
        ("standard_supcon", "corpus_supcon"),
        ("speaker_supcon", "proposed_supcon"),
    ]
    by_variant = {row["variant"]: row for row in rows}
    for left, right in pairwise:
        comparisons.append(
            {
                "comparison": f"{by_variant[right]['variant_title']} vs {by_variant[left]['variant_title']}",
                "accuracy_difference": by_variant[right]["test_accuracy"] - by_variant[left]["test_accuracy"],
                "uar_difference": by_variant[right]["test_uar"] - by_variant[left]["test_uar"],
                "f1_macro_difference": by_variant[right]["test_f1_macro"] - by_variant[left]["test_f1_macro"],
                "f1_weighted_difference": by_variant[right]["test_f1_weighted"] - by_variant[left]["test_f1_weighted"],
            }
        )
    return {"runs": rows, "comparisons": comparisons}


def write_local_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_results.json").open("w") as file:
        json.dump(summary, file, indent=2)

    fields = ["variant", "variant_title", "best_val_uar", "test_accuracy", "test_uar", "test_f1_macro", "test_f1_weighted"]
    with (output_dir / "ablation_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    lines = [
        "# Task 12: Ablation Study",
        "",
        "Results from the five full runs using seed 42 and the RAVDESS-only metadata available in this project.",
        "",
        "| Method | Best validation UAR | Test Accuracy | Test UAR | Test Macro-F1 | Test Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant_title']} | {row['best_val_uar']:.4f} | {row['test_accuracy']:.4f} | "
            f"{row['test_uar']:.4f} | {row['test_f1_macro']:.4f} | {row['test_f1_weighted']:.4f} |"
        )
    lines.extend(["", "## Component Differences", "", "| Comparison | Accuracy | UAR | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|---:|"])
    for comparison in summary["comparisons"]:
        lines.append(
            f"| {comparison['comparison']} | {comparison['accuracy_difference']:+.4f} | "
            f"{comparison['uar_difference']:+.4f} | {comparison['f1_macro_difference']:+.4f} | "
            f"{comparison['f1_weighted_difference']:+.4f} |"
        )
    lines.extend(["", "Confusion matrices are preserved in `ablation_results.json` for each run.", ""])
    (output_dir / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(rows: list[dict], aggregate: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_results.json").open("w") as file:
        json.dump({"runs": rows, "aggregate": aggregate}, file, indent=2)

    with (output_dir / "ablation_summary.csv").open("w", newline="") as file:
        fields = list(aggregate[0].keys())
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)

    lines = [
        "# Task 12: Ablation Study",
        "",
        "Mean and population standard deviation across the three cross-corpus test configurations.",
        "",
        "| Method | Configurations | Accuracy (mean +/- std) | UAR (mean +/- std) | Macro-F1 (mean +/- std) |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in aggregate:
        lines.append(
            f"| {result['variant_title']} | {result['configurations']} | "
            f"{result['mean_accuracy']:.4f} +/- {result['std_accuracy']:.4f} | "
            f"{result['mean_uar']:.4f} +/- {result['std_uar']:.4f} | "
            f"{result['mean_f1_macro']:.4f} +/- {result['std_f1_macro']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The component contribution is measured by comparing the proposed model with standard SupCon, speaker-aware SupCon, and corpus-aware SupCon.",
            "",
        ]
    )
    (output_dir / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize the Task 12 ablation study.")
    parser.add_argument("--run", action="store_true", help="Run all 15 cross-corpus experiments before summarizing.")
    parser.add_argument("--local-runs", action="store_true", help="Summarize the five local runs/*/metrics.json results.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true", help="Use the short cross-corpus dry run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=Path, default=Path("results/cross_corpus/cross_corpus_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/task12"))
    args = parser.parse_args()

    if args.run:
        run_all_cross_corpus_experiments(dry_run=args.dry_run, seed=args.seed)

    if args.local_runs:
        rows = load_local_rows(args.runs_dir)
        write_local_outputs(rows, build_local_summary(rows), args.output_dir)
        print(f"Wrote Task 12 local-run outputs to {args.output_dir}")
        return

    rows, aggregate = build_summary(load_rows(args.input))
    write_outputs(rows, aggregate, args.output_dir)
    print(f"Wrote Task 12 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()