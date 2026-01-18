import re
from pathlib import Path

import pandas as pd


def parse_name(filename):
    match = re.match(r"grid__([a-z0-9_]+)__([a-z0-9_]+)\.csv$", filename)
    if not match:
        return ("unknown", "unknown")
    return match.group(1), match.group(2)


def estimate_totals(csv_path):
    df = pd.read_csv(csv_path)

    n_candidates = len(df)

    split_cols = [c for c in df.columns if re.match(r"split\d+_test_f1_macro", c)]
    if not split_cols:
        split_cols = [c for c in df.columns if re.match(r"split\d+_test_accuracy", c)]
    n_splits = len(split_cols)

    mean_fit_time = df["mean_fit_time"].mean() if "mean_fit_time" in df else None
    total_time = (
        mean_fit_time * n_splits * n_candidates if mean_fit_time is not None else None
    )

    return n_candidates, n_splits, mean_fit_time, total_time


def main():
    csv_dir = Path("outputs")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in csv_dir.glob("grid__*.csv"):
        vec, model = parse_name(csv_path.name)
        n_candidates, n_splits, mean_fit, total_time = estimate_totals(csv_path)
        rows.append(
            {
                "file": csv_path.name,
                "vectorizer": vec,
                "model": model,
                "n_candidates": n_candidates,
                "n_splits": n_splits,
                "mean_fit_time": mean_fit,
                "total_fit_time_est": total_time,
            }
        )

    results = pd.DataFrame(rows)

    print("=== Per-file ===")
    print(results.sort_values(["vectorizer", "model"]))

    print("\n=== By model ===")
    by_model = results.groupby("model")[["n_candidates", "total_fit_time_est"]].sum()
    print(by_model)
    by_model.to_csv(output_dir / "grid_time_by_model.csv")

    print("\n=== By vectorizer ===")
    by_vectorizer = results.groupby("vectorizer")[
        ["n_candidates", "total_fit_time_est"]
    ].sum()
    print(by_vectorizer)
    by_vectorizer.to_csv(output_dir / "grid_time_by_vectorizer.csv")

    print("\n=== Total ===")
    total = results[["n_candidates", "total_fit_time_est"]].sum()
    print(total)
    total.to_frame(name="value").to_csv(output_dir / "grid_time_total.csv")


if __name__ == "__main__":
    main()
