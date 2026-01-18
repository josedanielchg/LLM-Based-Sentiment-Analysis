import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Config
# -------------------------
OUTPUTS_DIR = Path("outputs")
TEST_CSV = Path("test_df_processed.csv")

# File pattern: best__{vec}__{model}.joblib
BEST_PATTERN = re.compile(r"^best__(?P<vec>.+?)__(?P<model>.+?)\.joblib$")

VEC_ORDER = ["bow", "tfidf_word", "tfidf_char"]
MODEL_ORDER = ["mnb", "lr", "svm", "rf"]

VEC_LABELS = {
    "bow": "BoW",
    "tfidf_word": "TF-IDF (word)",
    "tfidf_char": "TF-IDF (char)",
}

MODEL_LABELS = {
    "mnb": "MultinomialNB",
    "lr": "Logistic Regression",
    "svm": "Linear SVM",
    "rf": "Random Forest",
}


# -------------------------
# Load test data
# -------------------------
test_df = pd.read_csv(TEST_CSV).dropna(subset=["processed_text", "sentiment_class"])
X_test = test_df["processed_text"].astype(str)
y_test = test_df["sentiment_class"].astype(int).values


# -------------------------
# Discover & evaluate models
# -------------------------
rows = []

for p in OUTPUTS_DIR.glob("*.joblib"):
    m = BEST_PATTERN.match(p.name)
    if not m:
        continue

    vec_key = m.group("vec")
    model_key = m.group("model")

    # Load pipeline (vectorizer + model)
    pipe = joblib.load(p)

    # Predict
    y_pred = pipe.predict(X_test)

    # Metrics (macro to be robust to imbalance)
    acc = accuracy_score(y_test, y_pred)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    rows.append({
        "file": p.name,
        "vec": vec_key,
        "model": model_key,
        "vec_label": VEC_LABELS.get(vec_key, vec_key),
        "model_label": MODEL_LABELS.get(model_key, model_key),
        "accuracy": acc,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
    })

results = pd.DataFrame(rows)

if results.empty:
    raise RuntimeError(
        f"No models found in {OUTPUTS_DIR}. Expected files like best__bow__mnb.joblib"
    )

# Enforce ordering for plotting/grouping
results["vec"] = pd.Categorical(results["vec"], categories=VEC_ORDER, ordered=True)
results["model"] = pd.Categorical(results["model"], categories=MODEL_ORDER, ordered=True)

# Also enforce readable label ordering
results["vec_label"] = pd.Categorical(
    results["vec_label"],
    categories=[VEC_LABELS[v] for v in VEC_ORDER],
    ordered=True
)
results["model_label"] = pd.Categorical(
    results["model_label"],
    categories=[MODEL_LABELS[m] for m in MODEL_ORDER],
    ordered=True
)

results = results.sort_values(["model", "vec"]).reset_index(drop=True)

# Save results table
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
summary_path = OUTPUTS_DIR / "summary_test_metrics.csv"
results.to_csv(summary_path, index=False)
print("Saved:", summary_path)


# -------------------------
# Plot: Horizontal barplot with hue (vectorization)
# -------------------------
def plot_barh_with_hue(
    df: pd.DataFrame,
    metric: str,
    title: str,
    save_path: Path | None = None
):
    """
    Horizontal seaborn barplot:
      y = model_label
      x = metric value
      hue = vec_label (vectorization)
    """

    # Style
    sns.set(style="whitegrid", font_scale=1.05)

    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=df,
        x=metric,
        y="model_label",
        hue="vec_label",
        orient="h",
        errorbar=None,       # No CI since these are single test scores
        dodge=True
    )

    ax.set_xlim(0, 1.0)
    ax.set_xlabel(metric)
    ax.set_ylabel("Model")
    ax.set_title(title)

    # Put legend in a consistent place
    ax.legend(title="Vectorization", loc="lower right")

    # Add numeric labels to each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved plot:", save_path)

    plt.show()


REPORTS_DIR = Path("reports")

# Primary metric: macro-F1
plot_barh_with_hue(
    results,
    metric="f1_macro",
    title="Test Set Comparison — Macro F1 (Hue = Vectorization)",
    save_path=REPORTS_DIR / "barh_models_hue_vectorization_f1_macro.png",
)

# Optional: accuracy and recall
plot_barh_with_hue(
    results,
    metric="accuracy",
    title="Test Set Comparison — Accuracy (Hue = Vectorization)",
    save_path=REPORTS_DIR / "barh_models_hue_vectorization_accuracy.png",
)

plot_barh_with_hue(
    results,
    metric="recall_macro",
    title="Test Set Comparison — Macro Recall (Hue = Vectorization)",
    save_path=REPORTS_DIR / "barh_models_hue_vectorization_recall_macro.png",
)
