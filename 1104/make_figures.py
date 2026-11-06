"""Draw the report figures from the CSV/JSON files in results/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = {"l1": ("#2a78d6", "L1 (Manhattan)", "o"), "l2": ("#eb6834", "L2 (Euclidean)", "s"),
          "cosine": ("#1baf7a", "Cosine", "^"), "linf": ("#eda100", "L∞ (Chebyshev)", "D")}
MEASURES = {"accuracy": "Accuracy", "precision": "Macro precision",
            "recall": "Macro recall", "f1": "Macro F1"}

plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 200,
})


def plot_k_curves(summary):
    """One small multiple per measure, sharing one k axis and one legend."""
    ks = sorted(summary["k"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (measure, title) in zip(axes.flat, MEASURES.items()):
        for metric, (color, label, marker) in SERIES.items():
            d = summary[summary["metric"] == metric].sort_values("k")
            mean, std = d[f"{measure}_mean"], d[f"{measure}_std"]
            ax.fill_between(d["k"], mean - std, mean + std, color=color, alpha=0.18, lw=0)
            ax.plot(d["k"], mean, color=color, marker=marker, ms=5, lw=2, label=label)
        ax.set_title(title, loc="left")
        ax.set_xscale("log")
        ax.set_xticks(ks[::2] + [ks[-1]] if ks[-1] not in ks[::2] else ks[::2])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for ax in axes[1]:
        ax.set_xlabel("k (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIGURES, "cv_k_metric.png"))
    plt.close(fig)


def plot_confusion(details):
    cm = np.array(details["confusion_matrix"])
    names = details["classes"]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.grid(False)
    ax.set_xticks(range(10), names, rotation=45, ha="right")
    ax.set_yticks(range(10), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Test confusion matrix: {SERIES[details['metric']][1]}, k={details['k']}",
                 loc="left")
    threshold = cm.max() * 0.55
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Images")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "confusion_matrix.png"))
    plt.close(fig)


def plot_per_class(details):
    """Precision and recall per class, sorted by F1, to expose their imbalance."""
    names = np.array(details["classes"])
    precision, recall, f1 = (np.array(details[key]) for key in ("precision", "recall", "f1"))
    order = np.argsort(f1)
    y = np.arange(10)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.hlines(y, precision[order], recall[order], color=GRID, lw=2, zorder=1)
    ax.scatter(precision[order], y, s=60, color=SERIES["l1"][0], marker="o",
               edgecolor="white", linewidth=1.5, label="Precision", zorder=2)
    ax.scatter(recall[order], y, s=60, color=SERIES["l2"][0], marker="s",
               edgecolor="white", linewidth=1.5, label="Recall", zorder=2)
    ax.set_yticks(y, [f"{n}  (F1 {f:.2f})" for n, f in zip(names[order], f1[order])])
    ax.set_xlabel("Score")
    ax.set_title("Test precision and recall per class", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "per_class.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    plot_k_curves(pd.read_csv(os.path.join(RESULTS, "cv_summary.csv")))
    with open(os.path.join(RESULTS, "test_best_details.json")) as f:
        best = json.load(f)
    plot_confusion(best)
    plot_per_class(best)
    print("figures written to", FIGURES)
