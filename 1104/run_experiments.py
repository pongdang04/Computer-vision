"""5-fold CV over k and distance metric, then a final evaluation on the test set.

Usage: python run_experiments.py <path to cifar-10-python.tar.gz>

Model selection uses the 50,000 training images only. The test set is used after
selection, so it cannot influence which k is chosen. Neighbor lists are cached in
results/cache, so an interrupted run resumes where it stopped.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import knn

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101]
K_MAX = max(K_VALUES)
SEED = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "cache")


def neighbors_cached(name, train, queries, metric):
    """Return the K_MAX nearest neighbor indices and the seconds spent computing them."""
    path = os.path.join(CACHE, f"{name}_{metric}.npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as saved:
            return saved["indices"], float(saved["seconds"])
    start = time.perf_counter()
    indices, _ = knn.kneighbors(train, queries, K_MAX, metric)
    seconds = time.perf_counter() - start
    np.savez(path, indices=indices, seconds=seconds)
    print(f"  {name} {metric}: {seconds:.1f}s", flush=True)
    return indices, seconds


def main(archive):
    os.makedirs(CACHE, exist_ok=True)
    X_train, y_train, X_test, y_test, names = knn.load_cifar10(archive)

    # ---- 5-fold cross-validation on the training set
    rows, timing = [], []
    for fold, (tr, va) in enumerate(knn.stratified_kfold(y_train, 5, SEED), start=1):
        print(f"fold {fold}/5", flush=True)
        for metric in knn.METRICS:
            indices, seconds = neighbors_cached(f"fold{fold}", X_train[tr], X_train[va], metric)
            timing.append({"metric": metric, "fold": fold, "seconds": seconds})
            neighbor_labels = y_train[tr][indices]
            for k in K_VALUES:
                scores = knn.evaluate(y_train[va], knn.vote(neighbor_labels, k))
                rows.append({"metric": metric, "k": k, "fold": fold, **scores})

    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(RESULTS, "cv_folds.csv"), index=False)
    summary = folds.groupby(["metric", "k"])[["accuracy", "precision", "recall", "f1"]].agg(
        ["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(RESULTS, "cv_summary.csv"), index=False)
    pd.DataFrame(timing).groupby("metric")["seconds"].mean().to_csv(
        os.path.join(RESULTS, "cv_seconds_per_fold.csv"))

    # ---- Select k for each metric by mean CV macro-F1
    best = summary.loc[summary.groupby("metric")["f1_mean"].idxmax()]
    overall = best.loc[best["f1_mean"].idxmax()]
    print(best[["metric", "k", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]])

    # ---- Test set: train on all 50,000 images with each metric's selected k
    test_rows, details = [], {}
    for _, row in best.iterrows():
        metric, k = row["metric"], int(row["k"])
        indices, seconds = neighbors_cached("test", X_train, X_test, metric)
        pred = knn.vote(y_train[indices], k)
        test_rows.append({"metric": metric, "k": k, **knn.evaluate(y_test, pred),
                          "seconds": seconds})
        if metric == overall["metric"]:
            cm = knn.confusion_matrix(y_test, pred)
            precision, recall, f1 = knn.per_class_scores(cm)
            details = {"metric": metric, "k": k, "classes": names,
                       "confusion_matrix": cm.tolist(), "precision": precision.tolist(),
                       "recall": recall.tolist(), "f1": f1.tolist()}
    pd.DataFrame(test_rows).to_csv(os.path.join(RESULTS, "test_results.csv"), index=False)
    with open(os.path.join(RESULTS, "test_best_details.json"), "w") as f:
        json.dump(details, f, indent=2)
    print(pd.DataFrame(test_rows))


if __name__ == "__main__":
    main(sys.argv[1])
