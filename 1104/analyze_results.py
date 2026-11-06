"""Supporting numbers for the report's explanations. Output: results/analysis.json

Usage: python analyze_results.py <path to cifar-10-python.tar.gz>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import knn

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
X_train, y_train, X_test, y_test, names = knn.load_cifar10(sys.argv[1])
tr, va = knn.stratified_kfold(y_train, 5, 42)[0]
out = {}

# How concentrated are predictions on a few classes as k grows? (fold 1, validation)
share = {}
for metric in knn.METRICS:
    with np.load(os.path.join(RESULTS, "cache", f"fold1_{metric}.npz")) as saved:
        labels = y_train[tr][saved["indices"]]
    for k in (1, 5, 21, 101):
        counts = np.bincount(knn.vote(labels, k), minlength=10)
        share[f"{metric}_k{k}"] = {"top3_share": float(np.sort(counts)[-3:].sum() / counts.sum()),
                                   "most_predicted": names[int(counts.argmax())],
                                   "least_predicted_count": int(counts.min())}
out["prediction_share_fold1"] = share

# Paired per-fold differences at each metric's selected k
folds = pd.read_csv(os.path.join(RESULTS, "cv_folds.csv"))
best = {"l1": 5, "l2": 3, "cosine": 5, "linf": 1}
acc = {m: folds[(folds.metric == m) & (folds.k == k)].sort_values("fold").accuracy.values
       for m, k in best.items()}
out["fold_accuracy"] = {m: a.round(4).tolist() for m, a in acc.items()}
out["l1_beats_cosine_folds"] = int((acc["l1"] > acc["cosine"]).sum())
out["cosine_beats_l2_folds"] = int((acc["cosine"] > acc["l2"]).sum())

# Exact ties at the nearest-neighbor distance (first 300 validation images of fold 1)
queries = X_train[va[:300]]
train = X_train[tr]
ties = {}
for metric in ("l1", "linf"):
    dists = (knn._elementwise_one(train, q, metric) for q in queries)
    tied = [int((d == d.min()).sum()) for d in dists]
    ties[metric] = {"median_tied": float(np.median(tied)), "max_tied": int(np.max(tied))}
out["ties_at_nearest_distance"] = ties

# Spread of Linf distances for the same queries, and predictions in the test matrix
linf = np.stack([knn._elementwise_one(train[:5000], q, "linf") for q in queries[:100]])
out["linf_distance_percentiles"] = np.percentile(linf, [1, 50, 99]).tolist()
out["linf_distinct_values_median"] = float(np.median([len(np.unique(r)) for r in linf]))
with open(os.path.join(RESULTS, "test_best_details.json")) as f:
    cm = np.array(json.load(f)["confusion_matrix"])
out["test_predicted_counts"] = dict(zip(names, cm.sum(axis=0).tolist()))

with open(os.path.join(RESULTS, "analysis.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=1))
