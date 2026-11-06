"""Check knn.py against hand-made cases and scikit-learn (as an independent reference).

Usage: python test_knn.py <path to cifar-10-python.tar.gz>
"""
import sys

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors

import knn

X_train, y_train, X_test, y_test, names = knn.load_cifar10(sys.argv[1])

# Dataset shape and balance
assert X_train.shape == (50000, 3072) and X_test.shape == (10000, 3072)
assert (np.bincount(y_train) == 5000).all() and (np.bincount(y_test) == 1000).all()
assert names[0] == "airplane" and names[9] == "truck"

# Voting: plain majority, then tie-break toward the closer neighbor
assert knn.vote(np.array([[1, 2, 2]]), 3)[0] == 2
assert knn.vote(np.array([[3, 1, 1, 3]]), 4)[0] == 3
assert knn.vote(np.array([[1, 3, 3, 1]]), 4)[0] == 1
assert knn.vote(np.array([[7, 2, 2]]), 1)[0] == 7

# Folds: disjoint, complete, stratified
folds = knn.stratified_kfold(y_train)
assert len(folds) == 5
assert np.array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(50000))
for tr, va in folds:
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == 50000
    assert (np.bincount(y_train[va]) == 1000).all()

# Distances and neighbor sets versus scikit-learn
tr, q = X_train[:3000], X_test[:500]
for metric, sk in [("l1", "manhattan"), ("l2", "euclidean"),
                   ("cosine", "cosine"), ("linf", "chebyshev")]:
    idx, dist = knn.kneighbors(tr, q, 25, metric)
    sk_dist, sk_idx = NearestNeighbors(n_neighbors=25, metric=sk, algorithm="brute").fit(
        tr.astype(np.float64)).kneighbors(q.astype(np.float64))
    assert np.allclose(dist, sk_dist, atol=1e-9), metric
    # Linf has many exact ties, so only rows with no ties at all must match index-wise.
    mismatch = (idx != sk_idx).any(axis=1)
    no_ties = (np.diff(sk_dist, axis=1) > 1e-12).all(axis=1)
    assert not (mismatch & no_ties).any(), metric
    print(f"{metric:7s} distances match sklearn; rows differing only by ties: {mismatch.sum()}")

# L2 and cosine versus a BLAS-free computation (sklearn uses the same BLAS as knn.py)
small_tr, small_q = tr[:1000].astype(np.int64), q[:50].astype(np.int64)
for i, query in enumerate(small_q):
    dots = np.sum(small_tr * query, axis=1)  # exact integer arithmetic
    l2 = np.sqrt(np.sum((small_tr - query) ** 2, axis=1))
    cosine = 1 - dots / (np.sqrt(np.sum(small_tr ** 2, axis=1)) * np.sqrt(np.sum(query ** 2)))
    for metric, truth in (("l2", l2), ("cosine", cosine)):
        _, dist = knn.kneighbors(tr[:1000], q[i:i + 1], 1000, metric)
        assert np.allclose(dist[0], np.sort(truth), atol=1e-9), metric
print("l2/cosine match the BLAS-free computation")

# Metrics versus scikit-learn, on real predictions
for k in (1, 7, 25):
    pred = knn.KNNClassifier(k, "l1").fit(tr, y_train[:3000]).predict(q)
    mine = knn.evaluate(y_test[:500], pred)
    p, r, f, _ = precision_recall_fscore_support(y_test[:500], pred, average="macro",
                                                 zero_division=0)
    assert np.allclose([mine["accuracy"], mine["precision"], mine["recall"], mine["f1"]],
                       [accuracy_score(y_test[:500], pred), p, r, f])

print("all checks passed")
