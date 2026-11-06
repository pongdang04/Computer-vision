"""K-Nearest Neighbors on CIFAR-10, implemented from scratch with NumPy.

scikit-learn is NOT used here; test_knn.py uses it only as an independent check.

Pixels stay as integers 0-255. L1, L2 and Linf rankings do not change under the
usual rescaling to [0, 1], cosine is scale-invariant, and integer arithmetic keeps
the distances exact, so no rounding error can reorder neighbors.
"""
from __future__ import annotations

import io
import os
import pickle
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

METRICS = ("l1", "l2", "cosine", "linf")
_THREADS = os.cpu_count() or 4
_TRAIN_BLOCK = 1024  # rows per block for L1/Linf, sized to stay cache-friendly


# ---------------------------------------------------------------- data loading
class _SafeUnpickler(pickle.Unpickler):
    """CIFAR-10 ships as pickles; allow only the NumPy types they actually use."""

    _ALLOWED = {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")
        return super().find_class(module, name)


def load_cifar10(archive: str):
    """Read cifar-10-python.tar.gz directly, without extracting it to disk.

    Returns X_train, y_train, X_test, y_test, class_names.
    X has shape (n, 3072) and dtype uint8; y has dtype int64.
    """
    batches, names = {}, None
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            base = os.path.basename(member.name)
            if base.startswith("data_batch_") or base == "test_batch" or base == "batches.meta":
                content = tar.extractfile(member).read()
                obj = _SafeUnpickler(io.BytesIO(content), encoding="bytes").load()
                if base == "batches.meta":
                    names = [n.decode() for n in obj[b"label_names"]]
                else:
                    batches[base] = (np.asarray(obj[b"data"], np.uint8),
                                     np.asarray(obj[b"labels"], np.int64))
    train = [batches[f"data_batch_{i}"] for i in range(1, 6)]
    return (np.concatenate([t[0] for t in train]), np.concatenate([t[1] for t in train]),
            batches["test_batch"][0], batches["test_batch"][1], names)


# ------------------------------------------------------------------- distances
def _dot_based(train: np.ndarray, queries: np.ndarray, metric: str, batch: int = 512):
    """Yield (start, distance block) for L2 and cosine using matrix products.

    L2 yields squared distance, which has the same neighbor order as L2.
    """
    tr = train.astype(np.float64)
    tr_sq = np.einsum("ij,ij->i", tr, tr)
    if metric == "cosine":
        tr_norm = np.sqrt(tr_sq)
        tr_norm[tr_norm == 0] = 1.0
    for start in range(0, len(queries), batch):
        q = queries[start:start + batch].astype(np.float64)
        # Accelerate BLAS can raise false floating-point flags on macOS. test_knn.py
        # compares this product with a BLAS-free computation, and the check below
        # rejects any non-finite value, so the warning is silenced only here.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            product = q @ tr.T
        if not np.isfinite(product).all():
            raise FloatingPointError("non-finite value in matrix product")
        q_sq = np.einsum("ij,ij->i", q, q)
        if metric == "l2":
            yield start, q_sq[:, None] + tr_sq[None, :] - 2.0 * product
        else:
            q_norm = np.sqrt(q_sq)
            q_norm[q_norm == 0] = 1.0
            yield start, 1.0 - product / (q_norm[:, None] * tr_norm[None, :])


def _elementwise_one(train: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    """L1 or Linf distance from one query to every training row."""
    out = np.empty(len(train), dtype=np.float32)
    for start in range(0, len(train), _TRAIN_BLOCK):
        block = train[start:start + _TRAIN_BLOCK]
        # |a - b| on uint8 without overflow: max(a, b) - min(a, b)
        diff = np.maximum(block, query) - np.minimum(block, query)
        if metric == "l1":
            out[start:start + len(block)] = diff.sum(axis=1, dtype=np.uint32)
        else:
            out[start:start + len(block)] = diff.max(axis=1)
    return out


def _top_k(dist: np.ndarray, k: int):
    """Indices and distances of the k smallest values per row, nearest first."""
    k = min(k, dist.shape[1])
    if k == dist.shape[1]:
        part = np.tile(np.arange(k), (len(dist), 1))
    else:
        part = np.sort(np.argpartition(dist, k - 1, axis=1)[:, :k], axis=1)
    part_dist = np.take_along_axis(dist, part, axis=1)
    # A stable sort over ascending indices breaks equal distances by lower index.
    order = np.argsort(part_dist, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(part_dist, order, axis=1)


def kneighbors(train: np.ndarray, queries: np.ndarray, k: int, metric: str):
    """Brute-force k nearest neighbors. Returns (indices, distances), both (m, k)."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric: {metric}")
    k = min(k, len(train))
    indices = np.empty((len(queries), k), dtype=np.int32)
    distances = np.empty((len(queries), k), dtype=np.float64)

    if metric in ("l2", "cosine"):
        for start, dist in _dot_based(train, queries, metric):
            idx, d = _top_k(dist, k)
            indices[start:start + len(idx)], distances[start:start + len(idx)] = idx, d
        if metric == "l2":
            distances = np.sqrt(np.maximum(distances, 0.0))
        return indices, distances

    def work(i):
        idx, d = _top_k(_elementwise_one(train, queries[i], metric)[None, :], k)
        indices[i], distances[i] = idx[0], d[0]

    with ThreadPoolExecutor(_THREADS) as pool:
        list(pool.map(work, range(len(queries))))
    return indices, distances


# ------------------------------------------------------------------ classifier
def vote(neighbor_labels: np.ndarray, k: int, n_classes: int = 10) -> np.ndarray:
    """Majority vote over the first k columns (neighbors are nearest first).

    A tie goes to the tied class whose closest neighbor ranks highest.
    """
    m = len(neighbor_labels)
    rows = np.arange(m)
    counts = np.zeros((m, n_classes), dtype=np.int64)
    first_rank = np.full((m, n_classes), k, dtype=np.int64)
    for rank in range(k - 1, -1, -1):
        labels = neighbor_labels[:, rank]
        counts[rows, labels] += 1
        first_rank[rows, labels] = rank
    return np.argmax(counts * (k + 1) - first_rank, axis=1)


class KNNClassifier:
    def __init__(self, k: int = 5, metric: str = "l2"):
        self.k, self.metric = k, metric

    def fit(self, X: np.ndarray, y: np.ndarray):
        # KNN has no training phase: it only stores the data.
        self.X_, self.y_ = X, y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        indices, _ = kneighbors(self.X_, X, self.k, self.metric)
        return vote(self.y_[indices], self.k)


# ------------------------------------------------------------ cross-validation
def stratified_kfold(y: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Return (train_idx, val_idx) pairs that keep the class ratio in every fold."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == c))
        fold_of[members] = np.arange(len(members)) % n_splits
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f))
            for f in range(n_splits)]


# --------------------------------------------------------------------- metrics
def confusion_matrix(y_true, y_pred, n_classes: int = 10) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    return np.bincount(n_classes * y_true + y_pred,
                       minlength=n_classes ** 2).reshape(n_classes, n_classes)


def per_class_scores(cm: np.ndarray):
    """Per-class precision, recall and F1; an undefined ratio counts as 0."""
    tp = np.diag(cm).astype(float)
    pred, actual = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def evaluate(y_true, y_pred, n_classes: int = 10) -> dict:
    """Accuracy plus macro-averaged precision, recall and F1."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision, recall, f1 = per_class_scores(cm)
    return {"accuracy": np.trace(cm) / cm.sum(), "precision": precision.mean(),
            "recall": recall.mean(), "f1": f1.mean()}
