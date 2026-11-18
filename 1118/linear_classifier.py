"""Linear classifier f(x) = Wx + b on CIFAR-10 with randomly initialized weights.

Everything is implemented with NumPy only. No training is performed: W is drawn
from a small Gaussian with a fixed seed and b is zero, as the assignment asks.
"""
from __future__ import annotations

import io
import os
import pickle
import tarfile

import numpy as np

NUM_CLASSES = 10
DIM = 32 * 32 * 3  # 3072


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


def load_cifar10_test(archive: str):
    """Read the test batch and class names from cifar-10-python.tar.gz.

    Returns X (10000, 3072) uint8, y (10000,) int64, class_names (list of 10 str).
    Each row of X is laid out as 1024 red, 1024 green, then 1024 blue values.
    """
    X = y = names = None
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            base = os.path.basename(member.name)
            if base not in ("test_batch", "batches.meta"):
                continue
            obj = _SafeUnpickler(io.BytesIO(tar.extractfile(member).read()),
                                 encoding="bytes").load()
            if base == "batches.meta":
                names = [n.decode() for n in obj[b"label_names"]]
            else:
                X = np.asarray(obj[b"data"], np.uint8)
                y = np.asarray(obj[b"labels"], np.int64)
    return X, y, names


def preprocess(X_uint8: np.ndarray) -> np.ndarray:
    """Scale raw pixels 0-255 to [0, 1] as float64."""
    return X_uint8.astype(np.float64) / 255.0


def to_image(row: np.ndarray) -> np.ndarray:
    """(3072,) CIFAR row -> (32, 32, 3) HWC image."""
    return row.reshape(3, 32, 32).transpose(1, 2, 0)


# ------------------------------------------------------------------- the model
def init_params(seed: int = 42, std: float = 1e-4,
                num_classes: int = NUM_CLASSES, dim: int = DIM):
    """W ~ N(0, std^2) with shape (C, D); b = 0 with shape (C,)."""
    rng = np.random.default_rng(seed)
    W = std * rng.standard_normal((num_classes, dim))
    b = np.zeros(num_classes)
    return W, b


def scores(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """f(x) = Wx + b for every row x of X. Returns (N, C).

    For a single column vector x this is exactly W @ x + b; stacking N images as
    rows of X gives X @ W.T + b.
    """
    # NumPy's Accelerate BLAS on Apple Silicon can raise spurious floating-point
    # warnings on matmul even when the result is correct. test_linear.py compares
    # this against per-vector products, and the finiteness check below rejects any
    # real overflow, so the warning is silenced only for this product.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = X @ W.T + b
    if not np.all(np.isfinite(S)):
        raise FloatingPointError("non-finite class scores")
    return S


def predict(S: np.ndarray) -> np.ndarray:
    """Predicted class = index of the highest score."""
    return np.argmax(S, axis=1)


# ---------------------------------------------------------------------- losses
def svm_loss_per_example(S: np.ndarray, y: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Multiclass SVM (hinge) loss L_i = sum_{j != y_i} max(0, s_j - s_{y_i} + delta)."""
    n = len(y)
    correct = S[np.arange(n), y][:, None]
    margins = np.maximum(0.0, S - correct + delta)
    margins[np.arange(n), y] = 0.0  # the j == y_i term is excluded
    return margins.sum(axis=1)


def svm_margins(S: np.ndarray, y: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Per-class margin terms max(0, s_j - s_y + delta), with 0 at j == y. (N, C)."""
    n = len(y)
    m = np.maximum(0.0, S - S[np.arange(n), y][:, None] + delta)
    m[np.arange(n), y] = 0.0
    return m


def softmax(S: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Subtracting the row max first avoids overflow and does
    not change the result, because softmax(s + c) = softmax(s)."""
    Z = S - S.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy_per_example(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Softmax (cross-entropy) loss L_i = -log p_{y_i}."""
    return -np.log(P[np.arange(len(y)), y])
