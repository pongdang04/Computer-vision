"""Checks for linear_classifier.py against straightforward loop implementations.

Usage: python test_linear.py <cifar-10-python.tar.gz>
"""
import sys

import numpy as np

import linear_classifier as lc


def svm_loss_loop(s, y, delta=1.0):
    """Textbook per-example hinge loss, written with explicit loops."""
    loss = 0.0
    for j in range(len(s)):
        if j == y:
            continue
        loss += max(0.0, s[j] - s[y] + delta)
    return loss


def main(archive):
    X_raw, y, names = lc.load_cifar10_test(archive)
    assert X_raw.shape == (10000, 3072) and y.shape == (10000,) and len(names) == 10
    X = lc.preprocess(X_raw[:200])
    y = y[:200]

    W, b = lc.init_params(seed=42)
    W2, b2 = lc.init_params(seed=42)
    assert np.array_equal(W, W2) and np.array_equal(b, b2), "seed must be reproducible"
    assert W.shape == (10, 3072) and np.all(b == 0)

    # f(x) = Wx + b, one column vector at a time, must equal the batched form.
    S = lc.scores(W, b, X)
    for i in range(5):
        s_i = sum(W[:, k] * X[i, k] for k in range(X.shape[1])) + b  # no BLAS
        assert np.allclose(s_i, S[i])

    # Hinge loss: vectorized vs loop.
    L = lc.svm_loss_per_example(S, y)
    for i in range(len(y)):
        assert np.isclose(L[i], svm_loss_loop(S[i], y[i]))
    # Hand-computed example from the lecture (cat/car/frog scores).
    s = np.array([[3.2, 5.1, -1.7]])
    assert np.isclose(lc.svm_loss_per_example(s, np.array([0]))[0], 2.9)

    # Softmax: rows sum to 1, all positive, shift-invariant, order-preserving.
    P = lc.softmax(S)
    assert np.allclose(P.sum(axis=1), 1.0) and np.all(P > 0)
    assert np.allclose(lc.softmax(S + 123.0), P)
    assert np.array_equal(np.argmax(P, 1), np.argmax(S, 1))
    # Large scores must not overflow.
    assert np.all(np.isfinite(lc.softmax(np.array([[1000.0, 0.0, -1000.0]]))))

    # Image reshape: channel 0 of the row is the red plane.
    img = lc.to_image(X_raw[0])
    assert img.shape == (32, 32, 3) and img[0, 0, 0] == X_raw[0, 0] and img[0, 0, 2] == X_raw[0, 2048]

    print("all checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
