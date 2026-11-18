"""Run the whole assignment: scores, SVM loss, softmax, examples, weight images.

Usage: python run.py <cifar-10-python.tar.gz>
Writes results/*.csv, results/summary.json, results/run.log and figures/*.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import linear_classifier as lc

SEED, STD, DELTA = 42, 1e-4, 1.0
HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")

# Colors (reference data-viz palette): blue = true class / correct,
# orange = wrong prediction, gray = everything else.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b8b7b0"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 130,
})

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(msg)


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")


def main(archive):
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    X_raw, y_all, names = lc.load_cifar10_test(archive)
    X_all = lc.preprocess(X_raw)

    # 1. Initialization -----------------------------------------------------
    W, b = lc.init_params(seed=SEED, std=STD)
    log(f"W shape {W.shape}, W ~ N(0, {STD}^2), seed={SEED}; b shape {b.shape}, all zero")
    log(f"W stats: mean={W.mean():.3e}, std={W.std():.3e}, min={W.min():.3e}, max={W.max():.3e}")

    # Example images: the first test image of each class (deterministic).
    idx = np.array([np.flatnonzero(y_all == c)[0] for c in range(10)])
    X, y = X_all[idx], y_all[idx]
    log(f"example test indices: {idx.tolist()}")

    # 2. Scores and predictions --------------------------------------------
    S = lc.scores(W, b, X)
    pred = lc.predict(S)
    log("\nscores f(x) = Wx + b  (rows: examples, columns: classes)")
    log("idx   true        pred        " + " ".join(f"{n[:6]:>9}" for n in names))
    for i in range(10):
        log(f"{idx[i]:<5} {names[y[i]]:<11} {names[pred[i]]:<11} "
            + " ".join(f"{v:+9.5f}" for v in S[i]))
    write_csv(os.path.join(RES, "scores.csv"), ["test_index", "true", "pred"] + names,
              [[idx[i], names[y[i]], names[pred[i]]] + [f"{v:.8f}" for v in S[i]]
               for i in range(10)])

    # 3. Multiclass SVM loss -----------------------------------------------
    M = lc.svm_margins(S, y, DELTA)
    L_svm = M.sum(axis=1)
    log(f"\nmulticlass SVM loss (delta={DELTA})")
    for i in range(10):
        log(f"{idx[i]:<5} {names[y[i]]:<11} L_i = {L_svm[i]:.5f}")
    log(f"average SVM loss over the 10 examples: {L_svm.mean():.5f}  "
        f"(expected about C-1 = 9 when all scores are near 0)")

    # 4. Softmax ------------------------------------------------------------
    P = lc.softmax(S)
    L_ce = lc.cross_entropy_per_example(P, y)
    log("\nsoftmax probabilities")
    log("idx   " + " ".join(f"{n[:6]:>8}" for n in names) + "      sum")
    for i in range(10):
        log(f"{idx[i]:<5} " + " ".join(f"{p:8.5f}" for p in P[i]) + f"  {P[i].sum():.12f}")
    sums = P.sum(axis=1)
    log(f"all rows sum to 1 (|sum-1| max = {np.abs(sums - 1).max():.2e})")
    log(f"argmax(scores) == argmax(probabilities) for all examples: "
        f"{bool(np.array_equal(pred, P.argmax(1)))}")
    log(f"average softmax (cross-entropy) loss: {L_ce.mean():.5f}  (ln 10 = {np.log(10):.5f})")
    write_csv(os.path.join(RES, "probabilities.csv"), ["test_index", "true"] + names + ["sum"],
              [[idx[i], names[y[i]]] + [f"{p:.8f}" for p in P[i]] + [f"{sums[i]:.12f}"]
               for i in range(10)])
    write_csv(os.path.join(RES, "losses.csv"),
              ["test_index", "true", "pred", "correct", "svm_loss", "softmax_loss",
               "score_true", "score_max", "p_true", "p_max"],
              [[idx[i], names[y[i]], names[pred[i]], int(pred[i] == y[i]),
                f"{L_svm[i]:.6f}", f"{L_ce[i]:.6f}", f"{S[i, y[i]]:.8f}", f"{S[i].max():.8f}",
                f"{P[i, y[i]]:.8f}", f"{P[i].max():.8f}"] for i in range(10)])

    # 5. Correct vs misclassified, plus the whole test set for context -----
    correct = pred == y
    log(f"\ncorrect among the 10 examples: {int(correct.sum())} -> "
        f"{[names[c] for c in y[correct]]}")
    S_all = lc.scores(W, b, X_all)
    pred_all = lc.predict(S_all)
    hits_all = int((pred_all == y_all).sum())
    counts = np.bincount(pred_all, minlength=10)
    log(f"whole test set (10000 images, context only): {hits_all} correct")
    log("prediction counts on the test set: "
        + ", ".join(f"{names[c]}={counts[c]}" for c in range(10)))
    wsum = W.sum(axis=1)
    log("sum of each class's weights: "
        + ", ".join(f"{names[c]}={wsum[c]:+.5f}" for c in range(10)))
    # Share of score variance explained by mean brightness * sum(w_j).
    approx = X_all.mean(axis=1, keepdims=True) * wsum[None, :]
    r = np.corrcoef(approx.ravel(), S_all.ravel())[0, 1]
    log(f"corr(score, mean pixel * sum(w_j)) over all test scores: {r:.4f}")
    # Correct test images per class, used as extra examples if few among the 10.
    extra = [int(i) for i in np.flatnonzero(pred_all == y_all) if i not in set(idx.tolist())][:5]

    # Per true class over the whole test set (1,000 images each).
    P_all = lc.softmax(S_all)
    Ls_all = lc.svm_loss_per_example(S_all, y_all, DELTA)
    Lc_all = lc.cross_entropy_per_example(P_all, y_all)
    log("\nper true class over the whole test set")
    log("class       n     correct  svm_loss  softmax_loss  p_true    top predictions")
    per_class_rows = []
    for c in range(10):
        m = y_all == c
        cnt = np.bincount(pred_all[m], minlength=10)
        top = [t for t in np.argsort(-cnt, kind="stable")[:3] if cnt[t] > 0]
        top_str = " ".join(f"{names[t]}={cnt[t]}" for t in top)
        row = [names[c], int(m.sum()), int(cnt[c]), Ls_all[m].mean(), Lc_all[m].mean(),
               P_all[m, c].mean(), top_str]
        per_class_rows.append(row)
        log(f"{row[0]:<11} {row[1]:<5} {row[2]:<8} {row[3]:8.4f}  {row[4]:12.4f}  "
            f"{row[5]:.5f}   {top_str}")
    log(f"{'all':<11} {len(y_all):<5} {hits_all:<8} {Ls_all.mean():8.4f}  {Lc_all.mean():12.4f}")
    write_csv(os.path.join(RES, "per_class.csv"),
              ["true_class", "n", "correct", "svm_loss", "softmax_loss", "p_true", "top_predictions"],
              [[r_[0], r_[1], r_[2], f"{r_[3]:.6f}", f"{r_[4]:.6f}", f"{r_[5]:.6f}", r_[6]]
               for r_ in per_class_rows])

    # 6. Weight images ------------------------------------------------------
    means = np.stack([X_all[y_all == c].mean(axis=0) for c in range(10)])
    corr_wm = [float(np.corrcoef(W[c], means[c])[0, 1]) for c in range(10)]
    log("\ncorrelation between W_c and the class-c mean image: "
        + ", ".join(f"{names[c]}={corr_wm[c]:+.4f}" for c in range(10)))
    # Neighboring-pixel correlation: high for natural images, ~0 for noise.
    def neighbor_corr(v):
        img = lc.to_image(v)
        return float(np.corrcoef(img[:, :-1].ravel(), img[:, 1:].ravel())[0, 1])
    nc_w = [neighbor_corr(W[c]) for c in range(10)]
    nc_m = [neighbor_corr(means[c]) for c in range(10)]
    log(f"mean horizontal neighbor-pixel correlation: W rows {np.mean(nc_w):+.4f}, "
        f"class mean images {np.mean(nc_m):+.4f}")

    # Supplementary: how the weight scale changes scores and softmax.
    log("\nsupplementary: effect of the weight scale (same seed, same 10 examples)")
    log("std       |score| mean   SVM loss   max prob   softmax loss")
    scale_rows = []
    for std in (1e-4, 1e-3, 1e-2, 1e-1):
        Wk, bk = lc.init_params(seed=SEED, std=std)
        Sk = lc.scores(Wk, bk, X)
        Pk = lc.softmax(Sk)
        row = [std, np.abs(Sk).mean(), lc.svm_loss_per_example(Sk, y, DELTA).mean(),
               Pk.max(axis=1).mean(), lc.cross_entropy_per_example(Pk, y).mean()]
        scale_rows.append(row)
        log(f"{std:<8g}  {row[1]:12.5f}  {row[2]:9.4f}  {row[3]:9.4f}  {row[4]:12.4f}")
    write_csv(os.path.join(RES, "weight_scale.csv"),
              ["std", "mean_abs_score", "svm_loss", "mean_max_prob", "softmax_loss"],
              [[f"{v:.6g}" for v in r_] for r_ in scale_rows])

    summary = {
        "seed": SEED, "weight_std": STD, "delta": DELTA, "example_test_indices": idx.tolist(),
        "average_svm_loss": float(L_svm.mean()), "average_softmax_loss": float(L_ce.mean()),
        "prob_row_sums": sums.tolist(), "correct_examples": int(correct.sum()),
        "test_set_correct": hits_all, "test_set_prediction_counts": dict(zip(names, counts.tolist())),
        "weight_sums": dict(zip(names, wsum.tolist())),
        "corr_score_vs_brightness_times_weightsum": float(r),
        "corr_W_vs_class_mean": dict(zip(names, corr_wm)),
        "neighbor_corr_W": float(np.mean(nc_w)), "neighbor_corr_class_mean": float(np.mean(nc_m)),
        "extra_correct_test_indices": extra,
    }
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RES, "run.log"), "w") as f:
        f.write("\n".join(LOG) + "\n")

    # ---------------------------------------------------------------- figures
    fig_scores(X_raw[idx], S, P, y, pred, names)
    fig_correct_wrong(X_raw, idx, y, pred, S_all, y_all, pred_all, extra, names)
    fig_weights(W, means, names)
    fig_prediction_counts(counts, wsum, names)
    log("figures written to figures/")


def fig_scores(imgs, S, P, y, pred, names):
    """One row per example: image | scores | softmax probabilities."""
    fig, axes = plt.subplots(10, 3, figsize=(9.5, 19),
                             gridspec_kw={"width_ratios": [0.55, 2, 2]})
    xs = np.arange(10)
    for i in range(10):
        a0, a1, a2 = axes[i]
        a0.imshow(lc.to_image(imgs[i]), interpolation="nearest")
        a0.set_xticks([]); a0.set_yticks([]); a0.grid(False)
        ok = pred[i] == y[i]
        a0.set_title(f"true: {names[y[i]]}\npred: {names[pred[i]]}", fontsize=8,
                     color=BLUE if ok else ORANGE)
        colors = [GRAY] * 10
        colors[pred[i]] = ORANGE
        colors[y[i]] = BLUE  # true class wins when the prediction is correct
        for ax, vals in ((a1, S[i]), (a2, P[i])):
            ax.bar(xs, vals, width=0.7, color=colors)
            ax.set_xticks(xs)
            ax.set_xticklabels([n[:4] for n in names] if i == 9 else [], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
        a1.axhline(0, color=INK2, lw=0.6)
        a2.axhline(0.1, color=INK2, lw=0.6, ls="--")
        pad = 0.15 * (P[i].max() - P[i].min())
        a2.set_ylim(P[i].min() - pad, P[i].max() + pad)  # zoomed: all bars are ~0.1
        a2.ticklabel_format(axis="y", useOffset=False)
    axes[0, 1].set_title("scores  s = Wx + b", fontsize=9, color=INK)
    axes[0, 2].set_title("softmax probabilities (zoomed, dashed = 0.1)", fontsize=9, color=INK)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
                        plt.Rectangle((0, 0), 1, 1, color=GRAY)],
               labels=["true class", "predicted class (wrong)", "other classes"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(FIG, "scores_and_probabilities.png"))
    plt.close(fig)


def fig_correct_wrong(X_raw, idx, y, pred, S_all, y_all, pred_all, extra, names):
    """Top row: the 10 chosen examples marked correct/wrong.
    Bottom row: extra correctly classified test images, since random weights
    rarely hit the right class among only 10 examples."""
    fig, axes = plt.subplots(2, 10, figsize=(12, 3.4))
    for i in range(10):
        ax = axes[0, i]
        ok = pred[i] == y[i]
        ax.imshow(lc.to_image(X_raw[idx[i]]), interpolation="nearest")
        ax.set_title(f"{'CORRECT' if ok else 'WRONG'}\n{names[y[i]]} → {names[pred[i]]}",
                     fontsize=7, color=BLUE if ok else ORANGE)
    for k in range(10):
        ax = axes[1, k]
        if k < len(extra):
            j = extra[k]
            ax.imshow(lc.to_image(X_raw[j]), interpolation="nearest")
            ax.set_title(f"CORRECT #{j}\n{names[y_all[j]]} → {names[pred_all[j]]}",
                         fontsize=7, color=BLUE)
        else:
            ax.axis("off")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_ylabel("10 examples", fontsize=8)
    axes[1, 0].set_ylabel("other correct\ntest images", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "correct_vs_misclassified.png"))
    plt.close(fig)


def _normalize(v):
    img = lc.to_image(v)
    return (img - img.min()) / (img.max() - img.min())


def fig_weights(W, means, names):
    """Each W_c as a 32x32x3 image (min-max scaled), with class mean images below
    as a reference for what a structured template would look like."""
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for c in range(10):
        axes[0, c].imshow(_normalize(W[c]), interpolation="nearest")
        axes[0, c].set_title(names[c], fontsize=8)
        axes[1, c].imshow(_normalize(means[c]), interpolation="nearest")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_ylabel("random W_c", fontsize=8)
    axes[1, 0].set_ylabel("class mean\n(reference)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "weights.png"))
    plt.close(fig)


def fig_prediction_counts(counts, wsum, names):
    """Two separate charts (no dual axis): prediction counts and weight sums."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    xs = np.arange(10)
    top = int(np.argmax(counts))
    a1.bar(xs, counts, width=0.7, color=[BLUE if c == top else GRAY for c in xs])
    a1.axhline(1000, color=INK2, lw=0.6, ls="--")
    a1.set_title("predictions per class on 10,000 test images (dashed = true 1,000)",
                 fontsize=9, color=INK)
    a2.bar(xs, wsum, width=0.7, color=[BLUE if c == top else GRAY for c in xs])
    a2.axhline(0, color=INK2, lw=0.6)
    a2.set_title("sum of weights  Σ_k W[c, k]", fontsize=9, color=INK)
    for ax in (a1, a2):
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "prediction_counts.png"))
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
