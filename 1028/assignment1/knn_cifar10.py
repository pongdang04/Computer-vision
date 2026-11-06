"""
Assignment: CIFAR-10 kNN 이미지 분류 (Week 7-1)

사용법
    python knn_cifar10.py                              # 학습 5000장 / 테스트 500장 (기본)
    python knn_cifar10.py --num-train 50000 --num-test 1000
    python knn_cifar10.py --skip-cv --k 7 --metric L2   # 교차검증 없이 바로 테스트

데이터
    data/ 폴더에 CIFAR-10 binary 버전이 없으면 자동으로 내려받음 (약 162MB)
    https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    레코드 = 라벨 1byte + R(1024) G(1024) B(1024) byte, 32x32 행 우선

결과: results/ 폴더
    cifar10_samples.png     슬라이드 3 왼쪽: 클래스별 학습 이미지 10장
    nearest_neighbors.png   슬라이드 3 오른쪽 / 15: 테스트 이미지 -> 가까운 학습 이미지 10장
                            테두리 초록 = 테스트 이미지와 같은 클래스, 빨강 = 다른 클래스
                            맨 오른쪽 = k-NN 예측 결과 (정답이면 초록, 틀리면 빨강)
    cross_validation.png    슬라이드 27: k 별 5-fold 교차검증 정확도 (L1 / L2)
    cv_accuracy.csv         metric, k, fold 별 정확도
    confusion_matrix.png    최종 (metric, k) 로 테스트셋을 분류한 분류표
    test_result.csv         클래스별 테스트 정확도

정의 (슬라이드 4, 7, 13, 16, 24~26)
    L1 거리  d(I1, I2) = sum_p |I1_p - I2_p|
    L2 거리  d(I1, I2) = sqrt( sum_p (I1_p - I2_p)^2 )
    train    학습 데이터를 그대로 저장(memorize)
    predict  가장 가까운 k 개 학습 이미지의 라벨로 다수결
             동점이면 동점인 클래스 중 가장 가까운 이웃의 클래스를 선택
    하이퍼파라미터 (k, metric) 는 학습 데이터 안에서 5-fold 교차검증으로 고르고
    테스트셋은 마지막에 한 번만 사용
"""
import argparse
import csv
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
K_CHOICES = [1, 3, 5, 7, 10, 15, 20, 50, 100]
NUM_FOLDS = 5

GREEN = "#2ca02c"
RED = "#d62728"


# ---------------------------------------------------------------- 데이터
def download_cifar10():
    batch_dir = DATA_DIR / "cifar-10-batches-bin"
    if (batch_dir / "test_batch.bin").exists():
        return batch_dir
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "cifar-10-binary.tar.gz"
    print(f"CIFAR-10 다운로드: {CIFAR_URL}")
    for attempt in range(20):
        # 연결이 멈추면 timeout 후 받은 곳부터 이어받기 (Range 요청)
        have = tar_path.stat().st_size if tar_path.exists() else 0
        req = urllib.request.Request(CIFAR_URL, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r, open(tar_path, "ab") as f:
                if have and r.status != 206:  # 서버가 이어받기를 지원하지 않으면 처음부터
                    f.truncate(0)
                total = have + int(r.headers["Content-Length"])
                while chunk := r.read(1 << 16):
                    f.write(chunk)
                    have += len(chunk)
                    print(f"\r  {have / 2**20:.1f} / {total / 2**20:.1f} MB", end="", flush=True)
            print()
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:  # 이미 다 받음
                break
            raise
        except OSError as e:
            print(f"\n  연결 끊김 ({e}), 이어받기 재시도 {attempt + 1}")
    with tarfile.open(tar_path) as tar:
        # Python 3.9 에는 extractall(filter=) 가 없으므로 data/ 안의 일반 파일/폴더만 풀기
        members = [m for m in tar.getmembers()
                   if (m.isfile() or m.isdir())
                   and (DATA_DIR / m.name).resolve().is_relative_to(DATA_DIR.resolve())]
        tar.extractall(DATA_DIR, members=members)
    return batch_dir


def read_batch(path):
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 1 + 3072)
    labels = raw[:, 0].astype(np.int64)
    images = raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # N x 32 x 32 x 3
    return images, labels


def load_cifar10():
    batch_dir = download_cifar10()
    parts = [read_batch(batch_dir / f"data_batch_{i}.bin") for i in range(1, 6)]
    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test, y_test = read_batch(batch_dir / "test_batch.bin")
    return x_train, y_train, x_test, y_test


def flatten(images):
    return images.reshape(len(images), -1).astype(np.float32)


# ---------------------------------------------------------------- kNN
class KNearestNeighbor:
    def __init__(self, metric="L1"):
        self.metric = metric

    def train(self, x, y):
        # 슬라이드 6: 학습은 데이터를 기억하는 것뿐
        self.x_train = x
        self.y_train = y

    def compute_distances(self, x, chunk=8):
        """테스트 x (M x D) 와 학습 데이터 (N x D) 사이 거리 행렬 M x N"""
        if self.metric == "L2":
            # (a - b)^2 = a^2 + b^2 - 2ab 로 한 번에 계산 (완전 벡터화)
            a2 = np.sum(x.astype(np.float64) ** 2, axis=1, keepdims=True)
            b2 = np.sum(self.x_train.astype(np.float64) ** 2, axis=1)
            ab = x.astype(np.float64) @ self.x_train.T.astype(np.float64)
            return np.sqrt(np.maximum(a2 + b2 - 2 * ab, 0))
        # L1 은 분해가 안 되므로 테스트 이미지를 몇 장씩 묶어 broadcasting
        dists = np.empty((len(x), len(self.x_train)), dtype=np.float32)
        for s in range(0, len(x), chunk):
            diff = x[s:s + chunk, None, :] - self.x_train[None, :, :]
            dists[s:s + chunk] = np.abs(diff).sum(axis=2)
        return dists

    def predict_from_dists(self, dists, k):
        order = np.argsort(dists, axis=1, kind="stable")[:, :k]
        neighbor_labels = self.y_train[order]
        pred = np.empty(len(dists), dtype=np.int64)
        for i, labels in enumerate(neighbor_labels):
            votes = np.bincount(labels, minlength=len(CLASSES))
            tied = np.flatnonzero(votes == votes.max())
            # 동점이면 가장 가까운 이웃 순서대로 보고 동점 클래스 중 먼저 나온 것
            pred[i] = next(l for l in labels if l in tied)
        return pred

    def predict(self, x, k):
        return self.predict_from_dists(self.compute_distances(x), k)


# ---------------------------------------------------------------- 교차검증
def cross_validate(x, y, metrics, k_choices, num_folds):
    """슬라이드 26: 학습 데이터를 fold 로 나눠 하나씩 validation 으로 사용"""
    x_folds = np.array_split(x, num_folds)
    y_folds = np.array_split(y, num_folds)
    acc = {m: {k: [] for k in k_choices} for m in metrics}
    for m in metrics:
        for f in range(num_folds):
            x_tr = np.concatenate(x_folds[:f] + x_folds[f + 1:])
            y_tr = np.concatenate(y_folds[:f] + y_folds[f + 1:])
            knn = KNearestNeighbor(m)
            knn.train(x_tr, y_tr)
            dists = knn.compute_distances(x_folds[f])  # 거리는 한 번만 계산하고 k 만 바꿈
            for k in k_choices:
                a = np.mean(knn.predict_from_dists(dists, k) == y_folds[f])
                acc[m][k].append(a)
            print(f"  [{m}] fold {f + 1}/{num_folds}: "
                  + ", ".join(f"k={k}:{acc[m][k][-1]:.3f}" for k in k_choices))
    return acc


# ---------------------------------------------------------------- 그림
def show(ax, img, border=None):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if border:
        ax.add_patch(Rectangle((-0.5, -0.5), 32, 32, fill=False,
                               edgecolor=border, linewidth=4))


def plot_samples(x, y, path, per_class=10, seed=0):
    """슬라이드 3 왼쪽: 클래스별 예시 이미지"""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), per_class + 1,
                             figsize=(per_class + 2, len(CLASSES) * 1.05),
                             gridspec_kw={"width_ratios": [2.2] + [1] * per_class})
    for c, name in enumerate(CLASSES):
        idx = rng.choice(np.flatnonzero(y == c), per_class, replace=False)
        axes[c, 0].axis("off")
        axes[c, 0].text(1, 0.5, name, ha="right", va="center",
                        fontsize=13, fontweight="bold", family="serif")
        for j, i in enumerate(idx):
            show(axes[c, j + 1], x[i])
    fig.suptitle("Example Dataset: CIFAR-10", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nearest_neighbors(knn, x_test_img, x_test, y_test, x_train_img, k, path,
                           num_neighbors=10):
    """슬라이드 3 오른쪽 / 15: 테스트 이미지 -> 가까운 학습 이미지 순서대로"""
    # 클래스마다 테스트 이미지 하나씩 (테스트셋 순서상 첫 번째)
    rows = [np.flatnonzero(y_test == c)[0] for c in range(len(CLASSES))]
    dists = knn.compute_distances(x_test[rows])
    order = np.argsort(dists, axis=1, kind="stable")[:, :num_neighbors]
    pred = knn.predict_from_dists(dists, k)

    ncols = num_neighbors + 3  # 테스트, 화살표, 이웃들, 예측
    fig, axes = plt.subplots(len(rows), ncols, figsize=(ncols * 1.05 + 1.2, len(rows) * 1.12),
                             gridspec_kw={"width_ratios": [1, 0.6] + [1] * num_neighbors + [2.2]})
    for r, (ti, nbrs) in enumerate(zip(rows, order)):
        true = y_test[ti]
        show(axes[r, 0], x_test_img[ti])
        axes[r, 0].set_ylabel(CLASSES[true], rotation=0, ha="right", va="center", fontsize=10)
        axes[r, 1].axis("off")
        axes[r, 1].annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", lw=3, color="black",
                                            mutation_scale=18))
        for j, ni in enumerate(nbrs):
            same = knn.y_train[ni] == true
            show(axes[r, j + 2], x_train_img[ni], border=GREEN if same else RED)
        ax = axes[r, -1]
        ax.axis("off")
        ok = pred[r] == true
        ax.text(0.1, 0.5, f"{'O' if ok else 'X'}  {CLASSES[pred[r]]}", va="center",
                fontsize=11, fontweight="bold", color=GREEN if ok else RED)
    axes[0, 0].set_title("test", fontsize=10)
    for j in range(num_neighbors):
        axes[0, j + 2].set_title(f"#{j + 1}", fontsize=10)
    axes[0, -1].set_title(f"{k}-NN pred", fontsize=10)
    fig.suptitle(f"Test images and nearest neighbors ({knn.metric} distance)  "
                 f"green = same class, red = different class", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cross_validation(acc, path):
    """슬라이드 27: k 별 fold 정확도 점 + 평균 ± 표준편차"""
    metrics = list(acc)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        ks = list(acc[m])
        for k in ks:
            ax.scatter([k] * len(acc[m][k]), acc[m][k], color="gray", s=18)
        mean = np.array([np.mean(acc[m][k]) for k in ks])
        std = np.array([np.std(acc[m][k]) for k in ks])
        ax.errorbar(ks, mean, yerr=std, color="C0", marker="o", capsize=3)
        best = ks[int(np.argmax(mean))]
        ax.axvline(best, color=RED, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_title(f"{m} distance (best k = {best}, {mean.max():.3f})")
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-validation accuracy")
    fig.suptitle(f"{NUM_FOLDS}-fold cross-validation on k", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, path):
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(n))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-train", type=int, default=5000)
    p.add_argument("--num-test", type=int, default=500)
    p.add_argument("--skip-cv", action="store_true", help="교차검증 없이 --k, --metric 사용")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--metric", choices=["L1", "L2"], default="L1")
    args = p.parse_args()
    RESULT_DIR.mkdir(exist_ok=True)

    x_train_img, y_train, x_test_img, y_test = load_cifar10()
    print(f"CIFAR-10 전체: train {x_train_img.shape}, test {x_test_img.shape}")
    plot_samples(x_train_img, y_train, RESULT_DIR / "cifar10_samples.png")

    # 앞에서부터 일부만 사용 (L1 거리 계산량이 N_train x N_test x 3072)
    x_train_img, y_train = x_train_img[:args.num_train], y_train[:args.num_train]
    x_test_img, y_test = x_test_img[:args.num_test], y_test[:args.num_test]
    x_train, x_test = flatten(x_train_img), flatten(x_test_img)
    print(f"사용: train {len(x_train)}, test {len(x_test)}")

    if args.skip_cv:
        best_metric, best_k = args.metric, args.k
    else:
        print(f"{NUM_FOLDS}-fold 교차검증 (k = {K_CHOICES})")
        acc = cross_validate(x_train, y_train, ["L1", "L2"], K_CHOICES, NUM_FOLDS)
        plot_cross_validation(acc, RESULT_DIR / "cross_validation.png")
        with open(RESULT_DIR / "cv_accuracy.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "k"] + [f"fold{i + 1}" for i in range(NUM_FOLDS)] + ["mean"])
            for m in acc:
                for k in acc[m]:
                    w.writerow([m, k] + [f"{a:.4f}" for a in acc[m][k]]
                               + [f"{np.mean(acc[m][k]):.4f}"])
        best_metric, best_k = max(((m, k) for m in acc for k in acc[m]),
                                  key=lambda mk: np.mean(acc[mk[0]][mk[1]]))
        print(f"교차검증 최고: {best_metric}, k={best_k} "
              f"({np.mean(acc[best_metric][best_k]):.4f})")

    # 슬라이드 24: 고른 하이퍼파라미터로 테스트셋은 마지막에 한 번만
    knn = KNearestNeighbor(best_metric)
    knn.train(x_train, y_train)
    y_pred = knn.predict(x_test, best_k)
    test_acc = np.mean(y_pred == y_test)
    print(f"테스트 정확도 ({best_metric}, k={best_k}): {test_acc:.4f} "
          f"({int(np.sum(y_pred == y_test))}/{len(y_test)})")

    cm = plot_confusion(y_test, y_pred,
                        f"CIFAR-10 {best_k}-NN ({best_metric}), "
                        f"train {len(x_train)} / test {len(x_test)}, acc {test_acc:.3f}",
                        RESULT_DIR / "confusion_matrix.png")
    with open(RESULT_DIR / "test_result.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "num_test", "correct", "accuracy"])
        for c, name in enumerate(CLASSES):
            w.writerow([name, cm[c].sum(), cm[c, c], f"{cm[c, c] / max(cm[c].sum(), 1):.4f}"])
        w.writerow(["total", len(y_test), int(np.trace(cm)), f"{test_acc:.4f}"])

    plot_nearest_neighbors(knn, x_test_img, x_test, y_test, x_train_img, best_k,
                           RESULT_DIR / "nearest_neighbors.png")
    print(f"결과 저장: {RESULT_DIR}")


if __name__ == "__main__":
    main()
