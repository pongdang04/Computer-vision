
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


TILE_GRID = (8, 8)   
CLAHE_CLIP = 3.0     

# channel colors
CHANNELS = (("R", 2, "#d64545"), ("G", 1, "#3a9a4a"), ("B", 0, "#3b6fd1"))
CDF_COLOR = "#333333"


def find_image(stem: str) -> Path:
    for ext in IMG_EXTS:
        for p in (BASE_DIR / f"{stem}{ext}", BASE_DIR / f"{stem}{ext.upper()}"):
            if p.exists():
                return p
    raise FileNotFoundError(f"{BASE_DIR}에 '{stem}.jpg' 또는 '{stem}.png' 파일이 없습니다.")


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if img is None:
        raise IOError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"저장 실패: {path}")
    buf.tofile(str(path))


def histogram_equalization(ch: np.ndarray) -> np.ndarray:
    hist = np.bincount(ch.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[hist > 0][0]         
    if cdf_min == ch.size:            
        return ch.copy()
    lut = np.rint((cdf - cdf_min) / (ch.size - cdf_min) * 255)
    return np.clip(lut, 0, 255).astype(np.uint8)[ch]


def adaptive_histogram_equalization(ch: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=0, tileGridSize=TILE_GRID).apply(ch)


def clahe(ch: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=TILE_GRID).apply(ch)


def apply_per_channel(func, bgr: np.ndarray) -> np.ndarray:
    return cv2.merge([func(ch) for ch in cv2.split(bgr)])


# ----------------psnr, 시각화 등 ----------------
def psnr(ref: np.ndarray, img: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(max_val ** 2 / mse)


def psnr_report(ref: np.ndarray, img: np.ndarray) -> dict:
    out = {"RGB": psnr(ref, img)}
    for name, idx, _ in CHANNELS:
        out[name] = psnr(ref[..., idx], img[..., idx])
    return out


def fmt(p: float) -> str:
    return "inf" if np.isinf(p) else f"{p:.2f}"


def channel_hist(img: np.ndarray, idx: int) -> np.ndarray:
    return np.bincount(img[..., idx].ravel(), minlength=256)


def draw_image(ax, bgr: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # OpenCV BGR → matplotlib RGB
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def draw_hist(ax, hist: np.ndarray, color: str, ymax: float, title: str) -> None:
    cdf = hist.cumsum()
    ax.bar(np.arange(256), hist, width=1.0, color=color, label="histogram")
    ax.plot(cdf / cdf[-1] * ymax, color=CDF_COLOR, linewidth=1.5, label="CDF (scaled)")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def image_title(name: str, p: dict) -> str:
    return f"{name}\nPSNR = {fmt(p['RGB'])} dB"


def hist_title(ch_name: str, p: dict) -> str:
    return f"{ch_name} channel  (PSNR {fmt(p[ch_name])} dB)"


def plot_comparison(results: dict, psnrs: dict, ymax: dict, save_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(4, n, figsize=(4.8 * n, 13.5),
                             gridspec_kw={"height_ratios": [1.35, 1, 1, 1]})
    for col, (name, img) in enumerate(results.items()):
        draw_image(axes[0, col], img, image_title(name, psnrs[name]))
        for row, (ch_name, idx, color) in enumerate(CHANNELS, start=1):
            draw_hist(axes[row, col], channel_hist(img, idx), color, ymax[ch_name],
                      hist_title(ch_name, psnrs[name]))
            axes[row, col].set_xlabel("Pixel intensity", fontsize=9)
    for row in (1, 2, 3):
        axes[row, 0].set_ylabel("Number of pixels", fontsize=9)
    fig.suptitle("Histogram Equalization on RGB channels  (PSNR vs. original)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # 윗부분을 비워서 전체 제목이 열 제목과 겹치지 않게
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_single(name: str, img: np.ndarray, p: dict, ymax: dict, save_path: Path) -> None:
    """방법 하나: 이미지 + R/G/B 히스토그램, 제목에 PSNR"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), gridspec_kw={"width_ratios": [1.45, 1, 1, 1]})
    draw_image(axes[0], img, name)
    for ax, (ch_name, idx, color) in zip(axes[1:], CHANNELS):
        draw_hist(ax, channel_hist(img, idx), color, ymax[ch_name], hist_title(ch_name, p))
        ax.set_xlabel("Pixel intensity", fontsize=9)
    axes[1].set_ylabel("Number of pixels", fontsize=9)
    fig.suptitle(f"{name}  |  PSNR vs. original = {fmt(p['RGB'])} dB"
                 f"   (R {fmt(p['R'])} / G {fmt(p['G'])} / B {fmt(p['B'])} dB)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_image("input")
    RESULT_DIR.mkdir(exist_ok=True)

    bgr = imread_unicode(in_path, cv2.IMREAD_COLOR)   # 컬러 3채널 그대로 (OpenCV는 B, G, R 순서)
    print(f"입력: {in_path.name}  크기: {bgr.shape[1]}x{bgr.shape[0]}  채널: {bgr.shape[2]}")

    diffs = [int(np.abs(histogram_equalization(c).astype(int) - cv2.equalizeHist(c)).max())
             for c in cv2.split(bgr)]
    print(f"HE 직접 구현 vs cv2.equalizeHist 최대 픽셀 차이 (B, G, R): {diffs}")

    results = {
        "Original": bgr,
        "HE": apply_per_channel(histogram_equalization, bgr),
        "AHE": apply_per_channel(adaptive_histogram_equalization, bgr),
        f"CLAHE (clip={CLAHE_CLIP:g})": apply_per_channel(clahe, bgr),
    }
    file_stems = ["original", "he", "ahe", "clahe"]

    psnrs = {name: psnr_report(bgr, img) for name, img in results.items()}
    ymax = {ch: max(channel_hist(img, idx).max() for img in results.values()) for ch, idx, _ in CHANNELS}

    for stem, (name, img) in zip(file_stems, results.items()):
        imwrite_unicode(RESULT_DIR / f"{stem}.png", img)
        plot_single(name, img, psnrs[name], ymax, RESULT_DIR / f"{stem}_hist.png")
    plot_comparison(results, psnrs, ymax, RESULT_DIR / "comparison.png")

    # 표 출력 + CSV 저장
    lines = ["method,psnr_rgb_db,psnr_r_db,psnr_g_db,psnr_b_db,mean,std"]
    print(f"\n{'method':<16}{'PSNR RGB':>10}{'R':>8}{'G':>8}{'B':>8}{'mean':>8}{'std':>7}   (dB)")
    for name, img in results.items():
        p = psnrs[name]
        print(f"{name:<16}{fmt(p['RGB']):>10}{fmt(p['R']):>8}{fmt(p['G']):>8}{fmt(p['B']):>8}"
              f"{img.mean():>8.1f}{img.std():>7.1f}")
        lines.append(f"{name},{fmt(p['RGB'])},{fmt(p['R'])},{fmt(p['G'])},{fmt(p['B'])},"
                     f"{img.mean():.2f},{img.std():.2f}")
    (RESULT_DIR / "psnr.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 저장: {RESULT_DIR}")


if __name__ == "__main__":
    main()
