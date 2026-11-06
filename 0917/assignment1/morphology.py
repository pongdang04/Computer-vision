"""
Assignment: Structuring Element(SE) 4종으로 Dilation / Erosion 적용

사용법
    python morphology.py                    # 테스트 이미지 생성 후 적용
    python morphology.py --image a.png      # 임의 이미지를 이진화(Otsu)해서 적용

결과: results/ 폴더
    test_image.png            생성한 테스트 이미지 (확대본, test_image_raw.png 는 원본 크기)
    dilation.png, erosion.png  SE 별로 한 줄씩 [Original -> SE(값 행렬) -> Result], 총 4줄
    dilation_*.png, erosion_*.png   개별 결과 (확대본)
    pixel_counts.csv          전경(1) 픽셀 수

SE (슬라이드 28)
    Vertical 3x1      [1 1 1]^T          padding: 위/아래 1
    Horizontal 1x3    [1 1 1]            padding: 왼/오 1
    Square 3x3        all ones           padding: 네 방향 1
    Cross/Disk 3x3    [[0 1 0] [1 1 1] [0 1 0]]   padding: 네 방향 1

규칙 (슬라이드 29~38)
    - 0으로 padding 후 SE 중심을 각 픽셀에 맞춰 이동
    - Dilation: SE의 1 위치 중 하나라도 1과 겹치면 1
    - Erosion : SE의 1 위치가 모두 1이면(full match) 1, 아니면 0
    - 결과는 padding을 제거해 입력과 같은 크기
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
UPSCALE = 16  # 개별 결과 저장 시 픽셀 하나를 16x16 으로 확대 (nearest)

SES = {
    "vertical": np.array([[1], [1], [1]], dtype=np.uint8),
    "horizontal": np.array([[1, 1, 1]], dtype=np.uint8),
    "square": np.ones((3, 3), dtype=np.uint8),
    "cross": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
}
SE_TITLES = {
    "vertical": "Vertical 3x1",
    "horizontal": "Horizontal 1x3",
    "square": "Square 3x3",
    "cross": "Cross/Disk 3x3",
}


# ---------------- 테스트 이미지 ----------------
def make_test_image() -> np.ndarray:
    """SE 모양/방향 차이가 드러나도록 설계한 40x40 이진 이미지"""
    img = np.zeros((40, 40), dtype=np.uint8)
    img[4, 4] = 1                    # (1) 점 1개: dilation 결과 = SE 모양
    img[4, 10:21] = 1                # (2) 1px 가로선: vertical SE erosion 에서 사라짐
    img[2:13, 30] = 1                # (3) 1px 세로선: horizontal SE erosion 에서 사라짐
    for i in range(11):              # (4) 1px 대각선: square(띠) vs cross(계단) 차이
        img[16 + i, 2 + i] = 1
    img[16:25, 18:27] = 1            # (5) 9x9 사각형: dilation 모서리 모양 차이
    img[16:23, 32:35] = 1            # (6) 3px 두께 세로 막대: erosion 후 방향별로 남는 정도 차이
    img[30:36, 2:8] = 1              # (7) 가로로 1px 틈이 있는 두 블록 -> horizontal/square/cross 가 연결
    img[30:36, 9:15] = 1
    img[28:33, 22:28] = 1            # (8) 세로로 1px 틈이 있는 두 블록 -> vertical/square/cross 가 연결
    img[34:39, 22:28] = 1
    rr, cc = np.mgrid[0:40, 0:40]    # (9) 반지름 3 마름모: erosion 에서 square(더 많이 깎임) vs cross 차이
    img[np.abs(rr - 33) + np.abs(cc - 35) <= 3] = 1
    return img


def load_binary_image(path: Path) -> np.ndarray:
    gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"이미지를 읽을 수 없습니다: {path}")
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


# ---------------- 연산 ----------------
def pad_for_se(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """슬라이드 28: SE 크기에 맞춰 0 padding (3x1 -> 위/아래 1, 1x3 -> 왼/오 1, 3x3 -> 네 방향 1)"""
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    return np.pad(img, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)


def se_windows(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """각 픽셀에 SE 중심을 맞췄을 때 SE의 1 위치에 놓인 값들 -> shape (H, W, SE의 1 개수)"""
    windows = sliding_window_view(pad_for_se(img, se), se.shape)
    return windows[..., se == 1]


def dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    return se_windows(img, se).any(axis=-1).astype(np.uint8)


def erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    return se_windows(img, se).all(axis=-1).astype(np.uint8)


# ---------------- 검증 ----------------
def verify() -> None:
    """슬라이드 예제 결과, OpenCV(0 padding) 결과와 비교"""
    a = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    slide_dilation = {  # 슬라이드 29~32
        "vertical": [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]],
        "horizontal": [[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 0]],
        "square": [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]],
        "cross": [[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]],
    }
    for name, expected in slide_dilation.items():
        assert np.array_equal(dilate(a, SES[name]), expected), f"slide dilation mismatch: {name}"

    # 슬라이드 36: 3x3 all-ones 를 square SE 로 erosion -> 가운데만 1
    ones = np.ones((3, 3), dtype=np.uint8)
    assert np.array_equal(erode(ones, SES["square"]), [[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    # 슬라이드 33 / 38: 그림 예제
    a33 = np.zeros((5, 5), dtype=np.uint8)
    a33[1:4, 2] = 1
    exp33 = np.zeros((7, 7), dtype=np.uint8)  # 7x7 캔버스 위에 그린 결과
    exp33[1:6, 3] = 1
    exp33[2:5, 2:5] = 1
    assert np.array_equal(dilate(np.pad(a33, 1), SES["cross"]), exp33), "slide 33 mismatch"

    a38 = np.zeros((5, 5), dtype=np.uint8)
    a38[:, 0:3] = 1
    a38[2:4, 3] = 1
    exp38 = np.zeros((5, 5), dtype=np.uint8)
    exp38[1:4, 1] = 1
    exp38[2:4, 2] = 1
    assert np.array_equal(erode(a38, SES["cross"]), exp38), "slide 38 mismatch"

    # OpenCV 와 비교 (border 를 0 으로 고정해야 슬라이드 규칙과 동일)
    rng = np.random.default_rng(0)
    rand = (rng.random((50, 60)) > 0.5).astype(np.uint8)
    for name, se in SES.items():
        cv_d = cv2.dilate(rand, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        cv_e = cv2.erode(rand, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        assert np.array_equal(dilate(rand, se), cv_d), f"cv2 dilation mismatch: {name}"
        assert np.array_equal(erode(rand, se), cv_e), f"cv2 erosion mismatch: {name}"
    print("검증 통과: 슬라이드 예제(29~33, 36, 38) 및 OpenCV 결과와 일치")


# ---------------- 시각화 ----------------
def show_binary(ax, img: np.ndarray, title: str, grid: bool) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    draw_grid(ax, img.shape, grid)


def draw_grid(ax, shape, grid: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if grid:
        ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
        ax.grid(which="minor", color="#555555", linewidth=0.3)
        ax.tick_params(which="minor", length=0)


def save_upscaled(path: Path, img: np.ndarray) -> None:
    big = np.kron(img * 255, np.ones((UPSCALE, UPSCALE), dtype=np.uint8))
    cv2.imwrite(str(path), big)


def show_se_matrix(ax, se: np.ndarray, title: str) -> None:
    """SE 를 행렬 값(0/1)으로 표시, 중심(origin)은 슬라이드처럼 빨간 네모로 표시"""
    for (r, c), v in np.ndenumerate(se):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#4292c6" if v else "#f2f6fb",
                                   edgecolor="#333333", linewidth=1.5))
        ax.text(c, r, str(v), ha="center", va="center", fontsize=18,
                color="white" if v else "#333333", weight="bold")
    cr, cc = se.shape[0] // 2, se.shape[1] // 2
    ax.add_patch(plt.Rectangle((cc - 0.4, cr - 0.4), 0.8, 0.8, fill=False, edgecolor="#e03030", linewidth=2.5))
    ax.set_title(title, fontsize=12)
    ax.set_xlim(se.shape[1] / 2 - 2.5, se.shape[1] / 2 + 1.5)  # 3x1, 1x3 도 3x3 과 같은 크기로 보이게
    ax.set_ylim(se.shape[0] / 2 + 1.5, se.shape[0] / 2 - 2.5)
    ax.set_aspect("equal")
    ax.axis("off")


def save_figures(img: np.ndarray, results: dict) -> None:
    """dilation.png / erosion.png: SE 별로 한 줄씩 [Original -> SE -> Result]"""
    grid = max(img.shape) <= 64
    for op in ("dilation", "erosion"):
        fig, axes = plt.subplots(len(SES), 3, figsize=(12, 16.5), gridspec_kw={"width_ratios": [1, 0.8, 1]})
        for i, (name, se) in enumerate(SES.items()):
            show_binary(axes[i, 0], img, "Original", grid)
            show_se_matrix(axes[i, 1], se, f"SE: {SE_TITLES[name]}")
            show_binary(axes[i, 2], results[name][op], f"{op.capitalize()} result", grid)
            for j in (0, 1):  # 칸 사이 화살표
                axes[i, j].annotate("", xy=(1.13, 0.5), xytext=(1.02, 0.5), xycoords="axes fraction",
                                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
        fig.suptitle(op.capitalize(), fontsize=20, weight="bold")
        fig.tight_layout(h_pad=2.0, w_pad=4.0)
        fig.savefig(RESULT_DIR / f"{op}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = sys.argv[1:]
    RESULT_DIR.mkdir(exist_ok=True)
    verify()

    if "--image" in args:
        img = load_binary_image(Path(args[args.index("--image") + 1]))
    else:
        img = make_test_image()
        cv2.imwrite(str(RESULT_DIR / "test_image_raw.png"), img * 255)
        save_upscaled(RESULT_DIR / "test_image.png", img)

    results = {}
    lines = ["se,operation,foreground_pixels", f"-,original,{int(img.sum())}"]
    for name, se in SES.items():
        results[name] = {"dilation": dilate(img, se), "erosion": erode(img, se)}
        for op, out in results[name].items():
            save_upscaled(RESULT_DIR / f"{op}_{name}.png", out)
            lines.append(f"{name},{op},{int(out.sum())}")
            print(f"{op:9s} {SE_TITLES[name]:15s} 전경 픽셀 {int(img.sum())} -> {int(out.sum())}")
    (RESULT_DIR / "pixel_counts.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    save_figures(img, results)
    print(f"결과 저장: {RESULT_DIR}")


if __name__ == "__main__":
    main()
