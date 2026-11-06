"""
Assignment 2: 두 이미지 간 산술 연산 (덧셈 / 뺄셈 / 곱셈 / 나눗셈)

사용법
    python image_operations.py           # 컬러로 연산 (f = image1, g = image2)
    python image_operations.py --gray    # 흑백으로 변환 후 연산
    python image_operations.py --swap    # 순서를 바꿔 f = image2, g = image1 로 연산

입력: 같은 폴더의 image1.(jpg/png/...), image2.(jpg/png/...)
결과: results/ 폴더
    operations.png      image1, image2 + 네 가지 연산 결과 한 장
    add.png, subtract.png, multiply.png, divide.png   네 가지 결과 개별 이미지
    overflow_check.png  슬라이드 규칙(clip) 적용 vs uint8 그대로 계산(wrap-around) 비교
    add_wrap.png, subtract_wrap.png                   비교용 개별 결과
    psnr.csv            image1 / image2 대비 PSNR

연산 (슬라이드 25)
    s(x,y) = f(x,y) + g(x,y)
    d(x,y) = f(x,y) - g(x,y)
    p(x,y) = f(x,y) x g(x,y)
    v(x,y) = f(x,y) / g(x,y)

규칙 (슬라이드 26)
    - 결과가 실수면 반올림
    - 255보다 크면 255, 0보다 작으면 0
    - 무한대(0으로 나눈 경우)는 0
"""
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


# ---------------- 입출력 ----------------
def find_image(stem: str) -> Path:
    for ext in IMG_EXTS:
        for p in (BASE_DIR / f"{stem}{ext}", BASE_DIR / f"{stem}{ext.upper()}"):
            if p.exists():
                return p
    raise FileNotFoundError(f"{BASE_DIR}에 '{stem}.jpg' 또는 '{stem}.png' 파일이 없습니다.")


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread는 윈도우에서 한글 경로를 못 읽어서 imdecode로 우회"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if img is None:
        raise IOError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"저장 실패: {path}")
    buf.tofile(str(path))


def to_display(img: np.ndarray) -> np.ndarray:
    """matplotlib 표시용: OpenCV의 BGR → RGB"""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------- 연산 ----------------
def to_pixel_range(x: np.ndarray) -> np.ndarray:
    """슬라이드 26 규칙을 한 번에 적용: 무한대/NaN → 0, 반올림, [0, 255]로 자르기
    np.rint는 .5를 짝수 쪽으로 반올림함 (0.5 → 0). 슬라이드 33의 5/10 = 0 과 일치, OpenCV도 같은 방식"""
    x = np.where(np.isfinite(x), x, 0)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def add_images(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """s(x,y) = f(x,y) + g(x,y). uint8 그대로 더하면 넘침(wrap)이 생기므로 실수로 먼저 계산"""
    return to_pixel_range(f.astype(np.float64) + g)


def subtract_images(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """d(x,y) = f(x,y) - g(x,y), 음수는 0으로"""
    return to_pixel_range(f.astype(np.float64) - g)


def multiply_images(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """p(x,y) = f(x,y) x g(x,y), 255를 넘으면 255로 (슬라이드 31: 100 x 100 = 255)"""
    return to_pixel_range(f.astype(np.float64) * g)


def divide_images(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """v(x,y) = f(x,y) / g(x,y). g=0이면 무한대(또는 0/0=NaN) → 0 (슬라이드 33: 10 / 0 = 0)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return to_pixel_range(f.astype(np.float64) / g)


def add_wrap(f, g):
    """비교용: uint8 그대로 더하면 255를 넘을 때 256으로 나눈 나머지가 됨 (예: 250+10=4)"""
    return f + g


def subtract_wrap(f, g):
    """비교용: uint8 그대로 빼면 음수가 256을 더한 값이 됨 (예: 0-10=246)"""
    return f - g


# 이름: (직접 구현, 같은 규칙의 OpenCV 함수(검증용), 그림 제목)
OPERATIONS = {
    "add": (add_images, cv2.add, "Addition  s = f + g"),
    "subtract": (subtract_images, cv2.subtract, "Subtraction  d = f - g"),
    "multiply": (multiply_images, cv2.multiply, "Multiplication  p = f x g"),
    "divide": (divide_images, cv2.divide, "Division  v = f / g"),
}


def psnr(ref: np.ndarray, img: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(max_val ** 2 / mse)


# ---------------- 슬라이드 예제 검증 ----------------
def verify_slide_example() -> None:
    A = np.array([[0, 100, 10], [4, 0, 10], [8, 0, 5]], dtype=np.uint8)
    B = np.array([[10, 100, 5], [2, 0, 0], [0, 10, 10]], dtype=np.uint8)
    expected = {  # 슬라이드 27, 29, 31, 33의 정답 C
        "add": [[10, 200, 15], [6, 0, 10], [8, 10, 15]],
        "subtract": [[0, 0, 5], [2, 0, 10], [8, 0, 0]],
        "multiply": [[0, 255, 50], [8, 0, 0], [0, 0, 50]],
        "divide": [[0, 1, 2], [2, 0, 0], [0, 0, 0]],
    }
    print("=== 슬라이드 예제 검증 (A, B는 슬라이드 27~33의 3x3 행렬) ===")
    for name, (func, _, _) in OPERATIONS.items():
        C = func(A, B)
        ok = "슬라이드와 일치" if np.array_equal(C, expected[name]) else "불일치!"
        print(f"[{name}] → {ok}\n{C}")
    print("참고) 규칙 없이 uint8 그대로 뺀 경우 A - B =\n", subtract_wrap(A, B))
    print()


# ---------------- 시각화 ----------------
def show(ax, img, title):
    ax.imshow(to_display(img), cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def value_range(img: np.ndarray) -> str:
    return f"min {img.min()} / max {img.max()} / mean {img.mean():.1f}"


def fmt(p):
    return "inf" if np.isinf(p) else f"{p:.2f}"


# ---------------- 메인 ----------------
def main():
    use_gray = "--gray" in sys.argv
    swap = "--swap" in sys.argv
    flag = cv2.IMREAD_GRAYSCALE if use_gray else cv2.IMREAD_COLOR
    RESULT_DIR.mkdir(exist_ok=True)

    verify_slide_example()

    p1, p2 = find_image("image1"), find_image("image2")
    img1, img2 = imread_unicode(p1, flag), imread_unicode(p2, flag)
    for p, img in ((p1, img1), (p2, img2)):
        print(f"{p.stem}: {p.name} {img.shape[1]}x{img.shape[0]}  픽셀 값 {value_range(img)}")
        if img.max() < 50:
            print(f"  [참고] {p.stem}의 최대 밝기가 {img.max()}뿐인 거의 검은 이미지라 "
                  "뺄셈/곱셈/나눗셈 결과도 거의 검게 나옵니다.")

    # 슬라이드 25: 두 이미지 크기가 같아야 픽셀 단위 연산 가능 → image2를 image1 크기로 맞춤
    if img1.shape[:2] != img2.shape[:2]:
        print("[경고] 두 이미지 크기가 달라서 image2를 image1 크기로 리사이즈합니다.")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    (f, f_name), (g, g_name) = ((img2, "Image 2"), (img1, "Image 1")) if swap else ((img1, "Image 1"), (img2, "Image 2"))
    print(f"\nf = {f_name}, g = {g_name}")

    results = {name: func(f, g) for name, (func, _, _) in OPERATIONS.items()}

    # 직접 구현이 같은 규칙(saturate, 0으로 나누면 0)을 쓰는 OpenCV 함수와 같은지 실제 이미지로 확인
    print("\n=== 결과 (OpenCV cv2.add/subtract/multiply/divide와 비교) ===")
    for name, (_, cv_func, _) in OPERATIONS.items():
        n_diff = np.count_nonzero(results[name] != cv_func(f, g))
        check = "OpenCV와 일치" if n_diff == 0 else f"OpenCV와 {n_diff}개 값 다름"
        print(f"{name:<10}{value_range(results[name]):<36}{check}")

    wraps = {"add_wrap": add_wrap(f, g), "subtract_wrap": subtract_wrap(f, g)}
    for name, img in {**results, **wraps}.items():
        imwrite_unicode(RESULT_DIR / f"{name}.png", img)

    # PSNR: 모든 이미지를 image1, image2 각각과 비교
    targets = {"image2": img2, **results, **wraps}
    lines = ["image,psnr_vs_image1,psnr_vs_image2"]
    print(f"\n{'image':<15}{'vs image1(dB)':>15}{'vs image2(dB)':>15}")
    for name, img in targets.items():
        a, b = psnr(img1, img), psnr(img2, img)
        print(f"{name:<15}{fmt(a):>15}{fmt(b):>15}")
        lines.append(f"{name},{fmt(a)},{fmt(b)}")
    (RESULT_DIR / "psnr.csv").write_text("\n".join(lines), encoding="utf-8")

    # 그림 1: 입력 두 장 + 네 가지 연산 결과
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.4))
    show(axes[0, 0], f, f"{f_name}  f(x,y)\n{value_range(f)}")
    show(axes[0, 1], g, f"{g_name}  g(x,y)\n{value_range(g)}")
    for ax, (name, (_, _, title)) in zip(axes.ravel()[2:], OPERATIONS.items()):
        show(ax, results[name], f"{title}\n{value_range(results[name])}")
    fig.suptitle("Arithmetic operations between images  (round, clip to [0, 255], x/0 = 0)", fontsize=14)
    plt.tight_layout(); fig.savefig(RESULT_DIR / "operations.png", dpi=150); plt.close(fig)

    # 그림 2: 슬라이드 규칙 적용 vs uint8 wrap-around
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    show(axes[0, 0], results["add"], "f + g  (clip to 255)")
    show(axes[0, 1], wraps["add_wrap"], "f + g  (uint8 wrap-around)")
    show(axes[1, 0], results["subtract"], "f - g  (clip to 0)")
    show(axes[1, 1], wraps["subtract_wrap"], "f - g  (uint8 wrap-around)")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "overflow_check.png", dpi=150); plt.close(fig)

    print(f"\n결과 저장: {RESULT_DIR}")


if __name__ == "__main__":
    main()
