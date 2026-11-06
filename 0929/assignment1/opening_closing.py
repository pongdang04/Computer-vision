"""
Assignment: Dilation / Erosion / Opening / Closing 과정 시각화

사용법
    python opening_closing.py                 # 테스트 이미지 생성 후 적용
    python opening_closing.py --image a.png   # 임의 이미지를 이진화(Otsu)해서 적용

결과: results/ 폴더
    같은 입력 A 에 Dilation / Erosion / Opening / Closing 을 모두 적용해 비교
    process_<입력>.png   행 = Dilation / Erosion / Opening / Closing
                         열 = SE, Input, (중간 결과), Output, Input vs Output
        process_small.png          15x23 작은 예제, 셀마다 0/1 값 표시
        process_test.png           노이즈/다리/구멍/틈/홈이 있는 48x52 테스트 이미지
        process_lines_circles.png  슬라이드 44: 선 + 원 (disk SE)
        process_disk_holes.png     슬라이드 50: 구멍 + pepper 노이즈가 있는 원판 (disk SE)
        (--image 사용 시 process_user.png)
    compare.png          행 = 입력 이미지, 열 = Input / Dilation / Erosion / Opening / Closing
    pixel_counts.csv     입력별, 연산별, 단계별 전경(1) 픽셀 수

색 표시 (process 그림)
    흰색 = 전경(1), 검정 = 배경(0)  (Input / 중간 결과 / Output 칸은 결과 그대로)
    Input vs Output 칸과 compare.png 만 입력 대비 변화를 색으로 표시
    빨강 = 입력에서 1 이었다가 결과에서 0 이 된 픽셀 (깎인 픽셀)
    초록 = 입력에서 0 이었다가 결과에서 1 이 된 픽셀 (붙은 픽셀)

정의 (슬라이드 35, 40~51)
    Erosion  A ⊖ B : SE 의 1 위치가 모두 1 이면(full match) 1, 아니면 0
    Dilation A ⊕ B : SE 의 1 위치 중 하나라도 1 과 겹치면 1
    Opening  A ∘ B = (A ⊖ B) ⊕ B   erosion 후 dilation, 같은 SE 사용
    Closing  A • B = (A ⊕ B) ⊖ B   dilation 후 erosion, 같은 SE 사용
    padding 은 0, 결과는 입력과 같은 크기
"""
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

SQUARE = np.ones((3, 3), dtype=np.uint8)
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

RED = np.array([0.90, 0.20, 0.20])
GREEN = np.array([0.20, 0.80, 0.30])
STEP_NAMES = {"Erosion": "Eroded", "Dilation": "Dilated"}  # 중간 결과 이름 (슬라이드 42/48)


def disk(radius: int) -> np.ndarray:
    rr, cc = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return (rr ** 2 + cc ** 2 <= radius ** 2).astype(np.uint8)


# ---------------- 연산 (0917 과제와 동일한 규칙) ----------------
def se_windows(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """각 픽셀에 SE 중심을 맞췄을 때 SE 의 1 위치에 놓인 값들 -> shape (H, W, SE의 1 개수)"""
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    return sliding_window_view(padded, se.shape)[..., se == 1]


def dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    return se_windows(img, se).any(axis=-1).astype(np.uint8)


def erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    return se_windows(img, se).all(axis=-1).astype(np.uint8)


def opening(img: np.ndarray, se: np.ndarray):
    """erosion -> dilation. (중간 결과, 최종 결과) 반환"""
    eroded = erode(img, se)
    return eroded, dilate(eroded, se)


def closing(img: np.ndarray, se: np.ndarray):
    """dilation -> erosion. (중간 결과, 최종 결과) 반환"""
    dilated = dilate(img, se)
    return dilated, erode(dilated, se)


# ---------------- 검증 ----------------
def verify() -> None:
    """OpenCV(0 padding) 결과 및 opening/closing 의 성질과 비교"""
    rng = np.random.default_rng(0)
    rand = (rng.random((50, 60)) > 0.5).astype(np.uint8)
    kw = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)
    for se in (SQUARE, CROSS, disk(3)):
        assert np.array_equal(erode(rand, se), cv2.erode(rand, se, **kw))
        assert np.array_equal(dilate(rand, se), cv2.dilate(rand, se, **kw))
        _, op = opening(rand, se)
        _, cl = closing(rand, se)
        assert np.array_equal(op, cv2.morphologyEx(rand, cv2.MORPH_OPEN, se, **kw)), "opening mismatch"
        assert np.array_equal(cl, cv2.morphologyEx(rand, cv2.MORPH_CLOSE, se, **kw)), "closing mismatch"
        # 성질: opening 은 원본의 부분집합, 두 번 해도 같음(idempotent)
        assert np.all(op <= rand) and np.array_equal(opening(op, se)[1], op)
        # closing 은 원본을 포함 (경계에서 떨어진 영역 기준), 두 번 해도 같음
        assert np.all(cl[3:-3, 3:-3] >= rand[3:-3, 3:-3]) and np.array_equal(closing(cl, se)[1], cl)
    print("검증 통과: OpenCV 결과와 일치, opening/closing 성질(부분집합/포함, idempotent) 확인")


# ---------------- 입력 이미지 ----------------
def make_small_example() -> np.ndarray:
    """0/1 값을 직접 보여줄 15x23 예제"""
    img = np.zeros((15, 23), dtype=np.uint8)
    img[2:9, 2:9] = 1      # 블록 A (7x7)
    img[5, 5] = 0          # A 안의 구멍 -> opening 은 유지, closing 은 메움
    img[5, 9:11] = 1       # A 에서 튀어나온 1px 선 -> opening 으로 제거
    img[2:9, 14:17] = 1    # 블록 B1, B2 (3px 폭) 사이 1px 틈 -> closing 으로 연결
    img[2:9, 18:21] = 1
    img[12, 4] = 1         # 점 노이즈 -> erosion/opening 으로 제거 (블록과 3칸 떨어져 closing 에서도 안 붙음)
    img[12, 17] = 1
    return img


def make_test_image() -> np.ndarray:
    """opening/closing 효과가 모두 드러나도록 설계한 48x52 이진 이미지"""
    img = np.zeros((48, 52), dtype=np.uint8)
    img[4:16, 4:16] = 1                  # (1) 사각형 + 외부 salt 노이즈 -> opening 으로 노이즈 제거
    for r, c in [(3, 20), (8, 22), (14, 19), (19, 3), (20, 10)]:
        img[r, c] = 1
    img[4:16, 28:34] = 1                 # (2) 1px 다리로 연결된 두 블록 -> opening 으로 분리
    img[9, 34:38] = 1
    img[4:16, 38:44] = 1
    img[24:38, 3:17] = 1                 # (3) pepper 노이즈(내부 구멍) -> closing 으로 메워짐
    for r, c in [(27, 6), (30, 12), (34, 8), (28, 14), (35, 14)]:
        img[r, c] = 0
    img[24:38, 22:28] = 1                # (4) 1px 틈으로 떨어진 두 블록 -> closing 으로 연결
    img[24:38, 29:34] = 1
    rr, cc = np.mgrid[0:48, 0:52]        # (5) 원 + 가장자리 좁은 홈 -> closing 으로 메워짐
    img[(rr - 38) ** 2 + (cc - 44) ** 2 <= 25] = 1
    img[38, 39:43] = 0
    return img


def make_lines_and_circles() -> np.ndarray:
    """슬라이드 44 재현: 가는 선 + 원 -> disk SE opening 으로 원만 남김"""
    rng = np.random.default_rng(1)
    img = np.zeros((200, 200), dtype=np.uint8)
    for _ in range(25):
        c = tuple(int(v) for v in rng.integers(12, 188, 2))
        cv2.circle(img, c, 8, 1, -1)
    for _ in range(10):
        p1 = tuple(int(v) for v in rng.integers(0, 200, 2))
        p2 = tuple(int(v) for v in rng.integers(0, 200, 2))
        cv2.line(img, p1, p2, 1, 2)
    return img


def make_disk_with_holes() -> np.ndarray:
    """슬라이드 50 재현: 원판 안의 구멍 + pepper 노이즈 -> disk SE closing 으로 SE 보다 작은 구멍만 메움"""
    rng = np.random.default_rng(2)
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, 1, -1)
    for _ in range(8):                   # SE 보다 작은 구멍 -> 메워짐
        c = tuple(int(v) for v in rng.integers(55, 145, 2))
        cv2.circle(img, c, 3, 0, -1)
    for c in [(70, 130), (135, 75)]:     # SE 보다 큰 구멍 -> 그대로 남음
        cv2.circle(img, c, 10, 0, -1)
    rr, cc = np.nonzero(img)
    pick = rng.choice(len(rr), 250, replace=False)
    img[rr[pick], cc[pick]] = 0
    return img


def load_binary_image(path: Path) -> np.ndarray:
    gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"이미지를 읽을 수 없습니다: {path}")
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


# ---------------- 시각화 ----------------
def change_rgb(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """cur 를 흑백으로 그리고 prev 대비 깎인 픽셀은 빨강, 붙은 픽셀은 초록으로 표시"""
    rgb = np.repeat(cur[..., None].astype(float), 3, axis=2)
    rgb[(prev == 1) & (cur == 0)] = RED
    rgb[(prev == 0) & (cur == 1)] = GREEN
    return rgb


def draw_grid(ax, shape) -> None:
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which="minor", color="#777777", linewidth=0.4)
    ax.tick_params(which="minor", length=0)


def show(ax, rgb_or_bin, title: str, grid: bool, values: np.ndarray = None) -> None:
    if rgb_or_bin.ndim == 2:
        ax.imshow(rgb_or_bin, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(rgb_or_bin, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if grid:
        draw_grid(ax, rgb_or_bin.shape)
    if values is not None:  # 셀마다 0/1 값 표시
        for (r, c), v in np.ndenumerate(values):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=7,
                    color="black" if v else "#bbbbbb")


def show_se(ax, se: np.ndarray, title: str) -> None:
    """SE 를 0/1 행렬로 표시, 중심(origin)은 빨간 네모"""
    for (r, c), v in np.ndenumerate(se):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#4292c6" if v else "#f2f6fb",
                                   edgecolor="#333333", linewidth=1.2))
        if se.size <= 25:
            ax.text(c, r, str(v), ha="center", va="center", fontsize=14,
                    color="white" if v else "#333333", weight="bold")
    cr, cc = se.shape[0] // 2, se.shape[1] // 2
    ax.add_patch(plt.Rectangle((cc - 0.4, cr - 0.4), 0.8, 0.8, fill=False, edgecolor="#e03030", linewidth=2))
    ax.set_title(title, fontsize=11)
    m = max(se.shape) / 2 + 1
    ax.set_xlim(se.shape[1] / 2 - m, se.shape[1] / 2 + m - 1)
    ax.set_ylim(se.shape[0] / 2 + m - 1, se.shape[0] / 2 - m)
    ax.set_aspect("equal")
    ax.axis("off")


def arrow(fig, ax_from, ax_to, label: str) -> None:
    """두 칸 사이에 라벨 달린 화살표 (슬라이드 42/48 의 process 그림처럼)"""
    b1, b2 = ax_from.get_position(), ax_to.get_position()
    y = (b1.y0 + b1.y1) / 2
    fig.add_artist(matplotlib.patches.FancyArrowPatch(
        (b1.x1 + 0.005, y), (b2.x0 - 0.005, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=18, lw=2, color="#c2185b"))
    fig.text((b1.x1 + b2.x0) / 2, y + 0.04, label, ha="center", va="bottom",
             fontsize=11, weight="bold", color="#c2185b")


def legend(fig) -> None:
    handles = [matplotlib.patches.Patch(facecolor="white", edgecolor="black", label="foreground (1)"),
               matplotlib.patches.Patch(facecolor=RED, label="removed vs input (1→0)"),
               matplotlib.patches.Patch(facecolor=GREEN, label="added vs input (0→1)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)


def run_all(img: np.ndarray, se: np.ndarray) -> dict:
    """같은 입력에 네 연산을 모두 적용: {연산: [(단계 이름, 결과), ...]}"""
    eroded = erode(img, se)
    _, opened = opening(img, se)
    dilated, closed = closing(img, se)
    return {"Dilation": [("Dilation", dilated)],
            "Erosion": [("Erosion", eroded)],
            "Opening": [("Erosion", eroded), ("Dilation", opened)],
            "Closing": [("Dilation", dilated), ("Erosion", closed)]}


FORMULAS = {"Dilation": "A ⊕ B", "Erosion": "A ⊖ B", "Opening": "A ∘ B = (A ⊖ B) ⊕ B", "Closing": "A • B = (A ⊕ B) ⊖ B"}

# 그림에 넣을 전후 변화 설명 (입력별, 연산별)
GENERIC_NOTES = {
    "Dilation": "Objects grow; small gaps and holes are filled",
    "Erosion": "Objects shrink; parts smaller than the SE disappear",
    "Opening": "Small objects / thin parts removed; remaining objects keep their size",
    "Closing": "Small holes / gaps filled; outer boundary keeps its size",
}
NOTES = {
    "small": {
        "Dilation": "Blocks grow by 1 px; hole and 1 px gap filled, but noise dots also grow to 3x3",
        "Erosion": "Blocks shrink by 1 px; noise dots and 1 px line removed, inner hole grows",
        "Opening": "Noise dots and 1 px line removed; blocks back to original size, hole kept",
        "Closing": "Inner hole and 1 px gap filled; outer size kept, noise dots kept",
    },
    "test": {
        "Dilation": "Objects grow; gap, holes and notch filled, but salt noise also grows",
        "Erosion": "Objects shrink; salt noise and 1 px bridge removed, pepper holes grow",
        "Opening": "Salt noise removed and 1 px bridge cut; large shapes keep their size",
        "Closing": "Pepper holes, 1 px gap and notch filled; outline unchanged, salt noise kept",
    },
    "lines_circles": {
        "Dilation": "Lines and circles thicken and merge into one connected mass",
        "Erosion": "Lines thinner than the SE vanish; circles shrink to small dots",
        "Opening": "Lines removed; circles restored to their original size",
        "Closing": "Narrow gaps between nearby shapes filled; lines are not removed",
    },
    "disk_holes": {
        "Dilation": "Disk grows; small holes and pepper noise filled, large holes shrink",
        "Erosion": "Disk shrinks; every hole and pepper dot grows, much of the disk is lost",
        "Opening": "Pepper noise widens into cracks; disk is damaged (wrong tool for pepper noise)",
        "Closing": "Holes smaller than the SE and pepper noise filled; large holes (r=10) kept",
    },
}


def process_figure(path: Path, title: str, img: np.ndarray, se: np.ndarray, se_name: str, grid: bool,
                   notes: dict, values: bool = False) -> dict:
    """같은 입력 A 로 행 = Dilation / Erosion / Opening / Closing
    열 = [SE] [Input] -step1-> [중간 결과] -step2-> [Output] [Input vs Output]"""
    results = run_all(img, se)
    h, w = img.shape
    cell = 4.2 if values else 3.4
    n = len(results)
    fig, axes = plt.subplots(n, 5, figsize=(cell * 4.6, cell * n * max(h / w, 0.55) + 3),
                             gridspec_kw={"width_ratios": [0.6, 1, 1, 1, 1], "wspace": 0.35, "hspace": 0.75})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.06)
    for r, (op_name, steps) in enumerate(results.items()):
        show_se(axes[r, 0], se, f"SE: {se_name}")
        show(axes[r, 1], img, f"Input A ({int(img.sum())} px)", grid, img if values else None)
        cols = [2] if len(steps) == 1 else [2, 3]
        for col, (i, (op, out)) in zip(cols, enumerate(steps)):  # 중간 과정은 결과 그대로 (흰색 = 1)
            label = "Output" if i == len(steps) - 1 else f"{STEP_NAMES[op]} image"
            show(axes[r, col], out, f"{label} ({int(out.sum())} px)", grid, out if values else None)
        if len(steps) == 1:
            axes[r, 3].axis("off")
        show(axes[r, 4], change_rgb(img, steps[-1][1]), "Input vs Output", grid, None)
        top = axes[r, 1].get_position().y1
        fig.text(0.02, top + 0.035, f"{op_name}   {FORMULAS[op_name]}", fontsize=15, weight="bold",
                 color="#1a3d6d")
        fig.text(0.27, top + 0.035, f"→ {notes[op_name]}", fontsize=13, color="#333333")
        prev_ax = axes[r, 1]
        for col, (op, _) in zip(cols, steps):
            arrow(fig, prev_ax, axes[r, col], op)
            prev_ax = axes[r, col]
    fig.suptitle(title, fontsize=17, weight="bold")
    legend(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return results


def compare_figure(path: Path, cases: list) -> None:
    """행 = 입력 이미지, 열 = Input / Dilation / Erosion / Opening / Closing (색 = 입력 대비 변화)"""
    fig, axes = plt.subplots(len(cases), 5, figsize=(20, 4.9 * len(cases)), squeeze=False)
    for r, (name, img, results, grid, notes) in enumerate(cases):
        show(axes[r, 0], img, f"{name}\nInput ({int(img.sum())} px)", grid)
        for c, op in enumerate(("Dilation", "Erosion", "Opening", "Closing"), start=1):
            out = results[op][-1][1]
            show(axes[r, c], change_rgb(img, out), f"{op} ({int(out.sum())} px)", grid)
            axes[r, c].set_xlabel(textwrap.fill(notes[op], 38), fontsize=10)
    fig.suptitle("Same input -> Dilation / Erosion / Opening / Closing (colors = change from input)", fontsize=15,
                 weight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.05, hspace=0.45)
    legend(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = sys.argv[1:]
    RESULT_DIR.mkdir(exist_ok=True)
    verify()

    if "--image" in args:
        user = load_binary_image(Path(args[args.index("--image") + 1]))
        inputs = [("user", "User image", user, SQUARE, "3x3 square")]
    else:
        inputs = [
            ("small", "Small example (cell values)", make_small_example(), SQUARE, "3x3 square"),
            ("test", "Test image", make_test_image(), SQUARE, "3x3 square"),
            ("lines_circles", "Lines + circles (slide 44)", make_lines_and_circles(), disk(5), "disk r=5"),
            ("disk_holes", "Disk with holes (slide 50)", make_disk_with_holes(), disk(5), "disk r=5"),
        ]

    cases = []
    rows = ["image,operation,step,foreground_pixels"]
    for key, title, img, se, se_name in inputs:
        grid = max(img.shape) <= 64
        notes = NOTES.get(key, GENERIC_NOTES)
        results = process_figure(RESULT_DIR / f"process_{key}.png", f"{title}: Dilation / Erosion / Opening / Closing",
                                 img, se, se_name, grid, notes, values=(key == "small"))
        cases.append((title, img, results, grid, notes))
        rows.append(f"{key},-,input,{int(img.sum())}")
        print(f"[{title}] input {int(img.sum())} px")
        for op, steps in results.items():
            flow = " ".join(f"-({s.lower()})-> {int(out.sum())}" for s, out in steps)
            print(f"  {op:8s}: {int(img.sum())} {flow} px")
            for i, (s, out) in enumerate(steps, start=1):
                rows.append(f"{key},{op.lower()},step{i}_{s.lower()},{int(out.sum())}")
    compare_figure(RESULT_DIR / "compare.png", cases)
    (RESULT_DIR / "pixel_counts.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"결과 저장: {RESULT_DIR}")


if __name__ == "__main__":
    main()
