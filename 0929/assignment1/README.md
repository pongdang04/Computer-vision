# Morphological Operations: Dilation / Erosion / Opening / Closing

## 실행 방법

```bash
python opening_closing.py                 # 테스트 이미지 4종 생성 후 적용
python opening_closing.py --image a.png   # 임의 이미지를 Otsu 이진화해서 적용
```

- 구현: `dilate`, `erode`, `opening`, `closing` (`opening_closing.py`). 0 패딩, 출력 크기 = 입력 크기
- 검증: 실행 시 OpenCV(`cv2.erode/dilate/morphologyEx`) 결과와 픽셀 단위로 일치하는지,
  opening ⊆ A ⊆ closing 및 idempotent 성질이 성립하는지 자동 확인

| 연산 | 정의 |
|---|---|
| Dilation | A ⊕ B : SE 의 1 위치 중 하나라도 전경과 겹치면 1 |
| Erosion | A ⊖ B : SE 의 1 위치가 모두 전경이면 1 |
| Opening | A ∘ B = (A ⊖ B) ⊕ B |
| Closing | A • B = (A ⊕ B) ⊖ B |

## 입력 이미지 선택 이유

임의의 사진 대신 **각 연산의 효과가 한눈에 보이도록 설계한 이진 이미지**를 사용했다.

| 입력 | SE | 들어 있는 요소 | 확인할 효과 |
|---|---|---|---|
| `small` (15×23) | 3×3 square | 블록, 내부 구멍 1px, 튀어나온 1px 선, 1px 틈, 점 노이즈 | 셀 값(0/1)으로 연산 규칙 직접 확인 |
| `test` (48×52) | 3×3 square | salt 노이즈, 1px 다리, pepper 노이즈(내부 구멍), 1px 틈, 원의 좁은 홈 | 축소 / 확장 / 노이즈 제거 / 틈 메우기 |
| `lines_circles` (200×200) | disk r=5 | 원(r=8) + 두께 2px 선 | SE 보다 가는 구조 제거 (슬라이드 44) |
| `disk_holes` (200×200) | disk r=5 | 원판 + 작은 구멍(r=3) + 큰 구멍(r=10) + pepper 노이즈 | SE 보다 작은 구멍만 메우기 (슬라이드 50) |

## 결과 파일 (`results/`)

- `process_<입력>.png`: 행 = Dilation / Erosion / Opening / Closing,
  열 = SE, 원본, (중간 결과), 결과, 원본 대비 변화
- `compare.png`: 입력 4종 × (원본, Dilation, Erosion, Opening, Closing) 한 장에 비교
- `pixel_counts.csv`: 입력별·연산별 전경 픽셀 수

각 그림에는 연산별 적용 전후 변화 설명이 함께 들어 있다
(`process_*.png` 는 행 제목 옆, `compare.png` 는 각 결과 아래).

색 표시: 흰색 = 전경, **빨강 = 원본에서 깎인 픽셀**, **초록 = 원본에 새로 붙은 픽셀**

## 전경 픽셀 수

| 입력 | 원본 | Dilation | Erosion | Opening | Closing |
|---|---:|---:|---:|---:|---:|
| small | 94 | 186 | 26 | 90 | 102 |
| test | 719 | 1077 | 400 | 697 | 741 |
| lines_circles | 6920 | 18486 | 778 | 4684 | 8374 |
| disk_holes | 19009 | 22463 | 5153 | 16774 | 19457 |

## 관찰 결과

### Dilation (확장)
- 모든 물체의 경계가 SE 반지름만큼 바깥으로 커진다 (test: 719 → 1077 px, +50%).
- 1px 틈으로 떨어진 두 블록이 하나로 붙고, 내부 구멍과 원의 홈이 메워진다.
- 하지만 salt 노이즈 점도 3×3 덩어리로 같이 커진다. 노이즈를 없애지 못하고 오히려 키운다.

### Erosion (축소)
- 모든 물체의 경계가 안쪽으로 깎인다 (test: 719 → 400 px, −44%).
- 점 노이즈, 1px 다리, 튀어나온 선처럼 SE 보다 작은 구조는 완전히 사라진다.
- 큰 물체도 함께 작아지고, 내부 구멍은 오히려 커진다 (pepper 노이즈 주변이 3×3 으로 넓어짐).
- lines_circles 에서는 선이 모두 사라지고 원만 작은 점으로 남는다 (6920 → 778 px).

### Opening (Erosion → Dilation): 작은 전경 제거
- Erosion 으로 노이즈·다리·선을 지운 뒤 Dilation 으로 남은 물체를 원래 크기로 되돌린다.
- test: salt 노이즈와 1px 다리가 제거되고 큰 사각형은 거의 원래 모양 그대로다 (719 → 697 px).
- lines_circles: SE(r=5) 보다 가는 선이 모두 사라지고 원만 원래 크기로 남는다.
- 항상 원본의 부분집합이다 (빨강만 있고 초록은 없음).
- 내부 구멍(pepper)은 메우지 못한다. disk_holes 처럼 pepper 노이즈가 촘촘하면
  구멍 사이의 얇은 부분까지 깎여 오히려 원판이 손상된다 (19009 → 16774 px).
  → pepper 노이즈에는 Opening 이 맞지 않는다.

### Closing (Dilation → Erosion): 작은 배경 메우기
- Dilation 으로 틈과 구멍을 메운 뒤 Erosion 으로 바깥 경계를 원래 위치로 되돌린다.
- test: pepper 구멍, 1px 틈, 원의 좁은 홈이 메워지고 바깥 크기는 그대로다 (719 → 741 px).
- disk_holes: SE 보다 작은 구멍(r=3)과 pepper 노이즈는 모두 메워지고,
  SE 보다 큰 구멍(r=10)은 그대로 남는다.
- 항상 원본을 포함한다 (초록만 있고 빨강은 없음).
- salt 노이즈(바깥의 작은 점)는 제거하지 못한다.

## 정리

| 연산 | 크기 변화 | 제거하는 것 | 메우는 것 | 원래 크기 유지 |
|---|---|---|---|---|
| Dilation | 커짐 | – | 틈, 작은 구멍 | ✗ |
| Erosion | 작아짐 | 작은 점, 가는 선/다리 | – | ✗ |
| Opening | 약간 작아짐 | 작은 점, 가는 선/다리 | – | ○ |
| Closing | 약간 커짐 | – | 틈, 작은 구멍, 홈 | ○ |

- Dilation 과 Erosion 은 서로 반대 효과지만, 두 연산 모두 원하는 효과와 함께 물체 크기도 바꾼다.
- Opening 과 Closing 은 같은 SE 로 반대 연산을 한 번 더 적용해 **크기는 거의 유지하면서**
  SE 보다 작은 전경(Opening)이나 SE 보다 작은 배경(Closing)만 골라 없앤다.
- 무엇이 지워지고 메워지는지는 SE 의 크기와 모양이 결정한다 (3×3 square vs disk r=5 결과 비교).
