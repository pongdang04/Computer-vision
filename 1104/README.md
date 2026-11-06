# KNN on CIFAR-10 (과제 11/4)

- 보고서: [한국어](report_ko.md) · [English](report.md)
- 코드: `knn.py`(구현), `run_experiments.py`(실험), `test_knn.py`(검증), `analyze_results.py`(보조 분석), `make_figures.py`(그림)
- 결과: `results/*.csv`, `results/*.json`, `figures/*.png`

## 실행

CIFAR-10 데이터셋(`cifar-10-python.tar.gz`)은 [공식 사이트](https://www.cs.toronto.edu/~kriz/cifar.html)에서 받습니다. 전체 실험은 약 50분 걸립니다.

```bash
pip install -r requirements.txt
python test_knn.py <cifar-10-python.tar.gz 경로>         # 구현 검증
python run_experiments.py <cifar-10-python.tar.gz 경로>  # 전체 실험
python analyze_results.py <cifar-10-python.tar.gz 경로>  # 보조 분석
python make_figures.py                                    # 그림 생성
```

`results/cache/`(거리 계산 캐시)는 용량이 커서 저장소에 포함하지 않았습니다.
