# Linear Classifier on CIFAR-10 (과제 11/18)

무작위로 초기화한 가중치(학습 없음)로 f(x) = Wx + b 점수, Multiclass SVM 손실, Softmax를 계산하고 가중치를 시각화합니다.

- 보고서: [한국어](report_ko.md) · [English](report.md)
- 코드: `linear_classifier.py`(구현), `run.py`(실험·그림), `test_linear.py`(검증)
- 결과: `results/*.csv`, `results/summary.json`, `results/run.log`, `figures/*.png`

## 실행

CIFAR-10 데이터셋(`cifar-10-python.tar.gz`)은 [공식 사이트](https://www.cs.toronto.edu/~kriz/cifar.html)에서 받습니다. 압축을 풀 필요 없이 경로만 넘기면 됩니다.

```bash
pip install -r requirements.txt
python test_linear.py <cifar-10-python.tar.gz 경로>   # 구현 검증
python run.py <cifar-10-python.tar.gz 경로>           # 전체 실험 + 그림 생성
```
