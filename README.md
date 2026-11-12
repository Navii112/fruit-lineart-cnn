# 과일 사진 기반 5종 분류

사진 속 과일을 개별 분리한 뒤 외곽선과 내부 무늬를 추출해 사과·바나나·레몬·딸기·파인애플을 분류하는 Streamlit 앱입니다.

`사진 → Grounding DINO로 탐지 → SlimSAM으로 분리 → OpenCV로 선화 생성 → CNN 분류`

사전학습 탐지·분리 모델을 연결하고, 외곽선 CNN과 내부 특징 선화 CNN을 동일한 구조와 학습 조건으로 직접 학습했습니다. 앱에서 두 모델의 예측과 전처리 결과를 비교할 수 있습니다.

## 실험 결과

| 입력 | 정확도 | Macro F1 |
|---|---:|---:|
| 외곽선 | 84.4% | 0.8423 |
| 외곽선 + 내부 특징 | **95.8%** | **0.9574** |

Fruits-360 테스트 500장의 단일 방향 분류 결과입니다. 내부 특징을 추가했을 때 정확도가 **11.4%p** 높아졌습니다. 선화 CNN은 4방향 예측 평균에서 94.0%, 학습 원본을 1,350장에서 6,431장으로 늘린 별도 실험에서 92.6%를 기록했습니다. 이 수치는 실제 사진의 탐지·분리까지 포함한 성능과 별도로 해석해야 합니다. 가림, 겹침, 절단면이 있는 사진은 탐지나 분류가 어려울 수 있습니다.

[상세 지표](metrics.csv) · [기본 학습 노트북](train.ipynb) · [데이터 확대 노트북](expanded_training.ipynb)

## 로컬 실행

Python 3.12 환경에서 저장소 루트의 터미널을 열고 실행합니다.

```sh
python -m pip install -r requirements.txt
python -m zipfile -e model-artifacts.zip .
python -m streamlit run app.py
```

`train.ipynb`에서 기본 실험의 두 CNN을 학습한 뒤 `python -m streamlit run app.py`를 실행합니다. 브라우저에서 사진을 업로드하고 **분석 시작**을 누릅니다. 첫 분석에서 Hugging Face의 탐지·분리 모델을 받으므로 인터넷 연결과 다운로드 시간이 필요합니다. CNN은 노트북에서 저장한 가중치를 불러옵니다.

기본 학습을 재현하려면 저장소 루트에서 `train.ipynb`를 실행합니다. 확대 실험은 기본 실험의 분할 자료를 사용하므로 기본 실험을 완료하고 `expanded_training.ipynb`의 1–4단계를 실행합니다. 다운로드한 사진과 모델 캐시는 로컬에 저장됩니다.

## 데이터와 모델 출처

- [Fruits-360 100×100](https://github.com/fruits-360/fruits-360-100x100)
- [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- [SlimSAM Uniform 77](https://huggingface.co/Zigeng/SlimSAM-uniform-77)
