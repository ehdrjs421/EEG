# 📕 README_ko.md (한국어)

# EEG-Seizure-Detection-JSSC

본 저장소는 **EEG 기반 뇌전증 발작 탐지 시스템**을 구현한 졸업 프로젝트 코드입니다. JSSC 수준의 SoC 기반 뇌전증 관리 시스템 구조를 참고하여, 알고리즘 구현의 충실성과 실험 재현성을 중심으로 설계되었습니다.

---

## 1. 프로젝트 개요

본 프로젝트의 주요 목표는 다음과 같습니다.
- 시간 단위 분류가 아닌 **발작 이벤트 단위 탐지**
- SoC / 임베디드 환경을 고려한 경량 특징 추출
- One-shot 학습과 online adaptation 결합
- 실험 실행과 성능 분석의 명확한 분리

실험은 **CHB-MIT Scalp EEG Database**를 사용하여 수행되었습니다.

---

## ✨ 브랜치 주요 변경 사항: feature/tca-expansion

본 브랜치는 뇌전증 발작 탐지 파이프라인의 **성능 고속화, 데이터 강건성(Robustness), 그리고 적응형 로직**에 초점을 맞춘 "차세대" 개선 사항들을 포함하고 있습니다.

- **특징 추출 8배 고속화**: NumPy `stride_tricks`를 사용한 `_windowed_sum_abs` 벡터화 계산을 통해 전처리 시간을 획기적으로 단축했습니다.
- **TCA 특징 집합 확장**: 기존 TA 특징에 **VAR (분산)** 특징을 추가하여(TA + VAR), 발작 전조 단계에서 나타나는 신호의 불규칙성을 더욱 정교하게 포착합니다.
- **강건한 전처리(Robust Preprocessing)**: **진폭 임계값 기반 아티팩트 제거**(500 μV) 로직을 통합하여, EEG 신호 내 노이즈를 자동으로 정제하고 채널 평균으로 대체합니다.
- **적응형 스코어 스무딩(Adaptive Smoothing)**: SVM 결정 스코어의 변동성과 신뢰도에 따라 스무딩 윈도우(30/60/90초)를 동적으로 변경하는 지능형 로직을 적용했습니다.
- **파이프라인 안정성 향상**: 덩어리화 임계값 상향(0.2) 및 컨텍스트 윈도우 확장(60초)을 통해 높은 탐지 민감도를 유지하면서도 오탐지(False Alarm)를 최소화했습니다.

---

## 2. 디렉토리 구조

```
EEG-Seizure-Detection-JSSC/
│
├── feature_extraction/
│   ├── preprocess.py        # 아티팩트 제거 로직 포함
│   ├── bandpass.py
│   ├── windowing.py
│   ├── tca_fe.py            # 벡터화 및 VAR 확장 구현
│   └── build_dataset.py
│
├── model/
│   ├── poly_svm.py
│   ├── sample_weighting.py
│   └── oversampling.py
│
├── post_processing/
│   ├── event_extraction.py
│   ├── post_filter.py       # 적응형 스무딩 포함
│   ├── context_filter.py
│   ├── seizure_merge.py
│   └── threshold_tuning.py
│
├── training_and_adaptation/
│   ├── one_shot_train.py
│   ├── online_tuning.py
│   ├── sampling.py
│   └── sequential_loader.py
│
├── evaluation_and_analysis/
│   ├── metrics.py
│   ├── latency.py
│   ├── resource_analysis.py
│   └── visualization.py
│
├── results/                 # 실험 결과 및 저장된 모델
├── run_experiment.py        # 메인 실행 스크립트
├── analyze_results.py      # 결과 집계 및 분석 스크립트
├── requirements.txt
└── README.md
```

---

## 3. 전체 처리 파이프라인

`run_experiment.py`에서 수행되는 전체 흐름은 다음과 같습니다.

1. **강건한 특징 추출 (Robust Feature Extraction)**
   - **아티팩트 제거**(진폭 임계값)를 포함한 EEG 전처리.
   - 다중 대역 bandpass filtering.
   - **벡터화**된 슬라이딩 윈도우 에너지 계산.
   - TCA 기반 특징 추출 (**TA + VAR** 확장).

2. **환자 단위 데이터 로딩**
   - 환자별 독립적인 학습 및 평가 수행.

3. **One-shot 학습**
   - 제한된 샘플을 사용한 초기 PolySVM 학습.
   - 클래스 불균형을 가중치 및 오버샘플링으로 보정.
   - 환자 특성에 따른 동적 가중치(dynamic positive weighting) 적용.

4. **Online Tuning**
   - 고확신 예측 피드백을 활용한 모델 적응.

5. **지능형 후처리 (Advanced Post-processing)**
   - **적응형 스무딩**: 스코어 변동성에 따른 동적 윈도우 조절.
   - **덩어리화(0.2 임계값)**: 상향된 신뢰도 기준으로 탐지 블록 병합.
   - **컨텍스트 필터(60초 윈도우)**: 확장된 패턴 분석을 통한 허위 탐지 제거.

6. **결과 저장**
   - 단계별 예측 시퀀스를 `results/` 폴더에 저장.
   - 요약 지표를 `final_result.csv`로 집계.

7. **성능 분석 및 시각화**
   - 이벤트 단위 지표 및 Latency 분석.
   - **자원 사용량 벤치마킹**: 모델 크기 및 추론 지연 시간 측정.

---

## 4. 시각화 및 분석

요약 분석 수행 및 집계 지표 생성:

```bash
python analyze_results.py
```

시각화 도구(`visualization.py`)를 통해 다음을 확인 가능합니다:
- 파이프라인 단계별 발작 탐지 타임라인 비교
- PolySVM 결정 경계 시각화
- 전체 latency 분포 히스토그램

---

## 5. 데이터셋

본 프로젝트는 CHB-MIT Scalp EEG Database를 사용합니다.
https://physionet.org/content/chbmit/1.0.0/

라이선스 문제로 데이터는 저장소에 포함되어 있지 않습니다.

## 6. 실행 환경

권장 환경: Python 3.9 이상
의존성 설치:
```bash
pip install -r requirements.txt
```

---

## 참고 논문

본 프로젝트는 다음 논문에서 제안된 **환자 맞춤형 뇌전증 관리 SoC 시스템 구조와 알고리즘 흐름**을 기반으로 구현되었습니다.

> S. Lee, J. Yoo, H.-J. Yoo  
> **"A Patient-Specific Closed-Loop Epilepsy Management SoC With One-Shot Learning and Online Tuning"**  
> IEEE Journal of Solid-State Circuits (JSSC), vol. 54, no. 1, pp. 117–129, 2019.