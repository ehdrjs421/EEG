# 📕 README_ko.md (한국어)

# EEG-Seizure-Detection-JSSC

본 저장소는 **EEG 기반 뇌전증 발작 탐지 시스템**을 구현한 졸업 프로젝트 코드입니다.
JSSC 수준의 SoC 기반 뇌전증 관리 시스템 구조를 참고하여,
알고리즘 구현의 충실성과 실험 재현성을 중심으로 설계되었습니다.

---

## 1. 프로젝트 개요

본 프로젝트의 주요 목표는 다음과 같습니다.

- 시간 단위 분류가 아닌 **발작 이벤트 단위 탐지**
- SoC / 임베디드 환경을 고려한 경량 특징 추출
- One-shot 학습과 online adaptation 결합
- 실험 실행과 성능 분석의 명확한 분리

실험은 **CHB-MIT Scalp EEG Database**를 사용하여 수행되었습니다.

---

## 2. 디렉토리 구조

```
EEG-Seizure-Detection-JSSC/
│
├── feature_extraction/
│   ├── preprocess.py
│   ├── bandpass.py
│   ├── windowing.py
│   ├── tca_fe.py
│   └── build_dataset.py
│
├── model/
│   ├── poly_svm.py
│   ├── sample_weighting.py
│   └── oversampling.py
│
├── post_processing/
│   ├── event_extraction.py
│   ├── post_filter.py
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

1. **특징 추출**
   - EEG 전처리 (채널 선택, 리샘플링)
   - 다중 대역 bandpass filtering
   - 슬라이딩 윈도우 기반 에너지 계산 (`windowing.py`)
   - TCA 기반 특징 추출 및 context window 구성

2. **환자 단위 데이터 로딩**
   - 환자별 독립적인 학습 및 평가 수행

3. **One-shot 학습**
   - 제한된 발작 샘플을 사용한 초기 PolySVM 학습
   - 클래스 불균형을 가중치 및 오버샘플링으로 보정
   - 환자 특성에 따른 동적 가중치(dynamic positive weighting) 적용

4. **Online Tuning**
   - 신뢰도가 높은 예측 결과를 활용한 모델 적응

5. **후처리 개선 (Post-processing Enhancements)**
   - **Seizure Merge (덩어리화)**: 적응적 간격을 사용하여 인접한 예측 블록을 하나의 이벤트로 병합.
   - **Context Filter**: 탐지 이벤트 직전의 신호 패턴을 분석하여 허위 탐지(False Alarm) 제거.

6. **결과 저장**
   - 환자별 시간 단위 예측 결과 (전/후/병합/필터링) 저장
   - 학습된 모델 및 임계값 파라미터 `results/` 폴더 내 저장

7. **성능 분석 및 시각화**
   - 이벤트 단위 탐지 성능 (Sensitivity, FA/h)
   - Latency 분석 (탐지 지연 시간, 조기 탐지율)
   - **자원 분석 (Resource Analysis)**: 모델 크기(KB) 및 추론 지연 시간 측정
   - 예측 타임라인 시각화

---

## 4. 이벤트 단위 발작 탐지 전략

본 연구는 개별 시점 분류가 아닌,
**연속된 발작 구간(이벤트) 단위 탐지**를 목표로 합니다.

- 발작 구간 내 예측 시퀀스가 특정 기준(예: vector-based sensitivity)을 만족하면 탐지 성공으로 간주
- latency는 발작 시작 시점부터 최초 탐지 시점까지의 시간으로 정의
- `seizure_merge` 및 `context_filter` 등의 후처리 도구를 사용하여 탐지 결과를 정제하고 허위 탐지를 줄임

---

## 5. 출력 파일 구성

### 5.1 환자별 예측 시퀀스
```
results/pred_sequence_<patient_id>.csv
```

포함 컬럼:
- `time_idx`, `y_true`
- `y_pred_before` (초기/온라인 튜닝 결과)
- `y_pred_merged` (덩어리화 적용 결과)
- `y_pred_filtered` (컨텍스트 필터 적용 후 최종 결과)
- `decision_score`, `patient`

---

### 5.2 전체 요약 결과
```
results/final_result.csv
```

환자 단위로 집계된 다음 정보가 포함됩니다.

- 정확도, 정밀도, 재현율, F1-score
- 이벤트 단위 민감도, 시간당 허위 탐지율(FA/h)
- Latency 통계 (평균, 중앙값)
- 모델 자원 사용량 (모델 크기, 예측 시간)

---

## 6. 시각화 및 분석

요약 분석을 수행하고 집계된 지표를 생성하려면:

```bash
python analyze_results.py
```

시각화 도구(`visualization.py`)를 통해 다음을 확인 가능합니다:
- 파이프라인 단계별 발작 탐지 타임라인 비교
- PolySVM 결정 경계 시각화
- 전체 latency 분포 히스토그램

---

## 7. 데이터셋

본 프로젝트는 CHB-MIT Scalp EEG Database를 사용합니다.

https://physionet.org/content/chbmit/1.0.0/

라이선스 문제로 데이터는 저장소에 포함되어 있지 않습니다.

## 8. 실행 환경

권장 환경: Python 3.9 이상

의존성 설치:
```bash
pip install -r requirements.txt
```

## 9. 참고 사항

본 저장소는 성능 경쟁보다는 구현 충실도와 구조적 설계를 중시합니다.

졸업 프로젝트 당시 사용한 알고리즘 흐름을 최대한 유지했습니다.

## 10. 향후 확장 방향

실시간 스트리밍 추론 구조 확장

임베디드 하드웨어 기반 실험

적응적 임계값 조정 기법

딥러닝 기반 방법과의 비교

## 참고 논문

본 프로젝트는 다음 논문에서 제안된 **환자 맞춤형 뇌전증 관리 SoC 시스템 구조와
알고리즘 흐름**을 기반으로 구현되었습니다.

> S. Lee, J. Yoo, H.-J. Yoo  
> **"A Patient-Specific Closed-Loop Epilepsy Management SoC With One-Shot Learning and Online Tuning"**  
> IEEE Journal of Solid-State Circuits (JSSC), vol. 54, no. 1, pp. 117–129, 2019.

해당 논문은 EEG 신호를 이용한 **환자 맞춤형 발작 탐지 시스템**을 제안하며,
다음과 같은 핵심 개념을 포함합니다.

- One-shot learning 기반 초기 모델 학습
- Online tuning을 통한 환자 적응
- 실시간 시스템을 고려한 발작 탐지 구조
- SoC 기반 폐루프(closed-loop) 뇌전증 관리 시스템

본 저장소에서는 논문의 **하드웨어 구현을 재현하는 것이 아니라**,  
제안된 **알고리즘 파이프라인을 소프트웨어 환경에서 구현하고 실험적으로 검증하는 것**을 목표로 합니다.