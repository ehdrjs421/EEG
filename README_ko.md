# EEG-Seizure-Detection-JSSC

This repository implements an EEG-based seizure detection system inspired by a
SoC-oriented epilepsy management architecture reported in JSSC-level research.

This project was conducted as a **graduation project**, with the goal of
faithfully implementing the algorithmic pipeline while restructuring the codebase
for reproducibility, modularity, and analysis clarity.

---

## 1. Project Overview

The main objectives of this project are:

- Event-level seizure detection rather than point-wise classification
- Lightweight feature extraction suitable for SoC / embedded systems
- One-shot training combined with online adaptation
- Clear separation between experiment execution and result analysis

The system is evaluated using the **CHB-MIT Scalp EEG Database**.

---

## 2. Repository Structure
# EEG-Seizure-Detection-JSSC

This repository implements an EEG-based seizure detection system inspired by a
SoC-oriented epilepsy management architecture reported in JSSC-level research.

This project was conducted as a **graduation project**, with the goal of
faithfully implementing the algorithmic pipeline while restructuring the codebase
for reproducibility, modularity, and analysis clarity.

---

## 1. Project Overview

The main objectives of this project are:

- Event-level seizure detection rather than point-wise classification
- Lightweight feature extraction suitable for SoC / embedded systems
- One-shot training combined with online adaptation
- Clear separation between experiment execution and result analysis

The system is evaluated using the **CHB-MIT Scalp EEG Database**.

---

## 2. Repository Structure
```
EEG-Seizure-Detection-JSSC/
│
├── 1_feature_extraction/
│ ├── preprocess.py
│ ├── bandpass.py
│ ├── tca_fe.py
│ └── build_dataset.py
│
├── 2_model/
│ ├── poly_svm.py
│ ├── sample_weighting.py
│ └── oversampling.py
│
├── 3_post_processing/
│ ├── event_extraction.py
│ ├── post_filter.py
│ └── threshold_tuning.py
│
├── 4_training_and_adaptation/
│ ├── one_shot_train.py
│ ├── online_tuning.py
│ └── sequential_loader.py
│
├── 5_evaluation_and_analysis/
│ ├── metrics.py
│ ├── latency.py
│ ├── event_analysis.py
│ ├── visualization.py
│ └── analyze_results.py
│
├── run_experiment.py
├── requirements.txt
└── README.md
EEG-Seizure-Detection-JSSC/
│
├── 1_feature_extraction/
│ ├── preprocess.py
│ ├── bandpass.py
│ ├── tca_fe.py
│ └── build_dataset.py
│
├── 2_model/
│ ├── poly_svm.py
│ ├── sample_weighting.py
│ └── oversampling.py
│
├── 3_post_processing/
│ ├── event_extraction.py
│ ├── post_filter.py
│ └── threshold_tuning.py
│
├── 4_training_and_adaptation/
│ ├── one_shot_train.py
│ ├── online_tuning.py
│ └── sequential_loader.py
│
├── 5_evaluation_and_analysis/
│ ├── metrics.py
│ ├── latency.py
│ ├── event_analysis.py
│ ├── visualization.py
│ └── analyze_results.py
│
├── run_experiment.py
├── requirements.txt
└── README.md
```

---

## 3. Processing Pipeline

The full pipeline executed in `run_experiment.py` consists of:

1. **Feature Extraction**
   - EEG preprocessing (channel selection, resampling)
   - Multi-band bandpass filtering
   - TCA-based feature extraction with context windows

2. **Patient-wise Data Loading**
   - Data is processed independently for each patient

3. **One-shot Training**
   - Initial PolySVM training using limited seizure samples
   - Class imbalance handled by weighting and oversampling

4. **Online Tuning**
   - High-confidence predictions are reused to adapt the model incrementally

5. **Result Storage**
   - Time-wise prediction sequences are saved per patient
   - Summary metrics are aggregated into a single CSV file

6. **Evaluation and Analysis**
   - Event-level detection performance
   - Latency analysis
   - Visualization of prediction timelines

---

## 4. Event-level Detection Strategy

Instead of evaluating individual time points, this project focuses on
**seizure events as contiguous temporal blocks**.

- A seizure event is considered detected if at least one positive prediction
  occurs within the event window.
- Detection latency is defined as the time difference between seizure onset
  and the first correct detection.

This approach better reflects clinical relevance and real-world system behavior.

---

## 5. Output Files

### 5.1 Prediction Sequences

For each patient:
```
pred_sequence_<patient_id>.csv
```

Columns include:
- `time_idx`
- `y_true`
- `y_pred_before`
- `y_pred_after`
- `decision_score`
- `patient`

These files are used for visualization and event-level analysis.

---

### 5.2 Summary Results
```
final_result.csv
```
Contains patient-wise aggregated metrics:
- Accuracy, Precision, Recall, F1-score
- Event-level sensitivity
- Latency statistics
- Resource usage metrics

---

## 6. Visualization

Visualization utilities are provided in:
```
5_evaluation_and_analysis/visualization.py
```

Example analyses:
- Seizure detection timelines (before vs after online tuning)
- Event-level latency plots
- Latency distribution histograms

To run analysis:

```bash
python 5_evaluation_and_analysis/analyze_results.py
```

## 7. Dataset

This project uses the CHB-MIT Scalp EEG Database.

Dataset access:
https://physionet.org/content/chbmit/1.0.0/

Due to licensing restrictions, the dataset is not included in this repository.

## 8. Environment Setup

Recommended environment:

Python 3.9+

NumPy, SciPy, scikit-learn

MNE

Matplotlib, Pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

## 9. Notes

This repository prioritizes implementation fidelity and system-level analysis.

The original experimental logic is preserved as much as possible.

Performance optimization is not the primary goal.

## 10. Future Work

Real-time streaming inference

Embedded hardware evaluation

Adaptive thresholding strategies

Comparison with deep learning approaches

## Reference

This project is inspired by the following research work:

> S. Lee, J. Yoo, and H.-J. Yoo,  
> **"A Patient-Specific Closed-Loop Epilepsy Management SoC With One-Shot Learning and Online Tuning,"**  
> IEEE Journal of Solid-State Circuits (JSSC), vol. 54, no. 1, pp. 117–129, Jan. 2019.

The goal of this repository is to reproduce and study the **algorithmic pipeline**
proposed in the paper, including:

- Patient-specific seizure detection
- One-shot learning using limited seizure samples
- Online model adaptation (online tuning)
- Event-level seizure detection and latency evaluation

While the original work focuses on a **hardware SoC implementation**, this project
reconstructs the **algorithmic workflow in software** using the CHB-MIT EEG dataset.


---

---

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
├── 1_feature_extraction/
│ ├── preprocess.py
│ ├── bandpass.py
│ ├── tca_fe.py
│ └── build_dataset.py
│
├── 2_model/
│ ├── poly_svm.py
│ ├── sample_weighting.py
│ └── oversampling.py
│
├── 3_post_processing/
│ ├── event_extraction.py
│ ├── post_filter.py
│ └── threshold_tuning.py
│
├── 4_training_and_adaptation/
│ ├── one_shot_train.py
│ ├── online_tuning.py
│ └── sequential_loader.py
│
├── 5_evaluation_and_analysis/
│ ├── metrics.py
│ ├── latency.py
│ ├── event_analysis.py
│ ├── visualization.py
│ └── analyze_results.py
│
├── run_experiment.py
├── requirements.txt
└── README.md
```


---

## 3. 전체 처리 파이프라인

`run_experiment.py`에서 수행되는 전체 흐름은 다음과 같습니다.

1. **특징 추출**
   - EEG 전처리 (채널 선택, 리샘플링)
   - 다중 대역 bandpass filtering
   - TCA 기반 특징 추출 및 context window 구성

2. **환자 단위 데이터 로딩**
   - 환자별 독립적인 학습 및 평가 수행

3. **One-shot 학습**
   - 제한된 발작 샘플을 사용한 초기 PolySVM 학습
   - 클래스 불균형을 가중치 및 오버샘플링으로 보정

4. **Online Tuning**
   - 신뢰도가 높은 예측 결과를 활용한 모델 적응

5. **결과 저장**
   - 환자별 시간 단위 예측 결과 저장
   - 요약 성능 지표를 CSV 파일로 집계

6. **성능 분석 및 시각화**
   - 이벤트 단위 탐지 성능
   - latency 분석
   - 예측 타임라인 시각화

---

## 4. 이벤트 단위 발작 탐지 전략

본 연구는 개별 시점 분류가 아닌,
**연속된 발작 구간(이벤트) 단위 탐지**를 목표로 합니다.

- 발작 구간 내 한 번이라도 탐지되면 해당 이벤트를 탐지 성공으로 간주
- latency는 발작 시작 시점부터 최초 탐지 시점까지의 시간으로 정의

이는 실제 임상 환경과 SoC 기반 시스템 요구사항을 반영한 설계입니다.

---

## 5. 출력 파일 구성

### 5.1 환자별 예측 시퀀스
```
pred_sequence_<patient_id>.csv
```

포함 컬럼:
- `time_idx`
- `y_true`
- `y_pred_before`
- `y_pred_after`
- `decision_score`
- `patient`

해당 파일은 타임라인 시각화 및 이벤트 분석에 사용됩니다.

---

### 5.2 전체 요약 결과
```
final_result.csv
```

환자 단위로 집계된 다음 정보가 포함됩니다.

- 정확도, 정밀도, 재현율, F1-score
- 이벤트 단위 민감도
- latency 통계
- 모델 자원 사용량

---

## 6. 시각화 및 분석

시각화 코드는 다음 경로에 위치합니다.
```
5_evaluation_and_analysis/visualization.py
```

제공되는 분석 예시는 다음과 같습니다.

- Online tuning 전/후 발작 탐지 타임라인
- 이벤트 단위 latency 그래프
- 전체 latency 분포 히스토그램

분석 실행:

```bash
python 5_evaluation_and_analysis/analyze_results.py
```

## 7. 데이터셋

본 프로젝트는 CHB-MIT Scalp EEG Database를 사용합니다.

https://physionet.org/content/chbmit/1.0.0/

라이선스 문제로 데이터는 저장소에 포함되어 있지 않습니다.

## 8. 실행 환경

권장 환경:

Python 3.9 이상

NumPy, SciPy, scikit-learn

MNE

Matplotlib, Pandas

의존성 설치:
```bash
pip install -r requirments.txt
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