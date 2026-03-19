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
