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
├── results/                 # Experiment results and saved models
├── run_experiment.py        # Main execution script
├── analyze_results.py      # Result aggregation and analysis script
├── requirements.txt
└── README.md
```

---

## 3. Processing Pipeline

The full pipeline executed in `run_experiment.py` consists of:

1. **Feature Extraction**
   - EEG preprocessing (channel selection, resampling)
   - Multi-band bandpass filtering
   - Sliding window energy calculation (`windowing.py`)
   - TCA-based feature extraction with context windows

2. **Patient-wise Data Loading**
   - Data is processed independently for each patient

3. **One-shot Training**
   - Initial PolySVM training using limited seizure samples
   - Class imbalance handled by weighting and oversampling
   - Dynamic positive weighting according to patient characteristics

4. **Online Tuning**
   - High-confidence predictions are used to adapt the model incrementally

5. **Post-processing Enhancements**
   - **Seizure Merge**: Groups contiguous prediction blocks using adaptive gaps.
   - **Context Filter**: Removes False Alarms by analyzing signal patterns prior to detection events.

6. **Result Storage**
   - Time-wise prediction sequences (before/after/merged/filtered) are saved per patient
   - Trained models and threshold parameters are stored in the `results/` folder

7. **Evaluation and Analysis**
   - Event-level detection performance (Sensitivity, FA/h)
   - Latency analysis (Detection latency, early detection rate)
   - **Resource Analysis**: Model size (KB) and inference latency measurements
   - Visualization of prediction timelines

---

## 4. Event-level Detection Strategy

Instead of evaluating individual time points, this project focuses on
**seizure events as contiguous temporal blocks**.

- A seizure event is considered detected if the prediction sequence within the event window satisfies specific criteria (e.g., vector-based sensitivity).
- Detection latency is defined as the time difference between seizure onset
  and the first correct detection.
- Post-processing tools like `seizure_merge` and `context_filter` are used to refine detection outcomes and reduce false alarms.

---

## 5. Output Files

### 5.1 Prediction Sequences

For each patient:
```
results/pred_sequence_<patient_id>.csv
```

Columns include:
- `time_idx`, `y_true`
- `y_pred_before` (Initial/Online result)
- `y_pred_merged` (After Seizure Merge)
- `y_pred_filtered` (Final result after Context Filter)
- `decision_score`, `patient`

---

### 5.2 Summary Results
```
results/final_result.csv
```
Contains patient-wise aggregated metrics:
- Accuracy, Precision, Recall, F1-score
- Event-level sensitivity, FA per hour
- Latency statistics (Mean, Median)
- Resource usage (Model size, prediction time)

---

## 6. Visualization and Analysis

To run summary analysis and generate aggregated metrics:

```bash
python analyze_results.py
```

Visualization utilities (`visualization.py`) are used to plot:
- Seizure detection timelines (comparison of pipeline stages)
- PolySVM decision boundaries
- Latency distribution histograms

---

## 7. Dataset

This project uses the CHB-MIT Scalp EEG Database.

Dataset access:
https://physionet.org/content/chbmit/1.0.0/

Due to licensing restrictions, the dataset is not included in this repository.

## 8. Environment Setup

Recommended environment: Python 3.9+

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
