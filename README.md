# EEG-Seizure-Detection-JSSC

This repository implements an EEG-based seizure detection system inspired by a SoC-oriented epilepsy management architecture reported in JSSC-level research.

This project was conducted as a **graduation project**, with the goal of faithfully implementing the algorithmic pipeline while restructuring the codebase for reproducibility, modularity, and analysis clarity.

---

## 1. Project Overview

The main objectives of this project are:
- Event-level seizure detection rather than point-wise classification
- Lightweight feature extraction suitable for SoC / embedded systems
- One-shot training combined with online adaptation
- Clear separation between experiment execution and result analysis

The system is evaluated using the **CHB-MIT Scalp EEG Database**.

---

## ✨ Branch Highlights: feature/tca-expansion

This branch introduces several "next-generation" improvements to the seizure detection pipeline, focusing on performance, robustness, and adaptive logic.

- **8x Faster Feature Extraction**: Vectorized `_windowed_sum_abs` calculation using NumPy `stride_tricks`, significantly reducing preprocessing time.
- **Expanded TCA Features**: Addition of **VAR (Variance)** to the feature vector (TA + VAR), capturing temporal signal irregularities common in pre-seizure stages.
- **Robust Preprocessing**: Integrated **Amplitude Thresholding** (500 μV) with mean substitution to automatically handle and clean EEG artifacts.
- **Adaptive Score Smoothing**: Dynamic logic that adjusts the smoothing window (30/60/90s) based on SVM decision score volatility and confidence levels.
- **Improved Pipeline Stability**: Higher merge thresholds (0.2) and extended context windows (60s) to minimize false alarms while maintaining detection sensitivity.

---

## 2. Repository Structure

```
EEG-Seizure-Detection-JSSC/
│
├── feature_extraction/
│   ├── preprocess.py        # Includes artifact removal
│   ├── bandpass.py
│   ├── windowing.py
│   ├── tca_fe.py            # Vectorized & VAR-expanded extraction
│   └── build_dataset.py
│
├── model/
│   ├── poly_svm.py
│   ├── sample_weighting.py
│   └── oversampling.py
│
├── post_processing/
│   ├── event_extraction.py
│   ├── post_filter.py       # Includes adaptive smoothing
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

1. **Robust Feature Extraction**
   - EEG preprocessing with **Artifact Removal** (Amplitude Thresholding).
   - Multi-band bandpass filtering.
   - **Vectorized** sliding window energy calculation.
   - TCA-based feature extraction (TA + **VAR** expansion).

2. **Patient-wise Data Loading**
   - Data is processed independently for each patient.

3. **One-shot Training**
   - Initial PolySVM training using limited samples.
   - Class imbalance handled by weighting and oversampling.
   - Dynamic positive weighting according to patient characteristics.

4. **Online Tuning**
   - Model adaptation using high-confidence prediction feedback.

5. **Advanced Post-processing**
   - **Adaptive Smoothing**: Dynamic window adjustment based on score volatility.
   - **Seizure Merge (0.2 Threshold)**: Grouping detection blocks with increased confidence requirements.
   - **Context Filter (60s Window)**: Using extended patterns to detect and remove False Alarms.

6. **Result Storage**
   - Multi-stage prediction sequences saved to `results/`.
   - Aggregate metrics collected in `final_result.csv`.

7. **Evaluation and Analysis**
   - Event-level metrics & Latency analysis.
   - **Resource Benchmarking**: Model size and inference latency measurements.

---

## 4. Visualization and Analysis

To run summary analysis and generate aggregated metrics:

```bash
python analyze_results.py
```

Visualization (`visualization.py`) supports:
- Multi-stage detection timelines.
- SVM boundary plots.
- Performance distribution analysis.

---

## 5. Dataset

This project uses the CHB-MIT Scalp EEG Database.
Dataset access: https://physionet.org/content/chbmit/1.0.0/

Due to licensing restrictions, the dataset is not included in this repository.

## 6. Environment Setup

Recommended environment: Python 3.9+
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Reference

This project is inspired by the following research work:

> S. Lee, J. Yoo, and H.-J. Yoo,  
> **"A Patient-Specific Closed-Loop Epilepsy Management SoC With One-Shot Learning and Online Tuning,"**  
> IEEE Journal of Solid-State Circuits (JSSC), vol. 54, no. 1, pp. 117–129, Jan. 2019.
