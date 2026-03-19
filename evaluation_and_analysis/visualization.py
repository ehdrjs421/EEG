import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA


# =====================================================================
# 기존 함수 (유지)
# =====================================================================

def plot_polysvm_decision_boundary(
    X, y, svm_model,
    save_path=None, use_pca=True, grid_resolution=300
):
    if use_pca:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
    else:
        if X.shape[1] != 2:
            raise ValueError("X must be 2D if use_pca=False")
        X_vis = X

    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid) if use_pca else grid
    Z = svm_model.decision_function(grid_original).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    markers = {0: 'o', 1: 's'}
    labels  = {0: 'Non-seizure', 1: 'Seizure'}
    for cls in np.unique(y):
        idx = np.where(y == cls)
        plt.scatter(X_vis[idx, 0], X_vis[idx, 1],
                    marker=markers[cls], edgecolors='k',
                    facecolors='none', label=labels[cls])
    if hasattr(svm_model, "X"):
        sv = svm_model.X
        sv_vis = pca.transform(sv) if use_pca else sv
        plt.scatter(sv_vis[:, 0], sv_vis[:, 1], s=150,
                    facecolors='none', edgecolors='k',
                    linewidths=1.2, label='Support Vectors')
    plt.contour(xx, yy, Z, levels=[-1, 0, 1],
                linestyles=['--', '-', '--'], colors='k')
    plt.title("PolySVM Decision Boundary (2D Projection)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# =====================================================================
# ✨ 1. 예측 타임라인 시각화
#    y_true / y_pred_before / y_pred_merged 를 발작 구간 중심으로 표시
# =====================================================================

def plot_prediction_timeline(
    pred_df,
    patient_id,
    save_path=None,
    max_points=3000
):
    """
    한 환자의 예측 타임라인을 3개 레이어로 시각화합니다.
    - Row 1: y_true (실제 발작)
    - Row 2: y_pred_after (덩어리화 전)
    - Row 3: y_pred_merged (덩어리화 후)
    decision_score를 배경 음영으로 함께 표시합니다.

    Parameters
    ----------
    pred_df : pd.DataFrame
        pred_sequence_{patient_id}.csv 로드한 데이터프레임
    patient_id : str
    save_path : str or None
    max_points : int
        너무 긴 시퀀스는 다운샘플링 (기본 3000포인트)
    """
    df = pred_df.copy()
    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step].reset_index(drop=True)

    t = df['time_idx'].values
    y_true   = df['y_true'].values
    y_before = df['y_pred_before'].values
    y_after = df['y_pred_after'].values
    y_merged = df['y_pred_merged'].values
    scores   = df['decision_score'].values

    # score 정규화 (0~1)
    s_min, s_max = scores.min(), scores.max()
    scores_norm = (scores - s_min) / (s_max - s_min + 1e-8)

    fig, axes = plt.subplots(4, 1, figsize=(16, 6), sharex=True)
    fig.suptitle(f"Prediction Timeline — {patient_id}", fontsize=13, fontweight='bold')

    rows = [
        (y_true,   'y_true',          '#2ecc71', '#27ae60'),
        (y_before, 'Before',    '#3498db', '#2980b9'),
        (y_after, 'After',    '#3498db', '#2980b9'),
        (y_merged, 'After Merge ✨',  '#e74c3c', '#c0392b'),
    ]

    for ax, (signal, label, face_col, edge_col) in zip(axes, rows):
        # decision_score 배경 음영
        ax.fill_between(t, 0, scores_norm * 0.4,
                        color='#f39c12', alpha=0.25, label='Decision Score')
        # 발작 구간 강조
        ax.fill_between(t, 0, signal,
                        step='post', color=face_col, alpha=0.7, label=label)
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 1])
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='x', alpha=0.3)

    axes[-1].set_xlabel("Time Index (steps)", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =====================================================================
# ✨ 2. 덩어리화 전후 이벤트 비교 시각화
#    발작 이벤트를 블록으로 표시 — 병합된 구간 강조
# =====================================================================

def plot_event_comparison(
    pred_df,
    patient_id,
    save_path=None
):
    """
    발작 이벤트를 블록으로 표시하여 덩어리화 전후를 직관적으로 비교합니다.

    Parameters
    ----------
    pred_df : pd.DataFrame
    patient_id : str
    save_path : str or None
    """
    from post_processing.event_extraction import extract_seizure_events

    y_true   = pred_df['y_true'].values
    y_before = pred_df['y_pred_after'].values
    y_merged = pred_df['y_pred_merged'].values
    T        = len(y_true)

    events_true   = extract_seizure_events(y_true)
    events_before = extract_seizure_events(y_before)
    events_merged = extract_seizure_events(y_merged)

    fig, ax = plt.subplots(figsize=(16, 3.5))
    fig.suptitle(f"Event Block Comparison — {patient_id}", fontsize=13, fontweight='bold')

    # 레이어별 y 위치
    layers = [
        (events_true,   2.0, '#2ecc71', 'y_true'),
        (events_before, 1.0, '#3498db', 'Before Merge'),
        (events_merged, 0.0, '#e74c3c', 'After Merge ✨'),
    ]

    bar_height = 0.6
    for events, y_pos, color, label in layers:
        for start, end in events:
            ax.barh(y_pos, end - start, left=start,
                    height=bar_height, color=color, alpha=0.8, edgecolor='white')
        # 레이블
        ax.text(-T * 0.01, y_pos, label,
                va='center', ha='right', fontsize=9, fontweight='bold')

    # 병합된 구간 표시 (before에 없고 merged에 있는 부분)
    before_mask = y_before.copy()
    merged_only = np.clip(y_merged - before_mask, 0, 1)
    merged_events_only = extract_seizure_events(merged_only)
    for start, end in merged_events_only:
        ax.barh(0.0, end - start, left=start,
                height=bar_height, color='#f39c12',
                alpha=0.9, edgecolor='white', hatch='//')

    ax.set_xlim(0, T)
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.set_xlabel("Time Index (steps)", fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 범례
    patches = [
        mpatches.Patch(color='#2ecc71', label='y_true'),
        mpatches.Patch(color='#3498db', label='Before Merge'),
        mpatches.Patch(color='#e74c3c', label='After Merge'),
        mpatches.Patch(color='#f39c12', hatch='//', label='Merged Gap'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =====================================================================
# ✨ 3. 환자별 성능 지표 비교 막대그래프 (전체 평균 포함)
# =====================================================================

def plot_metrics_comparison(
    result_csv_path,
    save_path=None
):
    """
    final_result.csv를 읽어 환자별 + 전체 평균 성능 지표를
    before / merged 쌍으로 비교하는 막대그래프를 생성합니다.

    Parameters
    ----------
    result_csv_path : str
        final_result.csv 경로
    save_path : str or None
    """
    df = pd.read_csv(result_csv_path)

    metrics = [
        ('sensitivity_before', 'sensitivity_merged', 'Sensitivity'),
        ('specificity_before', 'specificity_merged', 'Specificity'),
        ('f1_seizure_before',  'f1_seizure_merged',  'F1 (Seizure)'),
        ('vec_sens_before',    'vec_sens_merged',     'Vec Sensitivity'),
    ]

    # 전체 평균 행 추가
    mean_row = df[
        [col for pair in metrics for col in pair[:2]]
    ].mean().to_frame().T
    mean_row['patient'] = 'MEAN'
    df_plot = pd.concat([df, mean_row], ignore_index=True)

    patients = df_plot['patient'].tolist()
    n = len(patients)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    fig.suptitle("Performance Comparison: Before vs After Merge (per Patient)",
                 fontsize=14, fontweight='bold')

    colors = {'before': '#3498db', 'merged': '#e74c3c'}

    for ax, (col_before, col_merged, title) in zip(axes.flat, metrics):
        vals_before = df_plot[col_before].astype(float).values
        vals_merged = df_plot[col_merged].astype(float).values

        bars_b = ax.bar(x - width / 2, vals_before, width,
                        label='Before Merge', color=colors['before'], alpha=0.8)
        bars_m = ax.bar(x + width / 2, vals_merged, width,
                        label='After Merge ✨', color=colors['merged'], alpha=0.8)

        # MEAN 강조
        ax.bar(x[-1] - width / 2, vals_before[-1], width,
               color=colors['before'], alpha=1.0, edgecolor='black', linewidth=1.5)
        ax.bar(x[-1] + width / 2, vals_merged[-1], width,
               color=colors['merged'], alpha=1.0, edgecolor='black', linewidth=1.5)

        # 값 표시 (MEAN만)
        ax.text(x[-1] - width / 2, vals_before[-1] + 0.005,
                f"{vals_before[-1]:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(x[-1] + width / 2, vals_merged[-1] + 0.005,
                f"{vals_merged[-1]:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(patients, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axvline(x[-1] - width, color='gray', linestyle='--', alpha=0.5)  # MEAN 구분선

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()