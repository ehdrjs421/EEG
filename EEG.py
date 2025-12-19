import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def extract_seizure_events(y_true):
    """
    발작이 시작된 구간들을 리스트로 반환: [(start_idx, end_idx), ...]
    """
    events = []
    in_event = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_event:
            start = i
            in_event = True
        elif y_true[i] == 0 and in_event:
            end = i - 1
            events.append((start, end))
            in_event = False
    if in_event:
        events.append((start, len(y_true) - 1))
    return events

def apply_post_filter(pred, min_consec=3):
    """
    min_consec 길이 이상으로 연속된 1만 유지하고 나머지는 0으로
    """
    filtered = np.zeros_like(pred)
    i = 0
    while i < len(pred):
        if pred[i] == 1:
            count = 1
            while (i + count < len(pred)) and (pred[i + count] == 1):
                count += 1
            if count >= min_consec:
                filtered[i:i+count] = 1
            i += count
        else:
            i += 1
    return filtered

def tune_post_filter(y_true, y_pred_raw, max_window=10, metric='f1'):
    scores = []
    best_score = -1
    best_k = 1
    best_pred = None
    for k in range(1, max_window + 1):
        y_filtered = apply_post_filter(y_pred_raw, min_consec=k)
        report = classification_report(y_true, y_filtered, output_dict=True, zero_division=0)
        sensitivity = report['1.0']['recall']
        specificity = report['0.0']['recall']
        f1 = report['1.0']['f1-score']
        scores.append({'k': k, 'sensitivity': sensitivity, 'specificity': specificity, 'f1': f1})
        if report['1.0'][f'{metric}'] > best_score:
            best_score = report['1.0'][f'{metric}']
            best_k = k
            best_pred = y_filtered
    return best_k, best_score, best_pred, pd.DataFrame(scores)

def compute_sample_weights(y_true, info, base_weight=1.0, max_weight=10.0):
    """
    발작 샘플에 대해 latency 정보 기반 가중치 벡터 생성
    info: 각 샘플이 어떤 환자/파일/윈도우인지 담긴 DataFrame
    """
    sample_weights = np.ones(len(y_true)) * base_weight

    # 파일별로 발작 구간 추출
    for (patient, file), group in info.groupby(['patient', 'file']):
        y_file = y_true[group.index]
        events = extract_seizure_events(y_file.values)
        for start, end in events:
            duration = end - start + 1
            for i in range(start, end + 1):
                rel_pos = (i - start) / duration  # 발작 구간 내 상대 위치
                weight = base_weight + (max_weight - base_weight) * rel_pos
                sample_weights[group.index[i]] = weight

    return sample_weights

def oversample_seizure(X, y, ratio=5):
    X_aug, y_aug = [], []
    for xi, yi in zip(X, y):
        X_aug.append(xi)
        y_aug.append(yi)
        if yi == 1:
            for _ in range(ratio - 1):
                X_aug.append(xi)
                y_aug.append(yi)
    return np.array(X_aug), np.array(y_aug)

def compute_latency_in_event(y_true, y_pred, step_sec=1):
    try:
        seizure_idxs = np.where(y_true == 1)[0]
        pred_idxs = np.where(y_pred == 1)[0]
        seizure_start = seizure_idxs[0]
        seizure_end = seizure_idxs[-1]
        in_seizure_preds = pred_idxs[(pred_idxs >= seizure_start) & (pred_idxs <= seizure_end)]
        if len(in_seizure_preds) == 0:
            return None
        return (in_seizure_preds[0] - seizure_start) * step_sec
    except:
        return None

def compute_latency_per_event(y_true, y_pred, step_sec=1):
    """
    복수 발작(event) 각각에 대해 latency 계산
    latency = 최초 감지 지점 - 발작 시작 지점 (초 단위)
    """
    events = extract_seizure_events(y_true)
    latencies = []

    for start, end in events:
        pred_idxs = np.where(y_pred == 1)[0]
        in_event_preds = pred_idxs[(pred_idxs >= start) & (pred_idxs <= end)]
        if len(in_event_preds) > 0:
            latency = (in_event_preds[0] - start) * step_sec
            latencies.append(latency)
        else:
            # 감지 실패한 이벤트는 제외하거나 NaN 처리 가능
            latencies.append(None)

    return latencies

def evaluate_vector_based_detection(y_true, y_pred, threshold=0.6):
    """
    벡터 기반 발작 감지 평가
    - 발작 벡터의 일정 비율 이상 탐지되어야 감지 성공으로 간주
    - threshold = 0.6 이면, 60% 이상 탐지 시 성공
    """
    events = extract_seizure_events(y_true)
    if len(events) == 0:
        return None

    detected = 0
    for start, end in events:
        duration = end - start + 1
        detected_in_event = np.sum(y_pred[start:end+1])
        if duration > 0 and detected_in_event / duration >= threshold:
            detected += 1

    return detected / len(events)

def stratified_time_sampling(indices, total_len, n_samples, n_bins=10):
    bin_size = total_len // n_bins
    sampled = []

    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, total_len)
        candidates = [idx for idx in indices if start <= idx < end]
        if candidates:
            sampled += np.random.choice(candidates, min(len(candidates), max(1, n_samples // n_bins)), replace=False).tolist()

    return sampled

class PolySVM:
    def __init__(self, degree=2, coef0=1, C=1.0, gamma=1.0, lr=0.001, n_iters=1000, loss_weight=False, pos_weight=5.0):
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.gamma = gamma
        self.lr = lr
        self.n_iters = n_iters
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight


    def _poly_kernel(self, X1, X2):
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree

    def fit(self, X, y):
        n_samples = X.shape[0]
        y = np.where(y <= 0, -1, 1)
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        K = self._poly_kernel(X, X)

        for _ in range(self.n_iters):
            for i in range(n_samples):
                margin = y[i] * np.sum(self.alpha * y * K[:, i])

                if margin < 1:
                    weight = self.pos_weight if (self.loss_weight and y[i] == 1) else 1.0
                    self.alpha[i] += self.lr * weight

                self.alpha[i] = min(self.alpha[i], self.C)



    def prune_support_vectors(self, threshold=1e-3):
        """Alpha가 작아 기여도가 낮은 support vector 제거"""
        keep_idx = np.where(np.abs(self.alpha) > threshold)[0]
        pruned_count = len(self.alpha) - len(keep_idx)
        self.X = self.X[keep_idx]
        self.y = self.y[keep_idx]
        self.alpha = self.alpha[keep_idx]
        print(f"🔧 Pruned {pruned_count} support vectors → 남은 SV: {len(self.alpha)}")

    def project(self, X):
        K = self._poly_kernel(X, self.X)  # shape: (n_test, n_train)
        return np.dot(K, self.alpha * self.y)

    def predict(self, X):
        return np.where(self.project(X) >= 0, 1, 0)

    def decision_function(self, X):
        return self.project(X)
