import numpy as np

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
