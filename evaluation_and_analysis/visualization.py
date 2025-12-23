import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca_decision_boundary(
    X_train,
    y_train,
    X_test,
    y_test,
    svm
):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.decision_function(
        pca.inverse_transform(grid)
    ).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)
    plt.scatter(
        X_test_pca[y_test == 0, 0],
        X_test_pca[y_test == 0, 1],
        c='white', edgecolor='k', label='Non-Seizure'
    )
    plt.scatter(
        X_test_pca[y_test == 1, 0],
        X_test_pca[y_test == 1, 1],
        c='blue', edgecolor='k', label='Seizure'
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
