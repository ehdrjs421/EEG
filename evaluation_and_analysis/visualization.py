import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_polysvm_decision_boundary(
    X,
    y,
    svm_model,
    save_path=None,
    use_pca=True,
    grid_resolution=300
):
    # 1. Dimensionality reduction
    if use_pca:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
    else:
        if X.shape[1] != 2:
            raise ValueError("X must be 2D if use_pca=False")
        X_vis = X

    # 2. Prepare grid
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # NOTE: decision_function expects original feature space
    if use_pca:
        grid_original = pca.inverse_transform(grid)
    else:
        grid_original = grid

    Z = svm_model.decision_function(grid_original)
    Z = Z.reshape(xx.shape)

    # 3. Plot
    plt.figure(figsize=(8, 6))

    markers = {0: 'o', 1: 's'}
    labels = {0: 'Non-seizure', 1: 'Seizure'}

    for cls in np.unique(y):
        idx = np.where(y == cls)
        plt.scatter(
            X_vis[idx, 0],
            X_vis[idx, 1],
            marker=markers[cls],
            edgecolors='k',
            facecolors='none',
            label=labels[cls]
        )

    # Support vectors
    if hasattr(svm_model, "X"):
        sv = svm_model.X
        sv_vis = pca.transform(sv) if use_pca else sv
        plt.scatter(
            sv_vis[:, 0],
            sv_vis[:, 1],
            s=150,
            facecolors='none',
            edgecolors='k',
            linewidths=1.2,
            label='Support Vectors'
        )

    plt.contour(
        xx, yy, Z,
        levels=[-1, 0, 1],
        linestyles=['--', '-', '--'],
        colors='k'
    )

    plt.title("PolySVM Decision Boundary (2D Projection)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
