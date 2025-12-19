from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Check if data for last patient is available
if last_X_train_scaled is None or last_y_train is None or last_X_test_scaled is None or last_y_test is None or last_svm_model is None:
    print("❌ No data saved from the last patient. Please run the training loop first.")
else:
    # 🎯 1. PCA로 2D 차원 축소
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(last_X_train_scaled)
    X_test_pca = pca.transform(last_X_test_scaled)

    # 🎯 2. PolySVM 학습 (using the already trained model for the last patient)
    # The last_svm_model is already trained, no need to retrain unless parameters changed
    svm = last_svm_model

    # 🎯 3. Meshgrid 생성 (시각화용)
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # 🎯 4. 결정 함수 값 계산
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_2d) # Transform back to original feature space for SVM
    Z = svm.decision_function(grid_original).reshape(xx.shape)

    # 🎯 5. 시각화
    plt.figure(figsize=(10, 6))

    # ⚫ 결정 경계선 (Z == 0)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)

    # 🔵 실제 데이터 점 (Test data for visualization)
    plt.scatter(X_test_pca[last_y_test == 0, 0], X_test_pca[last_y_test == 0, 1],
                c='white', label='Non-Seizure', edgecolor='k', s=20)
    plt.scatter(X_test_pca[last_y_test == 1, 0], X_test_pca[last_y_test == 1, 1],
                c='blue', label='Seizure', edgecolor='k', s=30)

    plt.title("📊 PolySVM Decision Boundary with Contour (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
