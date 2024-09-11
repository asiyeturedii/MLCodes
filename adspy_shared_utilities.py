import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_boundaries(X, y, model, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])):
    # Fit the model
    model.fit(X, y)

    # Create a mesh grid
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict classifications for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap=cmap)
    plt.title('Decision Boundaries for K-Nearest Neighbors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
