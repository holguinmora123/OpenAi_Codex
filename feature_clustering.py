import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def cluster_features(X, feature_names, threshold=0.5):
    """Cluster features using hierarchical clustering on absolute correlation."""
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()
    # Distance matrix defined as 1 - |corr|
    distance = 1 - np.abs(corr)
    # linkage requires condensed distance matrix
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    selected_features = []
    for cluster_id in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cluster_id)[0][0]
        selected_features.append(feature_names[idx])
    return selected_features, corr, distance


def evaluate(X, y):
    """Evaluate a logistic regression model with cross-validation."""
    model = LogisticRegression(max_iter=1000)
    return cross_val_score(model, X, y, cv=5).mean()


def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Baseline performance with all features
    baseline_score = evaluate(X_scaled, y)

    # Cluster features and select representatives
    selected_features, corr, distance = cluster_features(X_scaled, feature_names)
    X_selected = pd.DataFrame(X_scaled, columns=feature_names)[selected_features].values
    selected_score = evaluate(X_selected, y)

    print("Correlation matrix sample:\n", corr.iloc[:5, :5])
    print("Distance matrix sample:\n", distance.iloc[:5, :5])
    print("Baseline features:", X_scaled.shape[1], "CV accuracy:", baseline_score)
    print("Selected features:", len(selected_features), "CV accuracy:", selected_score)
    print("Selected feature names:", selected_features)


if __name__ == "__main__":
    main()
