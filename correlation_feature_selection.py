"""Utilities for removing highly correlated features and evaluating model impact."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification


def reduce_correlated_features(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = 0.9,
    random_state: int | None = 0,
):
    """Remove highly correlated features and evaluate their impact.

    Parameters
    ----------
    df:
        DataFrame containing features and the target column.
    target_column:
        Name of the column representing the target variable.
    threshold:
        Absolute correlation above which features are considered redundant.
    random_state:
        Random seed for the ``RandomForestClassifier`` used in evaluation.

    Returns
    -------
    dict
        A dictionary containing:
        ``X_reduced`` - feature matrix after dropping correlated features.
        ``feature_importance`` - importance of remaining features.
        ``base_score`` - cross-validated score before dropping features.
        ``reduced_score`` - cross-validated score after dropping features.
        ``dropped_features`` - list of removed feature names.
        ``clusters`` - clusters of correlated features.
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    clusters: list[list[str]] = []

    for feature in corr.columns:
        if feature not in to_drop:
            cluster = corr[feature][corr[feature] > threshold].index.tolist()
            clusters.append(cluster)
            to_drop.update(cluster[1:])

    X_reduced = X.drop(columns=list(to_drop))

    rf = RandomForestClassifier(random_state=random_state)
    base_score = cross_val_score(rf, X, y, cv=5).mean()
    reduced_score = cross_val_score(rf, X_reduced, y, cv=5).mean()

    rf.fit(X_reduced, y)
    feature_importance = pd.Series(
        rf.feature_importances_, index=X_reduced.columns
    ).sort_values(ascending=False)

    return {
        "X_reduced": X_reduced,
        "feature_importance": feature_importance,
        "base_score": base_score,
        "reduced_score": reduced_score,
        "dropped_features": list(to_drop),
        "clusters": clusters,
    }


if __name__ == "__main__":
    # Demonstration with synthetic data
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=0
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    result = reduce_correlated_features(df, "target")

    print("Dropped features:", result["dropped_features"])
    print("Base score:", round(result["base_score"], 3))
    print("Reduced score:", round(result["reduced_score"], 3))
    print("Feature importance:\n", result["feature_importance"])
