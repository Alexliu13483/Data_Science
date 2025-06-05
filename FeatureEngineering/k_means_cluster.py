# Example of California housing dataset clustering using KMeans

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

df = pd.read_csv("../datasets/housing.csv")

X = df.loc[:, ["median_income", "latitude", "longitude"]]
X.head()

# Create cluster feature
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()

sns.relplot(
    x="longitude", y="latitude", hue="Cluster", data=X, height=6,
)

X["MedHouseVal"] = df["median_house_value"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6)


def score_dataset(X, y, model=XGBRegressor()):
    """
    Calculates the Root Mean Squared Log Error (RMSLE) for a given dataset and model using cross-validation.

    This function label-encodes categorical features in the input DataFrame `X`, then evaluates the provided regression model
    (default: XGBRegressor) using 5-fold cross-validation. The scoring metric used is negative mean squared log error,
    which is then converted to RMSLE.

    Args:
        X (pd.DataFrame): Feature matrix containing predictors. Categorical columns will be label-encoded.
        y (pd.Series or np.ndarray): Target variable.
        model (sklearn.base.RegressorMixin, optional): Regression model to evaluate. Defaults to XGBRegressor().

    Returns:
        float: The cross-validated RMSLE score for the model on the provided data.

    # Brief: Evaluates a regression model using cross-validated RMSLE after encoding categorical features.
    """
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# Prepare data
df = pd.read_csv("../datasets/ames.csv")

X = df.copy()
y = X.pop("SalePrice")


# Define a list of the features to be used for the clustering
features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "GrLivArea"]


# Standardize
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)


# Fit the KMeans model to X_scaled and create the cluster labels
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)

Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["SalePrice"] = y
sns.relplot(
    x="value", y="SalePrice", hue="Cluster", col="variable",
    height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
    data=Xy.melt(
        value_vars=features, id_vars=["SalePrice", "Cluster"],
    ),
)

score_dataset(X, y)

kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)


# Create the cluster-distance features using `fit_transform`
X_cd = kmeans.fit_transform(X_scaled)


# Label features and join to dataset
X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
X = X.join(X_cd)

score_dataset(X, y)