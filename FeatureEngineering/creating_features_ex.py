import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def score_dataset(X, y, model=XGBRegressor()):
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

'''
Create the following features:

- `LivLotRatio`: the ratio of `GrLivArea` to `LotArea`
- `Spaciousness`: the sum of `FirstFlrSF` and `SecondFlrSF` divided by `TotRmsAbvGrd`
- `TotalOutsideSF`: the sum of `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `Threeseasonporch`, and `ScreenPorch`
'''
X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = X["GrLivArea"] / X["LotArea"]
X_1["Spaciousness"] = (X["FirstFlrSF"] + X["SecondFlrSF"]) / X["TotRmsAbvGrd"]
X_1["TotalOutsideSF"] = X[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "Threeseasonporch", "ScreenPorch"]].sum(axis=1)

# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 =  pd.get_dummies(X['BldgType'], prefix="Bldg")
# Multiply
X_2 = X_2.mul(X["GrLivArea"], axis=0)

"""
Create a feature `PorchTypes` that counts how many of the following are greater than 0.0:

WoodDeckSF
OpenPorchSF
EnclosedPorch
Threeseasonporch
ScreenPorch
"""
X_3 = pd.DataFrame()

# YOUR CODE HERE
porches = [
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch"
]
X_3["PorchTypes"] = X[porches].gt(0).sum(axis=1)

df.MSSubClass.unique()

"""
You can see that there is a more general categorization described (roughly) by the first word of each category. Create a feature containing only these first words by splitting MSSubClass at the first underscore _. (Hint: In the split method use an argument n=1.)
"""
X_4 = pd.DataFrame()

# YOUR CODE HERE
X_4["MSClass"] = X["MSSubClass"].str.split('_', n=1).str[0]
print(X_4["MSClass"].unique())

"""
Create a feature MedNhbdArea that describes the median of GrLivArea grouped on Neighborhood.
"""
X_5 = pd.DataFrame()

# YOUR CODE HERE
X_5["MedNhbdArea"] = X.groupby("Neighborhood")["GrLivArea"].transform("median")
print(X_5["MedNhbdArea"].head())