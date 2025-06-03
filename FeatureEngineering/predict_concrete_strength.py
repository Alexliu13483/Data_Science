"""
The Concrete dataset contains a variety of concrete formulations
and the resulting product's compressive strength, which is a measure
of how much load that kind of concrete can bear. The task for this 
dataset is to predict a concrete's compressive strength given
its formulation.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("concrete.csv")
df.head()

"""
Create a baseline to evaluate the performance of the model.
"""
X = df.copy()
y = X.pop("CompressiveStrength") # pop() 方法會同時移除欄位並回傳其值。

# Train and score baseline model: 8.232
baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")

"""
Ratios of the features above would be a good predictor of CompressiveStrength.
"""
# Create synthetic features
X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

# Train and score model on dataset with additional ratio features: 7.948
model = RandomForestRegressor(criterion="absolute_error", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Features: {score:.4}")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# 保留的特徵只有這三個比例特徵做訓練：結果是 12.4073，變更差，
# 應該是除了比例特徵之外的其他特徵對於預測混凝土強度有很大的影響。
X_all = df.copy()
y = X_all.pop("CompressiveStrength")

# 建立三個比例特徵
X_ratio_only = pd.DataFrame()
X_ratio_only["FCRatio"] = X_all["FineAggregate"] / X_all["CoarseAggregate"]
X_ratio_only["AggCmtRatio"] = (X_all["CoarseAggregate"] + X_all["FineAggregate"]) / X_all["Cement"]
X_ratio_only["WtrCmtRatio"] = X_all["Water"] / X_all["Cement"]

# 建立模型並交叉驗證: 12.4073
model = RandomForestRegressor(criterion="absolute_error", random_state=0)
mae_score = -cross_val_score(model, X_ratio_only, y, cv=5, scoring="neg_mean_absolute_error").mean()

print(f"MAE using only ratio features: {mae_score:.4f}")


