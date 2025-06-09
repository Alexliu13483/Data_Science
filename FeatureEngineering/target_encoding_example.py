import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from category_encoders import MEstimateEncoder

# --- 1. 創建一個模擬資料集 ---
# 包含 'Neighborhood' (地區), 'Size' (房屋大小), 'Price' (房價)
# 特意設計一些稀有地區和一個只在測試集中出現的地區

df = pd.read_csv("../datasets/ames.csv")

# 將原始 DataFrame 分割為訓練集和測試集
# 重要：我們將 X 和 y 分開分割，這是標準的 sklearn 做法
X = df.copy()
y = X.pop('SalePrice')

# random_state=42 確保結果可重現
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

print("原始訓練集 X:")
print(X_train)
print("\n原始訓練集 y:")
print(y_train)
print("\n原始測試集 X:")
print(X_test)
print("\n原始測試集 y:")
print(y_test)
print("-" * 70)

# --- 2. 使用 category_encoders 模組實作 m-estimate 平滑目標編碼 ---
def m_estimate_target_encoding_with_ce(X_train, X_test, y_train, categorical_col, m_smoothing_factor):
    """
    使用 category_encoders.MEstimateEncoder 對類別特徵進行 m-estimate 平滑的目標編碼。

    參數:
    X_train (pd.DataFrame): 用於訓練編碼的訓練特徵資料集。
    X_test (pd.DataFrame): 需要被編碼的測試特徵資料集。
    y_train (pd.Series): 用於訓練編碼的訓練目標變數。
    categorical_col (str): 需要編碼的類別特徵欄位名稱。
    m_smoothing_factor (int/float): 平滑因子，MEstimateEncoder 中的 m 參數。

    回傳:
    tuple: 兩個 Series，分別是訓練集和測試集編碼後的結果。
    """
    # 初始化 MEstimateEncoder
    # m 參數就是我們定義的平滑因子
    # handle_unknown='value' (預設): 未知類別會被編碼為訓練集的全域平均值。
    # handle_missing='value' (預設): 如果有缺失值，也會被編碼為全域平均值。
    encoder = MEstimateEncoder(cols=[categorical_col], m=m_smoothing_factor)

    # 訓練編碼器 (fit) - 僅在訓練集上進行
    # 這是防止數據洩露的關鍵步驟
    encoder.fit(X_train[[categorical_col]], y_train)

    # 轉換訓練集 (transform)
    train_encoded_df = encoder.transform(X_train[[categorical_col]])
    train_encoded_series = train_encoded_df[categorical_col] # 提取 Series

    # 轉換測試集 (transform)
    test_encoded_df = encoder.transform(X_test[[categorical_col]])
    test_encoded_series = test_encoded_df[categorical_col] # 提取 Series

    # 打印一些編碼器的內部狀態作為參考
    # 注意: _category_mapping 屬性在 MEstimateEncoder 中是內部屬性，不建議直接依賴
    # 但為了學習目的可以看看
    # print("\nMEstimateEncoder 的內部映射 (部分):")
    # if hasattr(encoder, '_category_mapping'):
    #     for mapping in encoder._category_mapping:
    #         if mapping['col'] == categorical_col:
    #             print(mapping['mapping'])
    # else:
    #     print("無法直接訪問內部映射，但編碼已應用。")

    print(f"\n使用 MEstimateEncoder (m={m_smoothing_factor}) 完成編碼。")
    return train_encoded_series, test_encoded_series

# --- 3. 應用 MEstimateEncoder ---
M_SMOOTHING_FACTOR = 1 # 與手動實作範例使用相同 m 值進行比較

train_encoded_neighborhood, test_encoded_neighborhood = \
    m_estimate_target_encoding_with_ce(X_train, X_test, y_train, 'Neighborhood', M_SMOOTHING_FACTOR)

# 將編碼後的特徵加入到訓練集和測試集
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

X_train_encoded['Neighborhood_Encoded'] = train_encoded_neighborhood
X_test_encoded['Neighborhood_Encoded'] = test_encoded_neighborhood

# 刪除原始類別欄位
X_train_encoded = X_train_encoded.drop('Neighborhood', axis=1)
X_test_encoded = X_test_encoded.drop('Neighborhood', axis=1)

print("\n--- 編碼後的訓練集 X (部分) ---")
print(X_train_encoded.head())
print("\n--- 編碼後的測試集 X (部分) ---")
print(X_test_encoded.head())

# --- 4. 使用編碼後的特徵訓練機器學習模型 ---
# 定義特徵和目標
features = ['GrLivArea', 'Neighborhood_Encoded', 'YearSold'] # 現在 'Neighborhood_Encoded' 是數值特徵
target = y_train.name # 目標 y_train, y_test 已經有了

# 初始化並訓練線性迴歸模型
model = LinearRegression()
model.fit(X_train_encoded[features], y_train) # 注意這裡使用 X_train_encoded

# 在測試集上進行預測
y_pred = model.predict(X_test_encoded[features]) # 注意這裡使用 X_test_encoded

# 評估模型性能 (使用平均絕對誤差 MAE)
mae_te = mean_absolute_error(y_test, y_pred)
print(f"\n--- 模型訓練與評估 ---")
print(f"模型的平均絕對誤差 (MAE): {mae_te:.2f}")

# --- B. 使用獨熱編碼 (One-Hot Encoding) ---
print("\n--- B. 使用獨熱編碼 ---")

# 為了確保訓練集和測試集有相同的獨熱編碼欄位，
# 將它們的 'Neighborhood' 欄位先合併，然後進行獨熱編碼，再重新分割。
# 這可以處理訓練集或測試集中獨有類別的情況。
combined_neighborhood = pd.concat([X_train['Neighborhood'], X_test['Neighborhood']], axis=0)
# 使用 pd.get_dummies 生成獨熱編碼
ohe_dummies = pd.get_dummies(combined_neighborhood, prefix='Neighborhood_OHE')

# 根據原始的訓練集和測試集索引來重新分割獨熱編碼後的結果
X_train_ohe_df = ohe_dummies.loc[X_train.index]
X_test_ohe_df = ohe_dummies.loc[X_test.index]

# 將獨熱編碼後的特徵合併回 X_train 和 X_test
X_train_ohe = X_train.copy()
X_test_ohe = X_test.copy()

X_train_ohe = pd.concat([X_train_ohe.drop('Neighborhood', axis=1), X_train_ohe_df], axis=1)
X_test_ohe = pd.concat([X_test_ohe.drop('Neighborhood', axis=1), X_test_ohe_df], axis=1)

print("\n獨熱編碼後的訓練集 X (部分):")
print(X_train_ohe.head())
print("\n獨熱編碼後的測試集 X (部分):")
print(X_test_ohe.head())

# 訓練並評估模型 (獨熱編碼)
# 定義獨熱編碼後的特徵欄位
ohe_features = ['GrLivArea', 'YearSold'] + [col for col in X_train_ohe.columns if col.startswith('Neighborhood_OHE_')]

model_ohe = LinearRegression()
model_ohe.fit(X_train_ohe[ohe_features], y_train)
y_pred_ohe = model_ohe.predict(X_test_ohe[ohe_features])
mae_ohe = mean_absolute_error(y_test, y_pred_ohe)
print(f"\n使用獨熱編碼的模型平均絕對誤差 (MAE): {mae_ohe:.2f}")
print("-" * 70)

# --- C. 比較結果 ---
print("\n--- C. 比較模型性能 ---")
print(f"目標編碼模型 MAE: {mae_te:.2f}")
print(f"獨熱編碼模型 MAE: {mae_ohe:.2f}")

if mae_te < mae_ohe:
    print("\n結論: 在此範例中，使用 M-Estimate 平滑的目標編碼的模型效能優於獨熱編碼。")
    print("這可能是因為目標編碼更好地捕捉了類別與目標變數的關係，同時處理了稀有和未知類別的問題。")
elif mae_ohe < mae_te:
    print("\n結論: 在此範例中，獨熱編碼的模型效能優於 M-Estimate 平滑的目標編碼。")
    print("這可能發生在類別數量不多，或類別與目標關係不那麼強烈的情況。")
else:
    print("\n結論: 在此範例中，兩種編碼方法的模型效能相似。")

print("\n--- 補充說明 ---")
print("獨熱編碼創建的新欄位數量:", len(ohe_features) - 1)
print("目標編碼創建的新欄位數量:", 1)
neighborhood_nums = df['Neighborhood'].nunique()
neighborhood_train_nums = X_train['Neighborhood'].nunique()
neighborhood_test_nums = X_test['Neighborhood'].nunique()
print(f"原始資料中 'Neighborhood' 的獨特類別數量: {neighborhood_nums}")
print(f"訓練集中 'Neighborhood' 的獨特類別數量: {neighborhood_train_nums}")
print(f"測試集中 'Neighborhood' 的獨特類別數量: {neighborhood_test_nums}")

# 檢查測試集中的未知類別在兩種編碼後的表現
if neighborhood_train_nums < neighborhood_nums and neighborhood_test_nums < neighborhood_train_nums:
    ## 在測試資料集中找出訓練資料集裡沒有的類別。
    ### --- 2. 獲取訓練集和測試集的獨特類別 ---
    train_categories = set(X_train['Neighborhood'].unique())
    test_categories = set(X_test['Neighborhood'].unique())

    print("訓練集 'Neighborhood' 獨特類別數量:", len(train_categories))
    # print("訓練集 'Neighborhood' 獨特類別:", sorted(list(train_categories)))
    print("\n測試集 'Neighborhood' 獨特類別數量:", len(test_categories))
    # print("測試集 'Neighborhood' 獨特類別:", sorted(list(test_categories)))
    print("-" * 50)

    # --- 3. 找出測試集中存在但訓練集中不存在的類別 ---
    # 使用集合的差集操作：test_categories - train_categories
    unknown_categories_in_test = test_categories - train_categories

    print(f"\n測試集中存在但訓練集中不存在的 'Neighborhood' 類別 (未知類別):")
    if unknown_categories_in_test:
        print(sorted(list(unknown_categories_in_test)))
    else:
        print("沒有未知類別。")

    print("-" * 50)

    for unknown_category in unknown_categories_in_test:
        print(f"測試集中未知類別 '{unknown_category}' 在兩種編碼後的表現:")

        # 目標編碼
        if unknown_category in X_test_encoded['Neighborhood_Encoded'].unique():
            unknown_te = X_test_encoded[X_test_encoded['Neighborhood_Encoded'] == unknown_category]
            print(f"目標編碼後 '{unknown_category}' 的值 (應為全域平均): {unknown_te['Neighborhood_Encoded'].mean():.2f}")
        else:
            print(f"目標編碼後 '{unknown_category}' 不存在於編碼結果中。")

        # 獨熱編碼
        if f'Neighborhood_OHE_{unknown_category}' in X_test_ohe.columns:
            unknown_ohe = X_test_ohe[f'Neighborhood_OHE_{unknown_category}']
            print(f"獨熱編碼後 '{unknown_category}' 的欄位 (應為 [0,0,0,...,1,...] 形式):")
            print(unknown_ohe.iloc[0])
        else:
            print(f"獨熱編碼後 '{unknown_category}' 不存在於編碼結果中。")
