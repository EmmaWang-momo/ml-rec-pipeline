import os
import json
import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 0. 读取配置数据
config_file = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Ali_Display_Ad_Click/feature_spec.json'
with open(config_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
num_features = data['numeric_cols']
cat_features = data['categorical_cols']

# 1. 加载数据
# data = load_breast_cancer()
data_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Ali_Display_Ad_Click/'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'), nrows=None)
val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))

# 2. 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
label_name = data['label']
X_train, y_train = train_df[cat_features + num_features], train_df[label_name].astype(int).values
X_test, y_test = val_df[cat_features + num_features], val_df[label_name].astype(int).values

# 3. 转换为 XGBoost 专用数据结构
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 4. 设置参数（这里是二分类）
params = {
    "objective": "binary:logistic",  # 二分类逻辑回归
    "eval_metric": "auc",            # 评价指标
    "eta": 0.1,                       # 学习率
    "max_depth": 4,                   # 树深度
    "subsample": 0.8,                 # 采样比例
    "colsample_bytree": 0.8,          # 特征采样比例
    "seed": 42
}

# 5. 训练模型
evals = [(dtrain, "train"), (dtest, "test")]
model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=20)

# 6. 模型预测
y_pred_prob = model.predict(dtest)             # 预测概率
y_pred = (y_pred_prob > 0.5).astype(int)       # 转换成 0/1

# 7. 评估结果
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")

# 8. 保存模型
model_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/artifacts/saved_models/xgb/'
model.save_model(os.path.join(model_dir, "xgb_binary_model_ali_ad.json"))
print("模型已保存到 xgb_binary_model.json")
