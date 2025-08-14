"""
示例：把多个已训练的 SavedModel 做简单并联/堆叠融合预测（soft-voting / weighted average）。
用法：将若干模型保存到 artifacts/saved_models/m1, m2, ... 然后运行 ensemble 脚本对验证集做融合评估。
"""
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def load_saved_models(paths):
    models = [tf.keras.models.load_model(p) for p in paths]
    return models


def predict_ensemble(models, input_dict, weights=None):
    preds = [m.predict(input_dict, batch_size=4096).reshape(-1) for m in models]
    preds = np.vstack(preds)  # (n_models, n_samples)
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    weights = np.array(weights).reshape(-1,1)
    ensemble = (weights * preds).sum(axis=0) / weights.sum()
    return ensemble

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dirs', type=str, required=True, help='comma separated saved_model dirs')
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--nrows', type=int, default=None)
    args = parser.parse_args()

    model_dirs = args.model_dirs.split(',')
    models = load_saved_models(model_dirs)

    import pandas as pd
    df = pd.read_csv(args.val_csv, nrows=args.nrows)
    with open(os.path.join(model_dirs[0], 'feature_spec.json')) as f:
        spec = json.load(f)
    # minimal preprocessing: ensure columns exist
    features = {}
    for c in spec['cat_features']:
        features[c] = df[c].fillna('').astype(str).values.reshape(-1,1)
    for n in spec['num_features']:
        features[n] = df[n].fillna(0).astype(float).values.reshape(-1,1)
    y = df['click'].astype(int).values

    preds = predict_ensemble(models, features)
    auc = roc_auc_score(y, preds)
    print('Ensemble AUC:', auc)