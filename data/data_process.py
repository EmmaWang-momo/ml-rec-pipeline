"""
Data Process
"""
import os
import json
import ast
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.feature_extraction import FeatureHasher


# ------------------------
# Utility / EDA functions
# ------------------------
def basic_eda(df: pd.DataFrame, target: str = "click", n_display=10):
    print("=== BASIC EDA ===")
    print("Shape:", df.shape)
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nMissing values (top 20):")
    print(df.isnull().sum().sort_values(ascending=False).head(n_display))
    if target in df.columns:
        print("\nTarget distribution:")
        print(df[target].value_counts(dropna=False))
        print("\nTarget rate:", df[target].mean())
    print("\nNumeric summary (top 10 cols):")
    print(df.select_dtypes(include=np.number).describe().T.head(10))
    print("=================\n")


# ------------------------
# Bayesian smoothing for CTR-like rates
# ------------------------
def estimate_beta_params_from_counts(imps: np.ndarray, clicks: np.ndarray) -> Tuple[float, float]:
    """
    Method-of-moments estimator for Beta prior alpha, beta.
    imps: array of impression counts per key
    clicks: array of click counts per key
    We compute observed rates r_i = clicks_i / imps_i (for imps>0), and use weighted moments.
    Reference idea: fit Beta such that mean and var match empirical mean/var of rates.
    """
    # filter valid
    mask = imps > 0
    if mask.sum() == 0:
        return 1.0, 1.0
    rates = clicks[mask] / imps[mask]
    mean_r = rates.mean()
    var_r = rates.var(ddof=0)
    # avoid degenerate
    if var_r <= 0:
        # fall back to weak prior
        return 1.0, 1.0
    # method of moments for Beta: mean = a/(a+b), var = ab/((a+b)^2 (a+b+1))
    common = mean_r * (1 - mean_r) / var_r - 1
    a = max(mean_r * common, 1e-3)
    b = max((1 - mean_r) * common, 1e-3)
    return float(a), float(b)


def bayesian_smoothed_rate(clicks: np.ndarray, imps: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Bayesian smoothed CTR: (clicks + alpha) / (imps + alpha + beta)"""
    return (clicks + alpha) / (imps + alpha + beta)


# ------------------------
# Feature engineering helpers
# ------------------------
def count_encoding(series: pd.Series) -> pd.Series:
    counts = series.map(series.value_counts())
    return counts.astype(np.int32)


def add_hash_feature(df: pd.DataFrame, col: str, n_features: int = 32, new_name: str = None):
    """Feature hashing for a single categorical column -> multiple hashed numeric columns"""
    if new_name is None:
        new_name = f"{col}_hash"
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    payload = series_to_str_list(df[col])
    X = hasher.transform(payload)  # sparse matrix
    X = X.toarray()
    for i in range(X.shape[1]):
        df[f"{new_name}_{i}"] = X[:, i]
    return df


def series_to_str_list(s: pd.Series):
    # FeatureHasher expects iterable of strings per row (list of feature names)
    return s.fillna("").astype(str).apply(lambda x: [x])


def one_hot_encode_low_cardinality(df: pd.DataFrame, col: str, max_unique=10, prefix=None):
    uniques = df[col].nunique(dropna=False)
    if uniques > max_unique:
        raise ValueError(f"Column {col} has high cardinality ({uniques}), skip one-hot")
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    arr = enc.fit_transform(df[[col]].fillna("NA"))
    names = [f"{prefix or col}_oh_{v}" for v in enc.categories_[0]]
    for i, name in enumerate(names):
        df[name] = arr[:, i].astype(np.int8)
    return df, names


def numeric_transformations(df: pd.DataFrame, numeric_cols: List[str], log_transform: bool = True):
    """Apply log1p, then return scaled versions (std and minmax) and record scalers."""
    scalers = {}
    # copy arrays
    for col in numeric_cols:
        col_in = df[col].fillna(0).astype(float)
        if log_transform:
            df[f"{col}_log1p"] = np.log1p(col_in)
            source = df[f"{col}_log1p"]
        else:
            df[f"{col}_raw"] = col_in
            source = df[f"{col}_raw"]
    # standardize
    std_scaler = StandardScaler()
    raw_cols = [c for c in df.columns if c.endswith("_log1p") or c.endswith("_raw")]
    if raw_cols:
        df_std = std_scaler.fit_transform(df[raw_cols].fillna(0).values)
        for i, c in enumerate(raw_cols):
            df[f"{c}_std"] = df_std[:, i]
        scalers['std'] = std_scaler
    # minmax
    mm_scaler = MinMaxScaler()
    if raw_cols:
        df_mm = mm_scaler.fit_transform(df[raw_cols].fillna(0).values)
        for i, c in enumerate(raw_cols):
            df[f"{c}_mm"] = df_mm[:, i]
        scalers['minmax'] = mm_scaler
    return df, scalers


# ------------------------
# K-fold target encoding (CTR encoding) with smoothing to reduce leakage
# ------------------------
def kfold_target_encoding(df: pd.DataFrame, col: str, target: str, n_splits: int = 5, seed: int = 42,
                          prior_alpha: float = 1.0, prior_beta: float = 1.0) -> pd.Series:
    """
    Returns an array of encoded values for the column `col`.
    We compute smoothed CTR per fold using train portion and apply to validation fold.
    Smoothing uses Bayesian prior (alpha, beta).
    """
    n = len(df)
    res = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, valid_idx in kf.split(np.arange(n)):
        train_df = df.iloc[train_idx]
        # aggregate clicks / imps per key
        agg = train_df.groupby(col)[target].agg(['sum', 'count']).rename(columns={'sum': 'clicks', 'count': 'imps'})
        # compute alpha/beta maybe via method of moments on train agg
        alpha, beta = estimate_beta_params_from_counts(agg['imps'].values, agg['clicks'].values)
        # but allow overriding with provided priors
        alpha = prior_alpha or alpha
        beta = prior_beta or beta
        # smoothed rate per key
        agg['smoothed'] = bayesian_smoothed_rate(agg['clicks'].values, agg['imps'].values, alpha, beta)
        # map to validation
        res.iloc[valid_idx] = df.iloc[valid_idx][col].map(agg['smoothed']).fillna((prior_alpha) / (prior_alpha + prior_beta))
    return res.fillna((prior_alpha) / (prior_alpha + prior_beta))


# ------------------------
# Simple pairwise co-occurrence counts (coec)
# ------------------------
def add_pair_count_feature(df: pd.DataFrame, col_a: str, col_b: str, new_col_name: str = None):
    """Add a feature equal to the count of occurrences of pair (col_a, col_b)."""
    if new_col_name is None:
        new_col_name = f"cnt_{col_a}_{col_b}"
    pair_counts = df.groupby([col_a, col_b]).size().rename("pair_cnt")
    df = df.merge(pair_counts.reset_index(), on=[col_a, col_b], how="left")
    df[new_col_name] = df['pair_cnt'].fillna(0).astype(int)
    df.drop(columns=['pair_cnt'], inplace=True)
    return df


# ------------------------
# Splitting
# ------------------------
def train_val_test_split(df: pd.DataFrame, target: str, test_size: float = 0.1, val_size: float = 0.1,
                         timestamp_col: str = None, id_col: str = None, random_state: int = 42):
    """
    If timestamp_col provided, perform time-based split: oldest -> train, middle -> val, newest -> test.
    Else perform stratified random split by target.
    Returns train_df, val_df, test_df
    """
    if timestamp_col and timestamp_col in df.columns:
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df_sorted)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        train_df = df_sorted.iloc[: n - n_val - n_test].reset_index(drop=True)
        val_df = df_sorted.iloc[n - n_val - n_test: n - n_test].reset_index(drop=True)
        test_df = df_sorted.iloc[n - n_test:].reset_index(drop=True)
        return train_df, val_df, test_df
    else:
        # stratified
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=random_state)
        for train_idx, rem_idx in splitter.split(df, df[target]):
            train_df = df.iloc[train_idx]
            rem = df.iloc[rem_idx]
        # split rem into val/test
        val_frac = val_size / (test_size + val_size)
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
        for val_idx, test_idx in splitter2.split(rem, rem[target]):
            val_df = rem.iloc[val_idx]
            test_df = rem.iloc[test_idx]
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ------------------------
# get different types of numeric features
# ------------------------
def get_diff_num_feats(df, numeric_cols, high_val_percent=0.95, low_val_percent=0.05, max_unique_count=20):
    general_num_feats, conversion_rate_num_feats, bin_num_feats, cate_num_feats = [], [], [], []
    for col in numeric_cols:
        # print('col name: ', col)
        max_val, min_val, high_val, low_val = df[col].max(), df[col].min(), df[col].quantile(high_val_percent), df[col].quantile(low_val_percent)
        # print('col max min: ', max_val, min_val, high_val, low_val)
        unique_count = df[col].nunique()
        # print('col unique count: ', unique_count)

        if unique_count <= max_unique_count:
            cate_num_feats.append(col)
            continue

        if high_val <= 1 and low_val >= 0:
            conversion_rate_num_feats.append(col)
            continue

        if col.lower() == 'age':
            bin_num_feats.append(col)
            continue

        general_num_feats.append(col)

    return general_num_feats, conversion_rate_num_feats, bin_num_feats, cate_num_feats


# ------------------------
# Main processing pipeline
# ------------------------
def process_pipeline(input_csv, conf_dir, out_dir, target_col, id_col=None,
                     numeric_cols=None, categorical_cols=None, seq_cols=None,
                     timestamp_col=None):
    # 读取数据，新建输出目录
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    # print("Loaded", input_csv, "shape:", df.shape)

    unkown = 'UNKNOWN'
    unkown_id = 0
    numeric_feat_lst, categorical_feat_lst, categorical_feat_config, seq_feat_lst, seq_feat_config = [], [], {}, [], {}

    # 读取数值特征工程模板
    with open(os.path.join(conf_dir, 'numeric_feat_template.json'), "r", encoding="utf-8") as f:
        df_numeric_feat_config = json.load(f)  # 解析成 Python 对象
    # print(df_numeric_feat_config)

    for item in df_numeric_feat_config:
        col, num_feat_types = item['col_name'], item['numeric_col_type']
        for num_feat_type in num_feat_types.split('&'):
            if num_feat_type == 'general_num_feat':
                fill_na = float(item['conf_fill_na_val'])
                df[col] = df[col].fillna(fill_na)

                max_val, min_val = float(item['conf_max_val']), float(item['conf_min_val'])
                feat_name = col + "_mm"
                df[feat_name] = (df[col] - min_val) / (max_val - min_val)
                numeric_feat_lst.append(feat_name)

            if num_feat_type == 'conversion_rate_num_feat':
                # fill_na = float(item['conf_fill_na_val'])
                fill_na = 0.0
                df[col] = df[col].fillna(fill_na)

                lower_limit, upper_limit = 0.0, 1.0
                df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

                feat_name = col + "_conv"
                df[feat_name] = df[col]
                numeric_feat_lst.append(feat_name)

            if num_feat_type == 'cate_num_feat':
                feat_name = col + "_obj"
                df[feat_name] = df[col]
                df[feat_name] = df[feat_name].astype(str)

                # fill_na = float(item['conf_fill_na_val'])
                fill_na = unkown
                df[feat_name] = df[feat_name].fillna(fill_na)

                conf_candidate_len = int(item['conf_candidate_len'])
                conf_candidate_vals = item['conf_candidate_vals'].split(',')
                conf_candidate_mapping = {}
                for i, v in enumerate(conf_candidate_vals):
                    conf_candidate_mapping[v] = i
                # print(conf_candidate_mapping)
                df[feat_name] = df[feat_name].map(conf_candidate_mapping).fillna(unkown_id).astype(int)

                categorical_feat_lst.append(feat_name)
                categorical_feat_config[feat_name] = conf_candidate_len

            # if num_feat_type == 'bin_num_feat':
            #     x = 1

    # 读取类别特征工程模板
    with open(os.path.join(conf_dir, 'category_feat_template.json'), "r", encoding="utf-8") as f:
        df_cate_feat_config = json.load(f)  # 解析成 Python 对象
    print(df_cate_feat_config)

    if timestamp_col:
        df[timestamp_col + "_datetime"] = pd.to_datetime(df[timestamp_col])
        df[timestamp_col + '_hour'] = df[timestamp_col + "_datetime"].dt.hour
        df[timestamp_col + '_week'] = df[timestamp_col + "_datetime"].dt.weekday
    print(df)

    for item in df_cate_feat_config:
        col = item['col_name']
        feat_name = col + "_obj"
        df[feat_name] = df[col]
        df[feat_name] = df[feat_name].astype(str)

        # fill_na = float(item['conf_fill_na_val'])
        fill_na = unkown
        df[feat_name] = df[feat_name].fillna(fill_na)

        conf_candidate_len = int(item['conf_candidate_len'])
        conf_candidate_vals = item['conf_candidate_vals'].split(',')
        conf_candidate_mapping = {}
        for i, v in enumerate(conf_candidate_vals):
            conf_candidate_mapping[v] = i
        # print(conf_candidate_mapping)
        df[feat_name] = df[feat_name].map(conf_candidate_mapping).fillna(unkown_id).astype(int)

        categorical_feat_lst.append(feat_name)
        categorical_feat_config[feat_name] = conf_candidate_len

    # 读取序列特征工程模板
    if seq_cols:
        with open(os.path.join(conf_dir, 'seq_feat_template.json'), "r", encoding="utf-8") as f:
            df_seq_feat_config = json.load(f)  # 解析成 Python 对象
        print(df_seq_feat_config)

        for item in df_seq_feat_config:
            col = item['col_name']
            feat_name = col + "_seq"
            df[feat_name] = df[col]

            # 格式转化
            df[feat_name] = df[feat_name].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).strip() != "" else []
            )

            conf_candidate_len = int(item['conf_candidate_len'])
            conf_candidate_vals = item['conf_candidate_vals'].split(',')
            conf_sequence_len = int(item['conf_sequence_len'])
            conf_candidate_mapping = {}
            for i, v in enumerate(conf_candidate_vals):
                conf_candidate_mapping[v] = i

            def deduplicate(seq):
                seen = set()
                return [x for x in seq if not (x in seen or seen.add(x))]

            # 3️⃣ 编码 + 限长 + padding
            def process_sequence(seq):
                max_len = conf_sequence_len
                PAD_ID = unkown_id
                value2id = conf_candidate_mapping

                if not seq:
                    return [PAD_ID] * max_len
                # 去重
                seq = deduplicate(seq)
                # 取最近 max_len 个
                seq = seq[:max_len]
                # 编码
                seq_encoded = []
                # for v in seq:
                #     if v in value2id:
                #         seq_encoded.append(value2id[v])
                #     else:
                #         seq_encoded.append(PAD_ID)
                seq_encoded = [value2id[v] for v in seq if v in value2id]
                # padding（后补 0 保持最近行为在前面）
                if len(seq_encoded) < max_len:
                    seq_encoded = seq_encoded + [PAD_ID] * (max_len - len(seq_encoded))
                return seq_encoded

            df[feat_name] = df[feat_name].apply(process_sequence)

            seq_feat_lst.append(feat_name)
            seq_feat_config[feat_name] = [conf_sequence_len, conf_candidate_len]

    feature_cols = numeric_feat_lst + categorical_feat_lst + seq_feat_lst
    df = df[id_col + [target_col, timestamp_col] + feature_cols]

    # train/val/test split
    train_df, val_df, test_df = train_val_test_split(df, target=target_col, test_size=0.1, val_size=0.1, timestamp_col=timestamp_col, id_col=id_col)
    print("Split sizes:", train_df.shape, val_df.shape, test_df.shape)

    # Save processed DataFrames with only features + target + id/time
    cols_to_save = id_col + feature_cols + [target_col]
    # if id_col:
    #     cols_to_save = id_col + cols_to_save
    # if timestamp_col:
    #     cols_to_save = cols_to_save + [timestamp_col]

    # train_df[cols_to_save].to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    # val_df[cols_to_save].to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    # test_df[cols_to_save].to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

    train_df[cols_to_save].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df[cols_to_save].to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df[cols_to_save].to_csv(os.path.join(out_dir, "test.csv"), index=False)

    # Persist feature spec
    feat_spec = {
        'feature_columns': feature_cols,
        'numeric_cols': numeric_feat_lst,
        'categorical_cols': categorical_feat_lst,
        'categorical_cols_len': categorical_feat_config,
        'sequence_cols': seq_feat_lst,
        'sequence_cols_len': seq_feat_config,
        'label': target_col
    }
    with open(os.path.join(out_dir, "feature_spec.json"), "w") as f:
        json.dump(feat_spec, f, indent=2)

    print("Saved processed artifacts to", out_dir)
    return train_df, val_df, test_df, feat_spec


# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True, help="Input CSV path")
    # parser.add_argument('--out_dir', default='artifacts/processed', help="Output directory")
    # parser.add_argument('--target', default='click', help="Target column name")
    # parser.add_argument('--id_col', default='impression_id', help="ID column")
    # parser.add_argument('--timestamp_col', default=None, help="Timestamp column (for time split), e.g. hour")
    # parser.add_argument('--num_cols', default=None, help="Comma separated numeric columns (optional)")
    # parser.add_argument('--cat_cols', default=None, help="Comma separated categorical columns (optional)")
    args = parser.parse_args()

    """
    # Ali_Display_Ad_Click
    args.input = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/50krecords.csv'
    args.conf_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Avazu_CTR_Prediction_50K/feature/'
    args.out_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/'
    args.target = 'click'
    args.id_col = ['id']
    args.timestamp_col = 'hour'
    """
    # Ali_Display_Ad_Click
    args.input = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Ali_Display_Ad_Click/data.csv'
    args.conf_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Ali_Display_Ad_Click/feature/'
    args.out_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Ali_Display_Ad_Click/'
    args.target = 'clk'
    args.id_col = ['user_id', 'adgroup_id']
    args.timestamp_col = 'time_stamp'
    args.num_cols = ['price', 'age', 'ctr']
    args.cat_cols = [
        'cms_segid'
        , 'cms_group_id'
        , 'final_gender_code'
        , 'age_level'
        , 'pvalue_level'
        , 'shopping_level'
        , 'occupation'
        , 'new_user_class_level '
        , 'cate_id'
        , 'campaign_id'
        , 'customer'
        , 'brand'
        , 'pid'
        , 'is_good'
    ]
    args.seq_cols = [
        'user_pv_category_seq'
        , 'user_pv_brand_seq'
        , 'user_buy_category_seq'
        , 'user_buy_brand_seq'
        , 'user_fav_category_seq'
        , 'user_fav_brand_seq'
        , 'user_cart_category_seq'
        , 'user_cart_brand_seq'
    ]

    process_pipeline(args.input,
                     args.conf_dir,
                     args.out_dir,
                     args.target,
                     id_col=args.id_col,
                     numeric_cols=args.num_cols,
                     categorical_cols=args.cat_cols,
                     seq_cols=args.seq_cols,
                     timestamp_col=args.timestamp_col)
