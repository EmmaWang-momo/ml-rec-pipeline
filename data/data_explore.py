"""
Data Explore
"""
import os
import json
import argparse
import sys
import ast
from collections import defaultdict
from typing import List, Tuple, Dict
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.feature_extraction import FeatureHasher


# ------------------------
# basic info
# ------------------------
def basic_eda(df: pd.DataFrame, target: str = "click", n_display=10):
    print("====== BASIC EDA ======")
    print("Shape:", df.shape)
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nMissing values (top 20):")
    print(df.isnull().sum().sort_values(ascending=False).head(n_display))
    if target in df.columns:
        print("\nTarget distribution:")
        print(df[target].value_counts(dropna=False))
    print("\nNumeric summary (top 10 cols):")
    print(df.select_dtypes(include=np.number).describe().T.head(10))
    print("=================================================\n")


# ------------------------
# explore different types of numeric features
# ------------------------
def explore_numeric_feats(df, numeric_cols, save_dir, high_val_percent=0.95, low_val_percent=0.05, max_unique_count=20, bin_col_names=['age']):
    output_lst = []
    unkown = 'UNKNOWN'
    for col in numeric_cols:
        # max min high_percent low_percent
        max_val, min_val, high_val, low_val = df[col].max(), df[col].min(), df[col].quantile(high_val_percent), df[col].quantile(low_val_percent)

        # unique count
        unique_count = df[col].nunique()
        unique_val_lst = df[col].unique().tolist()
        candidate_vals = ''
        candidate_len = 0

        # （NaN and None）count
        nan_count = df[col].isnull().sum()

        # inf count
        inf_count = np.isinf(df[col]).sum()

        if col.lower() in bin_col_names:
            numeric_col_type = 'bin_num_feat'
        elif unique_count <= max_unique_count:
            numeric_col_type = 'cate_num_feat&general_num_feat'
            candidate_len = len(unique_val_lst) + 1
            candidate_vals = ','.join([unkown] + [str(i) for i in sorted(unique_val_lst)])
        elif high_val <= 1 and low_val >= 0:
            numeric_col_type = 'conversion_rate_num_feat'
        else:
            numeric_col_type = 'general_num_feat'

        output_lst.append([col, max_val, min_val, high_val, low_val, unique_count, nan_count, inf_count, candidate_vals, candidate_len, numeric_col_type])

    columns = ['col_name','max_val', 'min_val', 'high_val', 'low_val', 'unique_count', 'nan_count', 'inf_count', 'candidate_vals', 'candidate_len', 'numeric_col_type']
    df = pd.DataFrame(output_lst, columns=columns)
    df.to_csv(os.path.join(save_dir, 'numeric_feat_explore.csv'), index=False)
    # df.to_excel('output.xlsx', index=False)

    return df


# ------------------------
# explore different types of category features
# ------------------------
def explore_category_feats(df, categorical_cols, timestamp_col, save_dir, max_unique_count=100):
    output_lst = []
    unkown = 'UNKNOWN'
    for col in categorical_cols:
        # unique count
        unique_count = df[col].nunique()
        unique_val_lst = [str(i) for i in df[col].unique().tolist()]

        # （NaN and None）count
        nan_count = df[col].isnull().sum()

        # 统计每个值的数量和占比
        counts = df[col].value_counts()
        ratios = df[col].value_counts(normalize=True)

        # 合并数量和占比，并按占比倒序排序
        result = pd.DataFrame({'count': counts, 'ratio': ratios})
        result = result.sort_values(by='ratio', ascending=False)

        if unique_count <= max_unique_count:
            candidate_vals = ','.join([unkown] + sorted(unique_val_lst))
        else:
            top100_val = result.iloc[max_unique_count - 1]
            if top100_val['ratio'] > 0.01:
                print('col:{}, top100_ratio:{}'.format(col, top100_val['ratio']))
            candidate_vals = ','.join([unkown] + sorted([str(i) for i in result.head(max_unique_count).index.tolist()]))

        candidate_len = len(candidate_vals.split(','))
        output_lst.append([col, unique_count, nan_count, candidate_vals, candidate_len])

    if timestamp_col:
        # hour
        output_lst.append([timestamp_col + '_hour', 24, 0, ','.join([str(i) for i in range(24)] + [unkown]), 25])
        # week
        output_lst.append([timestamp_col + '_week', 7, 0, ','.join([str(i) for i in range(7)] + [unkown]), 8])

    columns = ['col_name', 'unique_count', 'nan_count', 'candidate_vals', 'candidate_len']
    df = pd.DataFrame(output_lst, columns=columns)
    df.to_csv(os.path.join(save_dir, 'category_feat_explore.csv'), index=False)
    # df.to_excel('output.xlsx', index=False)
    print(df)

    return df


# ------------------------
# explore sequence features
# ------------------------
def explore_seq_feats(df, seq_cols, save_dir, max_unique_count=1000):
    output_lst = []
    unkown = 'UNKNOWN'
    for col in seq_cols:
        # 获取序列的长度
        max_len = df[col].apply(len).max()
        p95_len = df[col].apply(len).quantile(0.95)
        seq_max_len = max_len
        seq_high_len = p95_len

        # 计算序列中值的范围，explode 会将 list 列拆成多行
        exploded = df.explode(col)
        unique_count = exploded[col].nunique()
        unique_val_lst = [str(i) for i in exploded[col].unique().tolist()]

        # 统计每个值的数量和占比
        counts = exploded[col].value_counts()
        ratios = exploded[col].value_counts(normalize=True)

        # 合并数量和占比，并按占比倒序排序
        result = pd.DataFrame({'count': counts, 'ratio': ratios})
        result = result.sort_values(by='ratio', ascending=False)

        if unique_count <= max_unique_count:
            candidate_vals = ','.join([unkown] + sorted(unique_val_lst))
        else:
            top_val = result.iloc[max_unique_count - 1]
            if top_val['ratio'] > 0.01:
                print('col:{}, top_ratio:{}'.format(col, top_val['ratio']))
            candidate_vals = ','.join([unkown] + sorted([str(i) for i in result.head(max_unique_count).index.tolist()]))

        candidate_len = len(candidate_vals.split(','))
        output_lst.append([col, unique_count, candidate_vals, candidate_len, seq_max_len, seq_high_len])

    columns = ['col_name', 'unique_count', 'candidate_vals', 'candidate_len', 'seq_max_len', 'seq_high_len']
    df = pd.DataFrame(output_lst, columns=columns)
    df.to_csv(os.path.join(save_dir, 'seq_feat_explore.csv'), index=False)
    # df.to_excel('output.xlsx', index=False)
    print(df)

    return df


# ------------------------
# Main processing pipeline
# ------------------------
def data_explore(input_csv: str, out_dir: str, target_col: str, id_cols: List[str] = None,
                     numeric_cols: List[str] = None, categorical_cols: List[str] = None, seq_cols: List[str] = None,
                     timestamp_col: str = None):
    # 读取数据，新建输出目录
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    print("Loaded", input_csv, "shape:", df.shape)

    # 数据探索
    basic_eda(df, target=target_col)

    # 提取数值和类别特征
    print("\n====== Numeric cols and Categorical cols ======")
    if seq_cols is None:
        seq_cols = [c for c in df.columns.tolist() if c.endswith("_seq")]
    if numeric_cols is None:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in [target_col]]
        # exclude id and timestamp if numeric
        numeric_cols = [c for c in numeric_cols if c not in id_cols and c != timestamp_col and c not in seq_cols]
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in numeric_cols + id_cols + seq_cols + [target_col, timestamp_col]]
    print("Numeric cols (first 30):", numeric_cols[:30])
    print("Categorical cols (first 30):", categorical_cols[:30])
    print("Sequence cols (first 30):", seq_cols[:30])
    print("=================================================\n")

    print("\n====== Numeric cols explore ======")
    # 数值特征处理
    # 1.常规数值特征限制百分位最大最小值，缺失值处理，进行标准化或归一化
    # 2.转化率特征限制0-1范围，缺失值处理，同时新增贝叶斯平滑新特征
    # 3.对年龄特征的分箱处理
    save_dir = args.out_dir
    numeric_explore_df = explore_numeric_feats(df, numeric_cols, save_dir)
    print(numeric_explore_df.head(3))
    print("=================================================\n")

    print("\n====== Category cols explore ======")
    # 一共有以下几种编码方式，主要用的还是one-hot后做embedding
    # 1.Label Encoding（标签编码）
    # 把类别转换成整数标签，适合有序类别或树模型。
    # 2.One-Hot Encoding（独热编码）
    # 把每个类别拆成独立的二进制特征，适合无序类别。
    # 3.Target Encoding（目标编码）
    # 用类别对应目标变量的均值代替类别，适合类别较多时减少维度。
    # 4.Frequency Encoding（频数编码）
    # 用类别出现的频数替代类别，简单且有效。
    save_dir = args.out_dir
    category_explore_df = explore_category_feats(df, categorical_cols, timestamp_col, save_dir)
    print(category_explore_df.head(3))
    print("=================================================\n")

    print("\n====== Sequence cols explore ======")
    # 转下list格式
    for col in seq_cols:
        # df[col] = df[col].apply(ast.literal_eval)  # Nan会报错
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).strip() != "" else []
        )
    # 确定下序列长度，以及每个值的voc_size
    save_dir = args.out_dir
    seq_explore_df = explore_seq_feats(df, seq_cols, save_dir)
    print(seq_explore_df.head(3))
    print("=================================================\n")


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
    args.input = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Avazu_CTR_Prediction_50K/50krecords.csv'
    args.out_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Avazu_CTR_Prediction_50K/feature/'
    args.target = 'click'
    args.id_col = ['id']
    args.timestamp_col = 'hour'
    """

    # Ali_Display_Ad_Click
    args.input = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Ali_Display_Ad_Click/data.csv'
    args.out_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/Ali_Display_Ad_Click/feature/'
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

    # 0: explore, 1:gen_template.csv, 2: gen_template.json
    args.process_type = 2

    if args.process_type == 0:
        data_explore(args.input,
                     args.out_dir,
                     args.target,
                     id_cols=args.id_col,
                     numeric_cols=args.num_cols,
                     categorical_cols=args.cat_cols,
                     seq_cols=args.seq_cols,
                     timestamp_col=args.timestamp_col)

    # copy file when complete exploration
    if args.process_type == 1:
        source = os.path.join(args.out_dir, "numeric_feat_explore.csv")
        destination = os.path.join(args.out_dir, "numeric_feat_template.csv")
        # shutil.copy(source, destination)
        df_source = pd.read_csv(source)
        df_source["conf_max_val"] = df_source.high_val
        df_source["conf_min_val"] = df_source.low_val
        df_source["conf_fill_na_val"] = df_source.low_val
        df_source["conf_candidate_vals"] = df_source.candidate_vals
        df_source["conf_candidate_len"] = df_source.candidate_len
        df_source.to_csv(destination, index=False, encoding="utf-8")

        source = os.path.join(args.out_dir, "category_feat_explore.csv")
        destination = os.path.join(args.out_dir, "category_feat_template.csv")
        df_source = pd.read_csv(source)
        df_source["conf_candidate_vals"] = df_source.candidate_vals
        df_source["conf_candidate_len"] = df_source.candidate_len
        df_source.to_csv(destination, index=False, encoding="utf-8")
        # shutil.copy(source, destination)

        if args.seq_cols:
            source = os.path.join(args.out_dir, "seq_feat_explore.csv")
            destination = os.path.join(args.out_dir, "seq_feat_template.csv")
            df_source = pd.read_csv(source)
            df_source["conf_candidate_vals"] = df_source.candidate_vals
            df_source["conf_candidate_len"] = df_source.candidate_len
            df_source["conf_sequence_len"] = df_source.seq_high_len
            df_source.to_csv(destination, index=False, encoding="utf-8")
            # shutil.copy(source, destination)

    if args.process_type == 2:
        source = os.path.join(args.out_dir, "numeric_feat_template.csv")
        destination = os.path.join(args.out_dir, "numeric_feat_template.json")
        df = pd.read_csv(source)
        df.to_json(destination, orient="records", force_ascii=False, indent=4)

        source = os.path.join(args.out_dir, "category_feat_template.csv")
        destination = os.path.join(args.out_dir, "category_feat_template.json")
        df = pd.read_csv(source)
        df.to_json(destination, orient="records", force_ascii=False, indent=4)

        if args.seq_cols:
            source = os.path.join(args.out_dir, "seq_feat_template.csv")
            destination = os.path.join(args.out_dir, "seq_feat_template.json")
            df = pd.read_csv(source)
            df.to_json(destination, orient="records", force_ascii=False, indent=4)

