"""
CSV replay 到 HTTP endpoint。CSV 中应包含 feature 列与 impression id。
示例：
python stream/generator/replay.py --csv data/sample.csv --target http://localhost:8080/predict --qps 50 --nrows 1000
"""
import argparse
import pandas as pd
import requests
import time
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--qps', type=float, default=50.0)
    parser.add_argument('--nrows', type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, nrows=args.nrows)
    # assume df has impression_id, user_id, and feature columns
    feature_cols = [c for c in df.columns if c not in ('impression_id', 'user_id', 'click', 'label', 'timestamp')]
    interval = 1.0 / args.qps

    for _, row in tqdm(df.iterrows(), total=len(df)):
        payload = {
            'impression_id': str(row.get('impression_id', '0')),
            'user_id': str(row.get('user_id', '')),
            'features': {c: row[c] for c in feature_cols if c in row.index},
            'timestamp': int(row.get('timestamp', 0))
        }
        try:
            r = requests.post(args.target, json=payload, timeout=1.0)
        except Exception as e:
            print('request failed', e)
        time.sleep(interval)