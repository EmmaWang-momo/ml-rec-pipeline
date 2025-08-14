"""
本地训练 TF DCN（Cross Layer + Deep Tower），保存为 SavedModel
用法示例：
python models/dcn/train.py --data_path data/sample.csv --model_dir artifacts/saved_models/dcn --nrows 20000
"""
import os
import argparse
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import (layers, initializers, metrics)


# ---- CrossLayer ----
class CrossLayer(layers.Layer):
    def __init__(self, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.ws = [self.add_weight(shape=(dim,1), initializer='glorot_uniform', name=f'w_{i}') for i in range(self.num_layers)]
        self.bs = [self.add_weight(shape=(dim,), initializer='zeros', name=f'b_{i}') for i in range(self.num_layers)]
        super().build(input_shape)

    def call(self, x):
        x0 = x
        xl = x
        for i in range(self.num_layers):
            dot = tf.matmul(xl, self.ws[i])  # (batch,1)
            xl = x0 * dot + self.bs[i] + xl
        return xl


# ---- Build DCN Model ----
def build_dcn_model(cat_features, cat_features_vocab_sizes, num_features, emb_dim, cross_layers, deep_units,
                    embed_init=initializers.glorot_normal(),
                    kernel_init=initializers.glorot_normal(),
                    kernel_regular=None):
    total_len = len(cat_features) + len(num_features)
    feature_inputs = keras.Input(shape=(total_len,), dtype=tf.float32, name="feature_inputs")

    # 拆分输入：假设前面是类别特征，后面是数值特征
    cate_inputs = feature_inputs[:, :len(cat_features)]  # shape (batch, cat_len)
    numeric_inputs = feature_inputs[:, len(cat_features):]  # shape (batch, num_len)

    # 离散特征统一拼接到一个空间
    print('---------------------------------------------------')
    cate_features = tf.cast(cate_inputs, tf.int32)
    cate_vocab_sizes = tf.constant(cat_features_vocab_sizes, dtype=tf.int32)
    cate_features_offsets = tf.concat([[0], tf.cumsum(cate_vocab_sizes[:-1])], axis=0)
    # cate_features_offsets = tf.expand_dims(cate_features_offsets, axis=0)
    cate_features = cate_features + cate_features_offsets
    cate_features = tf.clip_by_value(cate_features, clip_value_min=0, clip_value_max=sum(cat_features_vocab_sizes) - 1)

    # Embedding
    cate_embed = layers.Embedding(input_dim=sum(cat_features_vocab_sizes),
                                  output_dim=emb_dim,
                                  embeddings_initializer=embed_init,
                                  embeddings_regularizer=kernel_regular,
                                  input_length=None)(cate_features)
    cate_embed = tf.reshape(cate_embed, [-1, emb_dim * len(cat_features)])

    # 数值特征直接用
    num_feat = numeric_inputs

    # 拼接数值和embedding
    x = layers.Concatenate()([cate_embed, num_feat])

    # Cross Network
    cross_out = CrossLayer(num_layers=cross_layers)(x)

    # Deep Network
    deep_out = x
    for i, u in enumerate(deep_units):
        deep_out = layers.Dense(u, activation='relu', name=f'deep_{i}')(deep_out)
        deep_out = layers.Dropout(0.2)(deep_out)

    # 拼接
    concat = layers.Concatenate()([cross_out, deep_out])
    out = layers.Dense(1, activation='sigmoid', name='model_output')(concat)

    model = keras.Model(inputs=feature_inputs, outputs=out)
    return model


# ---- main ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', required=True)
    parser.add_argument('--model_dir', default='artifacts/saved_models/dcn')
    parser.add_argument('--nrows', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    args.data_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/data_set/'
    args.model_dir = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/artifacts/saved_models/dcn'
    args.config_file = '/adjoe/data_set/Avazu_CTR_Prediction_50K/feature_spec.json'
    args.nrows = None
    args.batch_size = 256
    args.epochs = 20

    import json
    with open(args.config_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    num_features = data['numeric_cols']
    cat_features = data['categorical_cols']

    cat_features_vocab_dict = data['categorical_cols_len']
    cat_features_vocab_sizes = []
    for col in cat_features:
        cat_features_vocab_sizes.append(cat_features_vocab_dict[col])
    print(cat_features_vocab_sizes)

    label_name = 'click'

    os.makedirs(args.model_dir, exist_ok=True)
    train_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), nrows=args.nrows)
    val_df = pd.read_csv(os.path.join(args.data_dir, 'val.csv'))
    print("df size: ", train_df.shape, val_df.shape)

    X_train, y_train = train_df[cat_features + num_features], train_df[label_name].astype(int).values
    X_val, y_val = val_df[cat_features + num_features], val_df[label_name].astype(int).values

    # ---- Hyperparams ----
    embedding_dim = 8
    cross_layer = 3
    deep_units = [128, 64]

    model = build_dcn_model(cat_features, cat_features_vocab_sizes, num_features, embedding_dim, cross_layer, deep_units)
    print(model.summary())
    print(model.input)  # 打印输入张量（Tensor），包含名字和形状
    print(model.input_names)  # 打印输入层的名字列表（一般是个list）
    print(model.output)  # 输出张量，里面包含名字
    print(model.output_names)  # 输出层的名字列表
    print(model.outputs)  # 输出张量列表

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # 保存 SavedModel
    model.save(args.model_dir, include_optimizer=False)
    print('SavedModel written to', args.model_dir)

    # 保存 feature spec
    spec = {'cat_features': cat_features, 'num_features': num_features}
    import json
    with open(os.path.join(args.model_dir, 'feature_spec.json'), 'w') as f:
        json.dump(spec, f)