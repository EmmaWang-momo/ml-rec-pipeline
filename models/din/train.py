# din_mixed_features.py
import argparse
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------
# Attention Block (DIN核心)
# ---------------------------
class DINAttention(layers.Layer):
    """
    Target-aware attention for DIN:
    For each timestep i: score_i = MLP([h_i, v, h_i - v, h_i * v])
    Then softmax over valid positions (masked).
    """
    def __init__(self, hidden_units: List[int] = [64, 32], activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.mlp = []
        for u in hidden_units:
            self.mlp.append(layers.Dense(u, activation=activation))
        self.proj = layers.Dense(1, activation=None)

    def call(self, hist_emb, target_emb, mask=None):
        """
        hist_emb: (B, T, D)
        target_emb: (B, D)  or (B, 1, D)
        mask: (B, T) bool, True for valid; False for padding
        """
        if tf.rank(target_emb) == 2:
            # (B, D) -> (B, 1, D) -> tile
            target = tf.expand_dims(target_emb, axis=1)
        else:
            target = target_emb  # assume (B, 1, D)
        T = tf.shape(hist_emb)[1]
        target_rep = tf.repeat(target, repeats=T, axis=1)  # (B, T, D)

        att_input = tf.concat(
            [hist_emb, target_rep, hist_emb - target_rep, hist_emb * target_rep],
            axis=-1
        )  # (B, T, 4D)

        x = att_input
        for dense in self.mlp:
            x = dense(x)
        logits = self.proj(x)  # (B, T, 1)

        # mask & softmax over time dimension
        if mask is not None:
            mask = tf.cast(mask, tf.bool)                # (B, T)
            mask = tf.expand_dims(mask, axis=-1)         # (B, T, 1)
            neg_inf = tf.ones_like(logits) * (tf.float32.min / 2.0)
            logits = tf.where(mask, logits, neg_inf)

        att_weights = tf.nn.softmax(logits, axis=1)      # (B, T, 1)
        context = tf.reduce_sum(att_weights * hist_emb, axis=1)  # (B, D)
        return context, att_weights


# -----------------------------------
# 构建可配置的 DIN 模型（Functional API）
# -----------------------------------
def build_din_model(
    num_dense: int,
    categorical_vocab: Dict[str, int],
    seq_len: int,
    category_vocab_size: int,
    brand_vocab_size: int,
    emb_dim_common: int = 16,
    emb_dim_seq: int = 16,
    mlp_units: List[int] = [128, 64, 32],
    att_units: List[int] = [64, 32],
    dropout: float = 0.2,
    final_hidden_activation: str = "relu",
):
    """
    参数说明：
    - num_dense: 连续特征数量
    - categorical_vocab: 其他离散特征及其词表大小（不含 category/brand 的序列与目标）
    - seq_len: 行为序列长度
    - category_vocab_size/brand_vocab_size: 品类/品牌词表大小（0 作为 padding）
    - emb_dim_common: 其他离散特征 & 目标特征 embedding 维度
    - emb_dim_seq: 序列 embedding 维度
    - mlp_units: 主干 MLP 层
    - att_units: 注意力 MLP 层
    """
    inputs = {}

    # 1) 连续特征
    if num_dense > 0:
        inputs["dense"] = layers.Input(shape=(num_dense,), dtype=tf.float32, name="dense_input")

    # 2) 其他离散特征（例：user_id, device, city 等）
    cat_inputs = {}
    for name, vocab in categorical_vocab.items():
        cat_inputs[name] = layers.Input(shape=(1,), dtype=tf.int32, name=f"{name}_input")
        inputs[f"{name}_input"] = cat_inputs[name]

    # 3) 目标品类/品牌
    target_category = layers.Input(shape=(1,), dtype=tf.int32, name="target_category_id")
    target_brand = layers.Input(shape=(1,), dtype=tf.int32, name="target_brand_id")
    inputs["target_category_id"] = target_category
    inputs["target_brand_id"] = target_brand

    # 4) 行为序列（品类/品牌）
    hist_category = layers.Input(shape=(seq_len,), dtype=tf.int32, name="hist_category_ids")
    hist_brand = layers.Input(shape=(seq_len,), dtype=tf.int32, name="hist_brand_ids")
    inputs["hist_category_ids"] = hist_category
    inputs["hist_brand_ids"] = hist_brand

    # 可选：提供显式的有效长度（若不提供，将用 id==0 当作 padding）
    seq_len_input = layers.Input(shape=(1,), dtype=tf.int32, name="seq_len_optional")
    inputs["seq_len_optional"] = seq_len_input

    # ----------------- Embeddings -----------------
    # 序列专用 embedding（也可与目标共享，这里分开更灵活）
    category_seq_emb = layers.Embedding(
        input_dim=category_vocab_size + 1, output_dim=emb_dim_seq, mask_zero=False, name="category_seq_emb"
    )
    brand_seq_emb = layers.Embedding(
        input_dim=brand_vocab_size + 1, output_dim=emb_dim_seq, mask_zero=False, name="brand_seq_emb"
    )

    # 目标/其他特征 embedding
    category_emb = layers.Embedding(
        input_dim=category_vocab_size + 1, output_dim=emb_dim_common, mask_zero=False, name="category_emb"
    )
    brand_emb = layers.Embedding(
        input_dim=brand_vocab_size + 1, output_dim=emb_dim_common, mask_zero=False, name="brand_emb"
    )

    other_cat_emb_layers = {
        name: layers.Embedding(input_dim=vocab + 1, output_dim=emb_dim_common, mask_zero=False, name=f"{name}_emb")
        for name, vocab in categorical_vocab.items()
    }

    # ----------------- Lookup -----------------
    hist_cat_e = category_seq_emb(hist_category)  # (B, T, Dseq)
    hist_brd_e = brand_seq_emb(hist_brand)        # (B, T, Dseq)

    tgt_cat_e = category_emb(target_category)     # (B, 1, Demb)
    tgt_brd_e = brand_emb(target_brand)           # (B, 1, Demb)
    tgt_cat_e_flat = layers.Flatten()(tgt_cat_e)  # (B, Demb)
    tgt_brd_e_flat = layers.Flatten()(tgt_brd_e)  # (B, Demb)

    other_embs = []
    for name, inp in cat_inputs.items():
        e = other_cat_emb_layers[name](inp)       # (B, 1, Demb)
        other_embs.append(layers.Flatten()(e))    # -> (B, Demb)

    # ----------------- Masks -----------------
    # 使用 id==0 作为 padding；若提供 seq_len_optional，也可辅助构造 mask
    base_mask_cat = tf.not_equal(hist_category, 0)  # (B, T)
    base_mask_brd = tf.not_equal(hist_brand, 0)     # (B, T)

    # 若提供 seq_len_optional > 0，则按长度再约束一层
    seq_len_b = tf.squeeze(seq_len_input, axis=-1)  # (B,)
    rng = tf.range(tf.shape(hist_category)[1])[tf.newaxis, :]  # (1, T)
    len_mask = tf.cast(rng, tf.int32) < tf.expand_dims(seq_len_b, 1)  # (B, T)
    final_mask_cat = tf.logical_and(base_mask_cat, len_mask)
    final_mask_brd = tf.logical_and(base_mask_brd, len_mask)

    # ----------------- DIN Attention -----------------
    att_cat = DINAttention(att_units, name="din_att_category")
    att_brd = DINAttention(att_units, name="din_att_brand")

    # 注意：让 target 的维度与对应序列的 embedding 维度一致
    # 若 emb_dim_common != emb_dim_seq，可用线性层对齐
    if emb_dim_common != emb_dim_seq:
        align_cat = layers.Dense(emb_dim_seq, use_bias=False, name="align_cat")(tgt_cat_e_flat)
        align_brd = layers.Dense(emb_dim_seq, use_bias=False, name="align_brd")(tgt_brd_e_flat)
    else:
        align_cat, align_brd = tgt_cat_e_flat, tgt_brd_e_flat

    user_interest_cat, _ = att_cat(hist_cat_e, align_cat, mask=final_mask_cat)
    user_interest_brd, _ = att_brd(hist_brd_e, align_brd, mask=final_mask_brd)

    # ----------------- 拼接所有特征 -----------------
    feats = []
    if num_dense > 0:
        feats.append(inputs["dense"])
    feats += other_embs
    feats += [tgt_cat_e_flat, tgt_brd_e_flat, user_interest_cat, user_interest_brd]

    x = layers.Concatenate(name="concat_all")(feats)

    # ----------------- 主干 MLP -----------------
    for i, u in enumerate(mlp_units):
        x = layers.Dense(u, activation=final_hidden_activation, name=f"fc_{i}")(x)
        x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    out = layers.Dense(1, activation="sigmoid", name="ctr")(x)
    model = Model(inputs=list(inputs.values()), outputs=out, name="DIN_MixedFeatures")
    return model


# -----------------------------------
# 随机数据生成（Demo用，便于跑通）
# -----------------------------------
def gen_dummy_data(
    num_samples: int,
    num_dense: int,
    categorical_vocab: Dict[str, int],
    seq_len: int,
    category_vocab_size: int,
    brand_vocab_size: int,
):
    rng = np.random.default_rng(42)

    data = {}
    if num_dense > 0:
        data["dense_input"] = rng.normal(size=(num_samples, num_dense)).astype("float32")

    # 其他离散特征：取值范围 [1, vocab]，0 预留给padding（这里不使用0）
    for name, vocab in categorical_vocab.items():
        data[f"{name}_input"] = rng.integers(1, vocab + 1, size=(num_samples, 1), dtype=np.int32)

    # 目标品类/品牌
    data["target_category_id"] = rng.integers(1, category_vocab_size + 1, size=(num_samples, 1), dtype=np.int32)
    data["target_brand_id"] = rng.integers(1, brand_vocab_size + 1, size=(num_samples, 1), dtype=np.int32)

    # 行为序列（带随机长度与padding=0）
    hist_cat = np.zeros((num_samples, seq_len), dtype=np.int32)
    hist_brd = np.zeros((num_samples, seq_len), dtype=np.int32)
    seq_lens = rng.integers(low=max(1, seq_len // 3), high=seq_len + 1, size=(num_samples,), dtype=np.int32)
    for i in range(num_samples):
        L = seq_lens[i]
        hist_cat[i, :L] = rng.integers(1, category_vocab_size + 1, size=(L,), dtype=np.int32)
        hist_brd[i, :L] = rng.integers(1, brand_vocab_size + 1, size=(L,), dtype=np.int32)
    data["hist_category_ids"] = hist_cat
    data["hist_brand_ids"] = hist_brd
    data["seq_len_optional"] = seq_lens.reshape(-1, 1)

    # 标签（CTR）
    y = rng.integers(0, 2, size=(num_samples, 1)).astype("float32")
    return data, y


# -----------------------------------
# 训练入口
# -----------------------------------
def parse_args():
    p = argparse.ArgumentParser("DIN for CTR with dense + categorical + behavior sequences")
    p.add_argument("--num_samples", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--num_dense", type=int, default=5)

    # 词表大小（示例）
    p.add_argument("--category_vocab_size", type=int, default=2000)
    p.add_argument("--brand_vocab_size", type=int, default=5000)

    # 其他离散特征（示例）——可以按需修改或在真实工程中从配置/特征字典读取
    p.add_argument("--user_vocab_size", type=int, default=100000)
    p.add_argument("--device_vocab_size", type=int, default=20)
    p.add_argument("--city_vocab_size", type=int, default=300)

    # Embedding & MLP
    p.add_argument("--emb_dim_common", type=int, default=16)
    p.add_argument("--emb_dim_seq", type=int, default=16)
    p.add_argument("--mlp_units", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--att_units", type=int, nargs="+", default=[64, 32])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()

    # 配置“其他离散特征”
    categorical_vocab = {
        "user_id": args.user_vocab_size,
        "device": args.device_vocab_size,
        "city": args.city_vocab_size,
    }

    model = build_din_model(
        num_dense=args.num_dense,
        categorical_vocab=categorical_vocab,
        seq_len=args.seq_len,
        category_vocab_size=args.category_vocab_size,
        brand_vocab_size=args.brand_vocab_size,
        emb_dim_common=args.emb_dim_common,
        emb_dim_seq=args.emb_dim_seq,
        mlp_units=args.mlp_units,
        att_units=args.att_units,
        dropout=args.dropout,
        final_hidden_activation="relu",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="AUC"), tf.keras.metrics.BinaryAccuracy(name="ACC")],
    )
    model.summary(line_length=120)

    # 生成/加载数据（这里用随机数据演示）
    data, y = gen_dummy_data(
        num_samples=args.num_samples,
        num_dense=args.num_dense,
        categorical_vocab=categorical_vocab,
        seq_len=args.seq_len,
        category_vocab_size=args.category_vocab_size,
        brand_vocab_size=args.brand_vocab_size,
    )

    # 训练
    model.fit(
        data,
        y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        verbose=1,
    )


if __name__ == "__main__":
    main()
