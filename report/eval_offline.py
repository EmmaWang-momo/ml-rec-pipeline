import numpy as np

def hr_at_n(y_true, y_score, n=10):
    # y_true: list of arrays per query (0/1), y_score: list of arrays per query
    hits = 0
    total = 0
    for yt, ys in zip(y_true, y_score):
        idx = np.argsort(ys)[-n:][::-1]
        if yt[idx].sum() > 0:
            hits += 1
        total += 1
    return hits / total

def ndcg_at_n(y_true, y_score, n=10):
    import math
    total = 0.0
    for yt, ys in zip(y_true, y_score):
        idx = np.argsort(ys)[-n:][::-1]
        dcg = 0.0
        for i, idxx in enumerate(idx):
            rel = yt[idxx]
            dcg += (2**rel - 1) / math.log2(i+2)
        # ideal
        ideal_idx = np.argsort(yt)[-n:][::-1]
        idcg = 0.0
        for i, idxx in enumerate(ideal_idx):
            rel = yt[idxx]
            idcg += (2**rel - 1) / math.log2(i+2)
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / len(y_true)