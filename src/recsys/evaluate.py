"""
评估层：用离线 hold-out 验证推荐效果。
指标：Precision@K / Recall@K / NDCG@K / HitRate@K（Top-K 推荐与用户真实正反馈的交集）。
"""
import numpy as np


def _metrics(recommended, relevant, k):
    rec_k = recommended[:k]
    rec_set = set(rec_k)
    hits = [1 if v in rec_set else 0 for v in relevant]
    precision = sum(hits) / k if k else 0.0
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg = sum((2 ** h - 1) / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum((2 ** 1 - 1) / np.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / ideal if ideal else 0.0
    hit_rate = 1.0 if any(hits) else 0.0
    return precision, recall, ndcg, hit_rate


def evaluate(recs: dict, test_pos: dict, k: int = 10):
    """recs: {user: [video_ids]}; test_pos: {user: [relevant video_ids]}。"""
    ps, rs, ns, hs = [], [], [], []
    for u, rel in test_pos.items():
        if u not in recs or not rel:
            continue
        p, r, n, h = _metrics(recs[u], rel, k)
        ps.append(p); rs.append(r); ns.append(n); hs.append(h)
    if not ps:
        return {f"Precision@{k}": 0, f"Recall@{k}": 0, f"NDCG@{k}": 0, f"HitRate@{k}": 0}
    return {
        f"Precision@{k}": round(float(np.mean(ps)), 4),
        f"Recall@{k}": round(float(np.mean(rs)), 4),
        f"NDCG@{k}": round(float(np.mean(ns)), 4),
        f"HitRate@{k}": round(float(np.mean(hs)), 4),
    }
