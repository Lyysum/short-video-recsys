"""
推荐引擎主流程：串联 召回 -> 排序 -> 评估 -> 画像，并把结果落库（ads_rec_result）。
同时输出离线评估指标（Precision/Recall/NDCG/HitRate@K）与示例用户推荐，供简历与看板使用。
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

_root = Path(__file__).resolve()
while not (list(_root.glob("config.py")) or _root == _root.parent):
    _root = _root.parent
sys.path.insert(0, str(_root))

import config as cfg  # noqa: E402
import db  # noqa: E402
from src.recsys import recall as rc  # noqa: E402
from src.recsys import ranking as rk  # noqa: E402
from src.recsys import evaluate as ev  # noqa: E402
from src.recsys import profile as pf  # noqa: E402


def recommend_for(user_id, interacted, user_pref, vf, item_freq, adj, user_items_all, model, pref, uf, k=cfg.REC_K):
    """对某用户做 召回+排序，返回 Top-K [(video_id, score)]。"""
    pool = set(vf["video_id"]) - set(interacted)
    if not pool:
        return []
    lst_itemcf = rc.itemcf_recall(user_items_all, item_freq, adj, user_id, pool)
    lst_content = rc.content_recall(user_pref, vf, pool)
    lst_hot = rc.hot_recall(vf, pool)
    merged = rc.merge_recall([lst_itemcf, lst_content, lst_hot], weights=[1.5, 1.0, 0.6])
    if not merged:
        return []
    cands = [v for v, _ in merged]
    pairs = pd.DataFrame({"user_id": user_id, "video_id": cands})
    X = rk.build_features_for_pairs(pairs, uf, vf, pref)
    sc = rk.predict_score(model, X)
    order = sorted(zip(cands, sc), key=lambda x: -x[1])
    return [(v, float(s)) for v, s in order[:k]]


def run_recsys(verbose: bool = True):
    conn = db.get_conn()
    exposure = db.read_sql(conn, "SELECT * FROM dwd_exposure")
    uf = db.read_sql(conn, "SELECT * FROM dws_user_feature")
    vf = db.read_sql(conn, "SELECT * FROM dws_video_feature")
    pref = db.read_sql(conn, "SELECT * FROM dws_user_category_pref")
    funnel = db.read_sql(conn, "SELECT * FROM ads_funnel WHERE scope='ALL'").iloc[0]

    # ---- 用户分层训练/测试 ----
    users = uf["user_id"].unique()
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    shuffled = rng.permutation(users)
    n_test = int(len(shuffled) * cfg.TEST_RATIO)
    test_users = set(shuffled[:n_test])
    train_users = set(shuffled[n_test:])

    pos = rc.get_positive(exposure)
    train_pos = pos[pos["user_id"].isin(train_users)]
    test_pos_full = pos[pos["user_id"].isin(test_users)]

    # ---- 召回模型（仅用训练集共现，避免泄漏）----
    user_items_all = pos.groupby("user_id")["video_id"].apply(set).to_dict()
    user_items_train, item_freq, adj = rc.build_itemcf(train_pos)

    # ---- 排序模型（训练集曝光样本）----
    exp_train = exposure[exposure["user_id"].isin(train_users)]
    X, y, _ = rk.build_features(exp_train, uf, vf, pref)
    model = rk.train_model(X, y, cfg.RANK_MODEL)

    # ---- 离线评估（留出法：每用户正样本拆成种子/留出，避免全部排除导致无法命中）----
    test_user_items = test_pos_full.groupby("user_id")["video_id"].apply(list).to_dict()
    recs, test_pos, eval_user_items = {}, {}, {}
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    for u, items in test_user_items.items():
        if len(items) < 4:   # 正样本太少无法评估
            continue
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n_seed = max(1, int(len(shuffled) * 0.7))
        seed_items = set(shuffled[:n_seed])
        holdout = shuffled[n_seed:]
        eval_user_items[u] = seed_items
        up = dict(zip(pref.loc[pref["user_id"] == u, "category"],
                      pref.loc[pref["user_id"] == u, "pref_score"]))
        rec = recommend_for(u, seed_items, up, vf, item_freq, adj, eval_user_items, model, pref, uf, k=20)
        recs[u] = [v for v, _ in rec]
        test_pos[u] = list(holdout)
    metrics = ev.evaluate(recs, test_pos, k=cfg.REC_K)
    metrics.update({k.replace("@10", "@20"): v for k, v in ev.evaluate(recs, test_pos, k=20).items()})

    # ---- 示例用户推荐（落库）----
    bots = uf[uf["is_bot"] == 1]["user_id"].head(1).tolist()
    normals = uf.sort_values("activity", ascending=False)["user_id"].head(4).tolist()
    demo_users = list(dict.fromkeys(bots + normals))[:6]
    vcat = vf.set_index("video_id")["category"].to_dict()

    rec_rows = []
    for u in demo_users:
        interacted = set(user_items_all.get(u, set()))
        up = dict(zip(pref.loc[pref["user_id"] == u, "category"],
                      pref.loc[pref["user_id"] == u, "pref_score"]))
        rec = recommend_for(u, interacted, up, vf, item_freq, adj, user_items_all, model, pref, uf)
        labels = pf.user_profile_labels(uf.loc[uf["user_id"] == u, "top_categories"].iloc[0])
        for rank, (vid, score) in enumerate(rec, 1):
            rec_rows.append({
                "user_id": u, "rank": rank, "video_id": vid,
                "score": round(score, 4), "category": vcat.get(vid, ""),
                "channel": "综合召回(ICF+内容+热门)", "user_tags": ",".join(labels),
            })
    rec_df = pd.DataFrame(rec_rows)
    db.write_table(conn, rec_df, "ads_rec_result")
    conn.close()

    # ---- 保存指标，供 README 引用 ----
    out = {
        "eval_metrics": metrics,
        "model": cfg.RANK_MODEL,
        "funnel": {
            "exposures": int(funnel["exposures"]), "ctr": float(funnel["ctr"]),
            "complete_rate": float(funnel["complete_rate"]), "like_rate": float(funnel["like_rate"]),
        },
        "recall_candidates": cfg.RECALL_CANDIDATES, "k": cfg.REC_K,
    }
    with open(cfg.RAW_DIR.parent / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[recsys] 排序模型={cfg.RANK_MODEL} 训练样本={len(y)} 正样本比={round(y.mean(),3)}")
        print(f"[recsys] 离线评估@K={cfg.REC_K}: {metrics}")
        print(f"[recsys] 示例推荐用户: {demo_users}")
    return out


if __name__ == "__main__":
    run_recsys()
