"""
召回层：从全量视频中快速筛选出候选集。
三路召回（工业界标准做法）：
  1) Item-CF   —— 基于"用户-物品"共现的协同过滤（余弦相似度）
  2) 内容召回  —— 基于用户品类偏好 × 视频内容标签的匹配
  3) 热门召回  —— 全局热度兜底，保证冷启动与候选充足
最后按"排名加权"融合三路结果。
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

_root = Path(__file__).resolve()
while not (list(_root.glob("config.py")) or _root == _root.parent):
    _root = _root.parent
sys.path.insert(0, str(_root))

import config as cfg  # noqa: E402


def get_positive(exposure: pd.DataFrame) -> pd.DataFrame:
    """正反馈（点赞/评论/转发/关注/完播）视为用户对视频的兴趣信号。"""
    cols = ["is_like", "is_comment", "is_share", "is_follow", "is_complete"]
    mask = exposure[cols].sum(axis=1) > 0
    return exposure.loc[mask, ["user_id", "video_id"]].reset_index(drop=True)


def build_itemcf(pos: pd.DataFrame):
    """构建物品共现图，返回 (用户->物品集合, 物品频次, 物品邻接表)。"""
    user_items = pos.groupby("user_id")["video_id"].apply(set).to_dict()
    item_freq = defaultdict(int)
    adj = defaultdict(dict)  # adj[i][j] = 共现次数
    for items in user_items.values():
        items = list(items)
        for it in items:
            item_freq[it] += 1
        n = len(items)
        for a in range(n):
            for b in range(a + 1, n):
                ia, ib = items[a], items[b]
                adj[ia][ib] = adj[ia].get(ib, 0) + 1
                adj[ib][ia] = adj[ib].get(ia, 0) + 1
    return user_items, dict(item_freq), dict(adj)


def itemcf_recall(user_items, item_freq, adj, target_user, candidate_pool, topn=cfg.RECALL_CANDIDATES):
    """Item-CF 召回：用户看过的物品的相似物品。"""
    interacted = user_items.get(target_user, set())
    scores = defaultdict(float)
    for it in interacted:
        fi = item_freq.get(it, 0)
        if fi == 0:
            continue
        for jt, c in adj.get(it, {}).items():
            if jt in interacted or jt not in candidate_pool:
                continue
            fj = item_freq.get(jt, 0)
            if fj == 0:
                continue
            scores[jt] += c / np.sqrt(fi * fj)
    cand = sorted(scores.items(), key=lambda x: -x[1])[:topn]
    return cand


def content_recall(user_pref: dict, video_df: pd.DataFrame, candidate_pool, topn=cfg.RECALL_CANDIDATES):
    """内容召回：用户偏好品类对应的视频得分更高（向量化）。"""
    pool = set(candidate_pool)
    vf = video_df[video_df["video_id"].isin(pool)].copy()
    vf["score"] = vf["category"].map(user_pref).fillna(0.0)
    vf = vf.sort_values("score", ascending=False).head(topn)
    return list(zip(vf["video_id"], vf["score"]))


def hot_recall(video_feature: pd.DataFrame, candidate_pool, topn=cfg.RECALL_CANDIDATES):
    """热门召回：全局热度榜兜底。"""
    pool = set(candidate_pool)
    vf = video_feature[video_feature["video_id"].isin(pool)].sort_values("hot_score", ascending=False).head(topn)
    return list(zip(vf["video_id"], vf["hot_score"]))


def merge_recall(lists, weights=None, topn=cfg.RECALL_CANDIDATES):
    """融合多路召回：各路分数先归一化到 [0,1]，再按权重×排名(1/(1+rank))融合，
    避免高热度分淹没协同过滤分。"""
    if weights is None:
        weights = [1.0] * len(lists)
    merged = defaultdict(float)
    for lst, w in zip(lists, weights):
        if not lst:
            continue
        mx = max(s for _, s in lst) or 1.0
        for rank, (vid, s) in enumerate(lst):
            merged[vid] += (float(s) / mx) * w * (1.0 / (1 + rank))
    return sorted(merged.items(), key=lambda x: -x[1])[:topn]
