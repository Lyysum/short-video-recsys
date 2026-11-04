"""
排序层：对召回候选集打分排序。
思路（对齐工业界 CTR 预估）：把"用户对视频是否会产生正反馈"建模为二分类问题。
特征 = 用户侧(活跃度/历史CTR/完播率) + 视频侧(品类/时长/热度) + 交叉特征(用户对该视频品类的偏好)。
模型可选逻辑回归(可解释) 或 GBDT(非线性)。用 sklearn 训练。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

_root = Path(__file__).resolve()
while not (list(_root.glob("config.py")) or _root == _root.parent):
    _root = _root.parent
sys.path.insert(0, str(_root))

import config as cfg  # noqa: E402


def _encode(series: pd.Series, vocab) -> pd.Series:
    mapping = {v: i for i, v in enumerate(vocab)}
    return series.map(mapping).fillna(-1).astype(int)


FEAT_COLS = ["activity", "ctr", "complete_rate", "like_rate",
             "duration_sec", "hot_score", "user_cat_pref",
             "city_tier_e", "age_group_e", "vcat_e"]


def build_features(exposure: pd.DataFrame, uf: pd.DataFrame, vf: pd.DataFrame,
                   pref: pd.DataFrame):
    """构造 (用户, 视频) 样本特征，返回 X / y / 主键。"""
    pos_cols = ["is_like", "is_comment", "is_share", "is_follow", "is_complete"]
    df = exposure[["user_id", "video_id", "is_click", "is_complete",
                   "is_like", "is_comment", "is_share", "is_follow",
                   "category", "duration_sec"]].copy()
    df["label"] = (df[pos_cols].sum(axis=1) > 0).astype(int)

    u = uf[["user_id", "city_tier", "age_group", "activity", "ctr", "complete_rate", "like_rate"]]
    v = vf[["video_id", "hot_score"]]  # 类别/时长取自曝光表，避免列名冲突
    df = df.merge(u, on="user_id", how="left").merge(v, on="video_id", how="left")

    # 交叉特征：用户对"该视频所属品类"的偏好分
    df = df.merge(pref, on=["user_id", "category"], how="left")
    df = df.rename(columns={"pref_score": "user_cat_pref"})
    df["user_cat_pref"] = df["user_cat_pref"].fillna(0.0)

    df["city_tier_e"] = _encode(df["city_tier"], sorted(uf["city_tier"].unique()))
    df["age_group_e"] = _encode(df["age_group"], cfg.AGE_GROUPS)
    df["vcat_e"] = _encode(df["category"], cfg.CATEGORIES)

    X = df[FEAT_COLS].fillna(0).astype(float).values
    y = df["label"].values
    keys = df[["user_id", "video_id"]].copy()
    return X, y, keys


def train_model(X, y, model_type=cfg.RANK_MODEL):
    if model_type == "gbdt":
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          random_state=cfg.RANDOM_SEED)
    else:
        model = LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_SEED)
    model.fit(X, y)
    return model


def predict_score(model, X):
    """返回正类概率（推荐打分）。"""
    return model.predict_proba(X)[:, 1]


def build_features_for_pairs(pairs_df: pd.DataFrame, uf: pd.DataFrame, vf: pd.DataFrame, pref: pd.DataFrame):
    """给 (用户, 视频) 候选对构造特征（不含标签，用于线上打分）。"""
    df = pairs_df[["user_id", "video_id"]].copy()
    u = uf[["user_id", "city_tier", "age_group", "activity", "ctr", "complete_rate", "like_rate"]]
    v = vf[["video_id", "category", "duration_sec", "hot_score"]]  # 仅取视频侧无冲突字段
    df = df.merge(u, on="user_id", how="left").merge(v, on="video_id", how="left")
    df = df.merge(pref, on=["user_id", "category"], how="left").rename(columns={"pref_score": "user_cat_pref"})
    df["user_cat_pref"] = df["user_cat_pref"].fillna(0.0)
    df["city_tier_e"] = _encode(df["city_tier"], sorted(uf["city_tier"].unique()))
    df["age_group_e"] = _encode(df["age_group"], cfg.AGE_GROUPS)
    df["vcat_e"] = _encode(df["category"], cfg.CATEGORIES)
    return df[FEAT_COLS].fillna(0).astype(float).values
