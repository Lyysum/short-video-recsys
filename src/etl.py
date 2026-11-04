"""
ETL + 数仓建模（SQLite）。
分层：
  ODS  贴源层：原始数据原样落库
  DWD  明细层：清洗（去空/去重/修时间）、维度退化、曝光粒度事实表
  DWS  汇总层：用户/视频 日粒度与特征宽表
  ADS  应用层：转化漏斗、热门榜单、用户画像、品类统计（直接服务看板与推荐）

建模方法：维度建模（星型模型）—— 事实表 fact_exposure + 维度表 dim_user / dim_video / dim_date。
该思想与 Hive/Spark 数仓一致，仅存储介质换成本地 SQLite，可平滑迁移到大数据集群。
"""
from __future__ import annotations
import sys
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

# 把项目根目录加入 sys.path，保证直接运行本脚本时也能 import config / db
_root = Path(__file__).resolve()
while not (list(_root.glob("config.py")) or _root == _root.parent):
    _root = _root.parent
sys.path.insert(0, str(_root))

import config as cfg
import db


EVENTS = ["impression", "click", "play", "complete", "like", "comment", "share", "follow"]
# 视频热度打分权重（互动权重高于曝光/点击，贴近推荐场景）
HOT_WEIGHTS = {"exposure": 0.2, "click": 1.0, "complete": 2.0,
               "like": 3.0, "comment": 2.0, "share": 4.0, "follow": 5.0}


def _load_raw():
    users = pd.read_csv(cfg.RAW_DIR / "dim_user.csv", dtype=str)
    videos = pd.read_csv(cfg.RAW_DIR / "dim_video.csv", dtype=str)
    log = pd.read_csv(cfg.RAW_DIR / "user_action_log.csv", dtype=str)
    return users, videos, log


def build_ods(conn, users, videos, log):
    db.write_table(conn, users, "ods_user")
    db.write_table(conn, videos, "ods_video")
    db.write_table(conn, log, "ods_action_log")


def build_dwd(conn, users, videos, log):
    # ---- 维度清洗 ----
    dwd_user = users.copy()
    dwd_user["activity"] = dwd_user["activity"].astype(int)
    dwd_video = videos.copy()
    dwd_video["duration_sec"] = dwd_video["duration_sec"].astype(int)
    dwd_video["publish_date"] = pd.to_datetime(dwd_video["publish_time"]).dt.date.astype(str)

    # ---- 事实日志清洗（数据质量治理）----
    raw_n = len(log)
    df = log.copy()
    # 1) 空 user_id 丢弃
    null_user = df["user_id"].isna() | (df["user_id"].astype(str).str.strip() == "")
    n_null = int(null_user.sum())
    df = df[~null_user].copy()
    # 2) 解析时间 + 迟到标记
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    end = pd.Timestamp(cfg.START_DATE) + pd.Timedelta(days=cfg.N_DAYS)
    df["is_late"] = df["event_time"] > end
    n_late = int(df["is_late"].sum())
    df["watch_duration"] = pd.to_numeric(df["watch_duration"], errors="coerce")
    # 3) 去重：同一曝光同一事件只保留最早一条（消除注入的重复事件）
    before = len(df)
    df = df.sort_values(["impression_id", "event_type", "event_time"])
    df = df.drop_duplicates(subset=["impression_id", "event_type"], keep="first")
    n_dup = before - len(df)
    df = df.sort_values(["impression_id", "event_time"]).reset_index(drop=True)

    db.write_table(conn, dwd_user, "dwd_user")
    db.write_table(conn, dwd_video, "dwd_video")
    db.write_table(conn, df, "dwd_action_log")

    # ---- 曝光粒度事实表：事件日志 -> 一行一次曝光 ----
    piv = df.pivot_table(index=["impression_id", "user_id", "video_id"],
                         columns="event_type", values="event_time", aggfunc="min").reset_index()
    for ev in EVENTS:
        if ev not in piv.columns:
            piv[ev] = pd.NaT
        piv[f"is_{ev}"] = piv[ev].notna().astype(int)
    wd = df.groupby("impression_id")["watch_duration"].max().rename("watch_duration")
    piv = piv.merge(wd, on="impression_id", how="left")
    piv = piv.rename(columns={"impression": "imp_time"})
    piv = piv.merge(dwd_video[["video_id", "duration_sec", "category", "author_id"]],
                    on="video_id", how="left")
    piv["watch_ratio"] = (piv["watch_duration"] / piv["duration_sec"]).fillna(0).clip(0, 1)
    piv = piv.drop(columns=[e for e in EVENTS if e != "impression"])
    db.write_table(conn, piv, "dwd_exposure")

    return {"raw_events": raw_n, "dropped_null_user": n_null,
            "dropped_dup": n_dup, "late_events": n_late,
            "clean_exposures": len(piv)}


def build_dws(conn):
    exp = db.read_sql(conn, "SELECT * FROM dwd_exposure")
    exp["imp_date"] = pd.to_datetime(exp["imp_time"]).dt.date.astype(str)

    # 用户日粒度
    ud = exp.groupby(["user_id", "imp_date"]).agg(
        exposures=("impression_id", "nunique"),
        clicks=("is_click", "sum"), completes=("is_complete", "sum"),
        likes=("is_like", "sum"), comments=("is_comment", "sum"),
        shares=("is_share", "sum"), follows=("is_follow", "sum"),
        watch_sec=("watch_duration", "sum"),
    ).reset_index()
    db.write_table(conn, ud, "dws_user_daily")

    # 视频日粒度
    vd = exp.groupby(["video_id", "imp_date"]).agg(
        exposures=("impression_id", "nunique"), clicks=("is_click", "sum"),
        completes=("is_complete", "sum"), likes=("is_like", "sum"),
        comments=("is_comment", "sum"), shares=("is_share", "sum"),
        follows=("is_follow", "sum"),
    ).reset_index()
    db.write_table(conn, vd, "dws_video_daily")

    # 用户特征宽表
    g = exp.groupby("user_id").agg(
        n_exposure=("impression_id", "nunique"), n_click=("is_click", "sum"),
        n_complete=("is_complete", "sum"), n_like=("is_like", "sum"),
        n_comment=("is_comment", "sum"), n_share=("is_share", "sum"),
        n_follow=("is_follow", "sum"),
    ).reset_index()
    g["ctr"] = (g["n_click"] / g["n_exposure"]).round(4)
    g["complete_rate"] = (g["n_complete"] / g["n_click"].replace(0, np.nan)).round(4).fillna(0)
    g["like_rate"] = (g["n_like"] / g["n_complete"].replace(0, np.nan)).round(4).fillna(0)

    # 品类偏好（基于正反馈曝光）
    pos = exp[exp[["is_like", "is_complete", "is_comment", "is_share", "is_follow"]].sum(axis=1) > 0]
    cat_cnt = pos.groupby(["user_id", "category"]).size().rename("cnt").reset_index()
    pref = cat_cnt.pivot_table(index="user_id", columns="category", values="cnt", fill_value=0)
    pref = pref.div(pref.sum(axis=1), axis=0).fillna(0)  # 归一化
    top = pref.apply(lambda r: r.nlargest(3).index.tolist(), axis=1)
    pref_df = pref.reset_index()
    pref_long = pref_df.melt(id_vars="user_id", var_name="category", value_name="pref_score")
    top_df = top.rename("top_categories").reset_index()
    top_df["top_categories"] = top_df["top_categories"].apply(
        lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x))
    g = g.merge(top_df, on="user_id", how="left")
    user_attr = db.read_sql(conn, "SELECT user_id, city_tier, age_group, gender, activity, is_bot FROM dwd_user")
    g = g.merge(user_attr, on="user_id", how="left")
    db.write_table(conn, g, "dws_user_feature")
    db.write_table(conn, pref_long, "dws_user_category_pref")

    # 视频特征宽表 + 热度分
    vf = exp.groupby("video_id").agg(
        n_exposure=("impression_id", "nunique"), n_click=("is_click", "sum"),
        n_complete=("is_complete", "sum"), n_like=("is_like", "sum"),
        n_comment=("is_comment", "sum"), n_share=("is_share", "sum"),
        n_follow=("is_follow", "sum"),
    ).reset_index()
    vf["ctr"] = (vf["n_click"] / vf["n_exposure"]).round(4)
    vf["complete_rate"] = (vf["n_complete"] / vf["n_click"].replace(0, np.nan)).round(4).fillna(0)
    vf["hot_score"] = (
        vf["n_exposure"] * HOT_WEIGHTS["exposure"] + vf["n_click"] * HOT_WEIGHTS["click"]
        + vf["n_complete"] * HOT_WEIGHTS["complete"] + vf["n_like"] * HOT_WEIGHTS["like"]
        + vf["n_comment"] * HOT_WEIGHTS["comment"] + vf["n_share"] * HOT_WEIGHTS["share"]
        + vf["n_follow"] * HOT_WEIGHTS["follow"]
    ).round(2)
    vf = vf.merge(
        db.read_sql(conn, "SELECT video_id, category, duration_sec, author_id, publish_date FROM dwd_video"),
        on="video_id", how="left")
    db.write_table(conn, vf, "dws_video_feature")
    return {"users": len(g), "videos": len(vf)}


def build_ads(conn):
    exp = db.read_sql(conn, "SELECT * FROM dwd_exposure")

    def funnel(df):
        imp = df["impression_id"].nunique()
        row = {
            "exposures": imp,
            "clicks": int(df["is_click"].sum()),
            "completes": int(df["is_complete"].sum()),
            "likes": int(df["is_like"].sum()),
            "comments": int(df["is_comment"].sum()),
            "shares": int(df["is_share"].sum()),
            "follows": int(df["is_follow"].sum()),
        }
        row["ctr"] = round(row["clicks"] / imp, 4) if imp else 0
        row["complete_rate"] = round(row["completes"] / row["clicks"], 4) if row["clicks"] else 0
        row["like_rate"] = round(row["likes"] / row["completes"], 4) if row["completes"] else 0
        row["interact_rate"] = round((row["likes"] + row["comments"] + row["shares"] + row["follows"]) / row["clicks"], 4) if row["clicks"] else 0
        return row

    # 整体 + 分品类
    rows = [{"scope": "ALL", **funnel(exp)}]
    for cat, sub in exp.groupby("category"):
        rows.append({"scope": cat, **funnel(sub)})
    ads_funnel = pd.DataFrame(rows)
    db.write_table(conn, ads_funnel, "ads_funnel")

    # 热门视频榜
    vf = db.read_sql(conn, "SELECT video_id, category, author_id, duration_sec, n_exposure, n_click, n_complete, n_like, ctr, hot_score FROM dws_video_feature")
    hot = vf.sort_values("hot_score", ascending=False).head(50).reset_index(drop=True)
    db.write_table(conn, hot, "ads_video_hot")

    # 用户画像（看板用）
    uf = db.read_sql(conn, "SELECT user_id, city_tier, age_group, gender, top_categories, ctr, complete_rate, like_rate, is_bot FROM dws_user_feature")
    uf["top_categories"] = uf["top_categories"].astype(str)
    db.write_table(conn, uf, "ads_user_profile")

    # 品类统计
    cat = exp.groupby("category").agg(
        exposures=("impression_id", "nunique"), clicks=("is_click", "sum"),
        completes=("is_complete", "sum"), likes=("is_like", "sum"),
    ).reset_index()
    cat["ctr"] = (cat["clicks"] / cat["exposures"]).round(4)
    db.write_table(conn, cat, "ads_category_stat")
    return {"funnel_rows": len(ads_funnel), "hot_top": len(hot)}


def run_etl(verbose=True):
    conn = db.get_conn()
    conn.executescript("""
        DROP TABLE IF EXISTS ods_user; DROP TABLE IF EXISTS ods_video; DROP TABLE IF EXISTS ods_action_log;
        DROP TABLE IF EXISTS dwd_user; DROP TABLE IF EXISTS dwd_video; DROP TABLE IF EXISTS dwd_action_log; DROP TABLE IF EXISTS dwd_exposure;
        DROP TABLE IF EXISTS dws_user_daily; DROP TABLE IF EXISTS dws_video_daily; DROP TABLE IF EXISTS dws_user_feature; DROP TABLE IF EXISTS dws_video_feature; DROP TABLE IF EXISTS dws_user_category_pref;
        DROP TABLE IF EXISTS ads_funnel; DROP TABLE IF EXISTS ads_video_hot; DROP TABLE IF EXISTS ads_user_profile; DROP TABLE IF EXISTS ads_category_stat;
    """)
    conn.commit()

    users, videos, log = _load_raw()
    build_ods(conn, users, videos, log)
    dwd_stat = build_dwd(conn, users, videos, log)
    dws_stat = build_dws(conn)
    ads_stat = build_ads(conn)
    conn.close()

    if verbose:
        print("[etl] 数据质量:", dwd_stat)
        print(f"[etl] 数仓就绪: 用户={dws_stat['users']} 视频={dws_stat['videos']} 漏斗维度={ads_stat['funnel_rows']}")
    return {**dwd_stat, **dws_stat, **ads_stat}


if __name__ == "__main__":
    run_etl()
