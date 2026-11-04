"""
数据生成器：合成一套"抖音式"短视频平台行为数据。
产物（落到 data/raw/）：
  - dim_user.csv        用户维表
  - dim_video.csv       视频维表
  - user_action_log.csv 用户行为事件日志（事件粒度，模拟前端 SDK 埋点）

设计要点：
  1) 每个用户有隐式"品类偏好"，视频选择偏向其偏好品类 —— 保证推荐系统可学、可解释。
  2) 刻意注入生产级脏数据：重复事件、空 user_id、乱序/迟到时间戳、机器人异常行为，
     用于检验 ETL 数据质量治理能力（这正是数据开发岗面试高频点）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

rng = np.random.default_rng(cfg.RANDOM_SEED)


def _fmt_ts(base_day: np.ndarray, hour: np.ndarray, minute: np.ndarray, second: np.ndarray) -> np.ndarray:
    """把 (天偏移, 时, 分, 秒) 数组拼成字符串时间戳数组。"""
    return np.array([_fmt_ts_scalar(d, h, m, s)
                    for d, h, m, s in zip(base_day, hour, minute, second)], dtype=object)


def _fmt_ts_scalar(day, hour, minute, second) -> str:
    """单条时间戳（标量版）。"""
    start = pd.Timestamp(cfg.START_DATE)
    d = start + pd.Timedelta(days=int(day), hours=int(hour), minutes=int(minute), seconds=int(second))
    return d.strftime("%Y-%m-%d %H:%M:%S")


def generate_users() -> pd.DataFrame:
    n = cfg.N_USERS
    users = pd.DataFrame({
        "user_id": [f"U{ i:05d}" for i in range(1, n + 1)],
        "city_tier": rng.choice(cfg.CITY_TIERS, n, p=[0.18, 0.22, 0.25, 0.22, 0.13]),
        "age_group": rng.choice(cfg.AGE_GROUPS, n, p=[0.45, 0.3, 0.18, 0.07]),
        "gender": rng.choice(cfg.GENDERS, n),
        "reg_channel": rng.choice(cfg.REG_CHANNELS, n, p=[0.4, 0.3, 0.2, 0.1]),
        "device": rng.choice(cfg.DEVICES, n),
    })
    # 活跃度：决定该用户产生多少行为（泊松分布，长尾）
    users["activity"] = rng.poisson(lam=40, size=n).clip(5, 200)
    # 隐式品类偏好（给推荐系统埋信号）：先选主业品类，再 Dirichlet 平滑
    primary = rng.integers(0, len(cfg.CATEGORIES), n)
    alpha = np.ones(len(cfg.CATEGORIES)) * 0.4
    pref = rng.dirichlet(alpha, n)
    pref[np.arange(n), primary] += rng.uniform(0.8, 1.8, n)
    pref = pref / pref.sum(axis=1, keepdims=True)
    # 保存偏好矩阵供内部使用（不写进维表，避免"剧透"给推荐系统）
    users["_pref"] = list(pref)
    # 机器人用户：几乎对所有内容都互动（异常行为，供数据质量识别）
    bot_idx = rng.choice(n, cfg.N_BOT_USERS, replace=False)
    users["is_bot"] = False
    users.loc[users.index[bot_idx], "is_bot"] = True
    return users


def generate_videos() -> pd.DataFrame:
    n = cfg.N_VIDEOS
    n_authors = 400
    cat = rng.choice(cfg.CATEGORIES, n)
    videos = pd.DataFrame({
        "video_id": [f"V{ i:05d}" for i in range(1, n + 1)],
        "author_id": [f"A{ i:04d}" for i in rng.integers(0, n_authors, n)],
        "category": cat,
        "duration_sec": rng.integers(8, 180, n),
        "publish_time": [_fmt_ts(rng.integers(0, cfg.N_DAYS, 1),
                                 rng.integers(0, 24, 1),
                                 rng.integers(0, 60, 1),
                                 rng.integers(0, 60, 1))[0] for _ in range(n)],
    })
    # 内容标签：从所属品类的标签池抽 2~3 个
    tags = []
    for c in cat:
        pool = cfg.TAGS_BY_CATEGORY[c]
        k = rng.integers(2, min(4, len(pool) + 1))
        tags.append("|".join(rng.choice(pool, k, replace=False)))
    videos["tags"] = tags
    return videos


def generate_interactions(users: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    cat_index = {c: i for i, c in enumerate(cfg.CATEGORIES)}
    v_cat = videos["category"].map(cat_index).to_numpy()
    pref_matrix = np.vstack(users["_pref"].to_numpy())
    is_bot = users["is_bot"].to_numpy()
    user_ids = users["user_id"].to_numpy()
    video_ids = videos["video_id"].to_numpy()
    n_videos = len(videos)

    rows = []  # (impression_id, user_id, video_id, event_type, event_time, watch_duration)
    imp_counter = 0
    fp = cfg.FUNNEL_PROB
    v_dur = videos["duration_sec"].to_numpy()

    for u in range(len(users)):
        uid = user_ids[u]
        pref_u = pref_matrix[u]
        mult = pref_u[v_cat] / pref_u.max()           # 0~1，偏好强度
        n_imp = int(users["activity"].iloc[u])
        if is_bot[u]:
            mult = np.full(n_videos, 0.95)            # 机器人：无差别高互动
            n_imp = max(n_imp, 80)

        # 视频抽样权重：偏好为主 + 少量随机探索
        w = 0.85 * mult + 0.15
        w = w / w.sum()
        picked = rng.choice(n_videos, size=n_imp, p=w)

        # 时间戳：随机落在 14 天窗口内，晚 18-23 点权重更高
        day_off = rng.integers(0, cfg.N_DAYS, n_imp)
        hour = rng.choice(24, n_imp, p=_hour_dist())
        minute = rng.integers(0, 60, n_imp)
        second = rng.integers(0, 60, n_imp)

        # 漏斗概率（按偏好调制）
        p_click = fp["click"] * (0.4 + 0.6 * mult[picked])
        p_play = np.full(n_imp, fp["play"])
        p_complete = fp["complete"] * (0.6 + 0.4 * mult[picked])
        p_like = fp["like"] * (0.5 + 0.5 * mult[picked])
        p_comment = np.full(n_imp, fp["comment"])
        p_share = np.full(n_imp, fp["share"])
        p_follow = np.full(n_imp, fp["follow"])
        if is_bot[u]:
            p_like = np.full(n_imp, 0.9); p_comment = np.full(n_imp, 0.5)
            p_share = np.full(n_imp, 0.4); p_follow = np.full(n_imp, 0.3)

        r = rng.random((7, n_imp))  # 7 个独立均匀随机数
        clicked = r[0] < p_click
        played = clicked & (r[1] < p_play)
        completed = played & (r[2] < p_complete)
        liked = completed & (r[3] < p_like)
        commented = completed & (r[4] < p_comment)
        shared = completed & (r[5] < p_share)
        followed = completed & (r[6] < p_follow)

        for i in range(n_imp):
            imp_counter += 1
            imp_id = f"IMP{imp_counter:08d}"
            vid = video_ids[picked[i]]
            dur = int(v_dur[picked[i]])
            base = (day_off[i], hour[i], minute[i], second[i])
            # 事件序列时间：曝光 -> 点击 -> 播放 -> 完播 -> 互动，依次向后偏移几秒
            t0 = _fmt_ts_scalar(*base)
            rows.append((imp_id, uid, vid, "impression", t0, None))
            if clicked[i]:
                t1 = _offset(base, 2)
                rows.append((imp_id, uid, vid, "click", t1, None))
            if played[i]:
                t2 = _offset(base, 5)
                wd_play = int(dur * float(rng.uniform(0.3, 0.8)))
                rows.append((imp_id, uid, vid, "play", t2, wd_play))
            if completed[i]:
                t3 = _offset(base, dur)
                rows.append((imp_id, uid, vid, "complete", t3, dur))
            if liked[i]:
                rows.append((imp_id, uid, vid, "like", _offset(base, 8), None))
            if commented[i]:
                rows.append((imp_id, uid, vid, "comment", _offset(base, 10), None))
            if shared[i]:
                rows.append((imp_id, uid, vid, "share", _offset(base, 12), None))
            if followed[i]:
                rows.append((imp_id, uid, vid, "follow", _offset(base, 14), None))

    log = pd.DataFrame(rows, columns=["impression_id", "user_id", "video_id", "event_type", "event_time", "watch_duration"])

    # ---------- 注入脏数据 ----------
    n = len(log)
    # 1) 重复事件
    dup_mask = rng.random(n) < cfg.DIRTY_DUP_RATIO
    log = pd.concat([log, log[dup_mask]], ignore_index=True)
    # 2) 空 user_id
    null_mask = rng.random(len(log)) < cfg.DIRTY_NULL_USER
    log.loc[null_mask, "user_id"] = ""
    # 3) 乱序 / 迟到时间戳（事件时间被推到窗口之外，模拟迟上报）
    late_mask = rng.random(len(log)) < cfg.DIRTY_LATE_RATIO
    late_ts = (_fmt_ts(np.full(late_mask.sum(), cfg.N_DAYS + 30),
                       rng.integers(0, 24, late_mask.sum()),
                       rng.integers(0, 60, late_mask.sum()),
                       rng.integers(0, 60, late_mask.sum())))
    log.loc[late_mask, "event_time"] = late_ts

    log = log.sample(frac=1, random_state=cfg.RANDOM_SEED).reset_index(drop=True)
    return log


def _hour_dist() -> np.ndarray:
    """一天 24 小时的活跃分布，晚高峰更密。"""
    base = np.ones(24)
    base[18:24] += 2.5
    base[12:14] += 1.0
    return base / base.sum()


def _offset(base, sec: int) -> str:
    start = pd.Timestamp(cfg.START_DATE)
    d = start + pd.Timedelta(days=int(base[0]), hours=int(base[1]),
                             minutes=int(base[2]), seconds=int(base[3]) + sec)
    return d.strftime("%Y-%m-%d %H:%M:%S")


def main():
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    users = generate_users()
    videos = generate_videos()
    log = generate_interactions(users, videos)

    # 维表去掉内部列
    users.drop(columns=["_pref"]).to_csv(cfg.RAW_DIR / "dim_user.csv", index=False, encoding="utf-8-sig")
    videos.to_csv(cfg.RAW_DIR / "dim_video.csv", index=False, encoding="utf-8-sig")
    log.to_csv(cfg.RAW_DIR / "user_action_log.csv", index=False, encoding="utf-8-sig")

    print(f"[data] users={len(users)} videos={len(videos)} events={len(log)}")
    print(f"[data] 脏数据: 重复+{int(len(log)*cfg.DIRTY_DUP_RATIO)} 空user~{int(len(log)*cfg.DIRTY_NULL_USER)} 迟到~{int(len(log)*cfg.DIRTY_LATE_RATIO)}")
    print(f"[data] saved -> {cfg.RAW_DIR}")


if __name__ == "__main__":
    main()
