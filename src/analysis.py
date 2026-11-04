"""
SQL 多维分析：转化漏斗、品类热度、用户留存、时段分布。
结果写回数仓 ADS 层（ads_retention / ads_hourly），同时打印分析报告，供看板与简历佐证使用。
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve()
while not (list(_root.glob("config.py")) or _root == _root.parent):
    _root = _root.parent
sys.path.insert(0, str(_root))

import config as cfg  # noqa: E402
import db  # noqa: E402
import pandas as pd  # noqa: E402


def compute_retention(conn) -> pd.DataFrame:
    """按首次活跃日分群，计算 D1/D3/D7 次留。"""
    ud = db.read_sql(conn, "SELECT user_id, imp_date, exposures FROM dws_user_daily WHERE exposures > 0")
    active = ud[["user_id", "imp_date"]].drop_duplicates()
    dates = sorted(active["imp_date"].unique())
    by_date = {d: set(g["user_id"]) for d, g in active.groupby("imp_date")}
    rows = []
    for i, d in enumerate(dates):
        today = by_date[d]
        sz = len(today)
        row = {"cohort_date": d, "cohort_active": sz}
        for k in (1, 3, 7):
            if i + k < len(dates):
                nxt = by_date[dates[i + k]]
                row[f"d{k}_retention"] = round(len(today & nxt) / sz, 4) if sz else 0
            else:
                row[f"d{k}_retention"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def compute_hourly(conn) -> pd.DataFrame:
    """全天各小时曝光量与 CTR（识别晚高峰）。"""
    exp = db.read_sql(conn, "SELECT imp_time, is_click FROM dwd_exposure")
    exp["hour"] = pd.to_datetime(exp["imp_time"]).dt.hour
    g = exp.groupby("hour").agg(exposures=("imp_time", "count"),
                                clicks=("is_click", "sum")).reset_index()
    g["ctr"] = (g["clicks"] / g["exposures"]).round(4)
    return g


def run_analysis(verbose: bool = True):
    conn = db.get_conn()
    funnel = db.read_sql(conn, "SELECT * FROM ads_funnel")
    category = db.read_sql(conn, "SELECT * FROM ads_category_stat")
    retention = compute_retention(conn)
    hourly = compute_hourly(conn)
    db.write_table(conn, retention, "ads_retention")
    db.write_table(conn, hourly, "ads_hourly")
    conn.close()
    if verbose:
        all_row = funnel[funnel["scope"] == "ALL"].iloc[0]
        print(f"[analysis] 整体漏斗: 曝光={all_row['exposures']} 点击={all_row['clicks']} "
              f"CTR={all_row['ctr']} 完播={all_row['completes']} 完播率={all_row['complete_rate']} "
              f"点赞={all_row['likes']} 点赞率={all_row['like_rate']}")
        print(f"[analysis] 留存(首日群): D1={retention['d1_retention'].iloc[0]} "
              f"D3={retention['d3_retention'].iloc[0]} D7={retention['d7_retention'].iloc[0]}")
        print(f"[analysis] 晚高峰时段(CTR最高前3):")
        print(hourly.sort_values('ctr', ascending=False).head(3).to_string(index=False))
    return {"funnel": funnel, "category": category, "retention": retention, "hourly": hourly}


if __name__ == "__main__":
    run_analysis()
