"""
SQLite 数仓助手：连接、建表、写表、查询。
本项目用 SQLite 作为本地数仓，零配置、无需 Docker/Java，
但建模思想（分层 + 星型模型）与 Hive/Spark 数仓完全一致，可平滑迁移。
"""
import sqlite3
from pathlib import Path
import pandas as pd
import config as cfg


def get_conn() -> sqlite3.Connection:
    cfg.WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def write_table(conn, df: pd.DataFrame, name: str, if_exists: str = "replace"):
    """把 DataFrame 落为数仓表（默认逐行 executemany，避免 SQLite 变量数上限）。"""
    df.to_sql(name, conn, if_exists=if_exists, index=False)
    conn.commit()


def read_sql(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def table_rows(conn, name: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
