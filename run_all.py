"""
一键运行入口：数据生成 -> 数仓ETL -> 多维分析 -> 推荐引擎。
用法：python run_all.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (list(_root.glob("config.py")) or _root == _root.parent):
    _root = _root.parent
sys.path.insert(0, str(_root))

import config as cfg  # noqa: E402
import src.data_generator as dg  # noqa: E402
import src.etl as etl  # noqa: E402
import src.analysis as analysis  # noqa: E402
import src.recsys.pipeline as recsys  # noqa: E402


def main():
    print("=== 1/4 数据生成 ===")
    dg.main()
    print("\n=== 2/4 数仓 ETL ===")
    etl.run_etl()
    print("\n=== 3/4 多维分析 ===")
    analysis.run_analysis()
    print("\n=== 4/4 推荐引擎 ===")
    recsys.run_recsys()
    print("\n✅ 全流程完成。运行 `streamlit run app.py` 启动可视化看板。")


if __name__ == "__main__":
    main()
