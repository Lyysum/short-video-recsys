"""
Streamlit 可视化看板：业务漏斗 / 品类热度 / 热门视频 / 时段分布 / 用户留存 / 推荐 Demo。
用法：streamlit run app.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (list(_root.glob("config.py")) or _root == _root.parent):
    _root = _root.parent
sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st
import config as cfg
import db


@st.cache_resource
def get_conn():
    return db.get_conn()


@st.cache_data
def load(sql):
    conn = get_conn()
    return db.read_sql(conn, sql)


def main():
    st.set_page_config(page_title="短视频内容推荐系统", layout="wide")
    st.title("🎬 短视频内容推荐系统 · 数据看板")
    st.caption("抖音式推荐场景 · 数据开发 + 数据科学全链路 Demo")

    funnel = load("SELECT * FROM ads_funnel WHERE scope='ALL'").iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("曝光量", f"{int(funnel['exposures']):,}")
    col2.metric("点击率 CTR", f"{float(funnel['ctr'])*100:.1f}%")
    col3.metric("完播率", f"{float(funnel['complete_rate'])*100:.1f}%")
    col4.metric("点赞率", f"{float(funnel['like_rate'])*100:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["转化漏斗", "内容 & 热门", "时段 & 留存", "推荐 Demo"])

    with tab1:
        st.subheader("整体转化漏斗（曝光→点击→完播→点赞）")
        fdf = pd.DataFrame({
            "环节": ["曝光", "点击", "完播", "点赞"],
            "人数/次数": [int(funnel["exposures"]), int(funnel["clicks"]),
                       int(funnel["completes"]), int(funnel["likes"])],
        })
        st.bar_chart(fdf.set_index("环节"))
        st.subheader("分品类 CTR")
        cat = load("SELECT category, exposures, clicks, ctr FROM ads_category_stat ORDER BY ctr DESC")
        st.dataframe(cat, use_container_width=True)

    with tab2:
        st.subheader("热门视频 Top 10（按热度分）")
        hot = load("SELECT video_id, category, hot_score, ctr, n_click FROM ads_video_hot LIMIT 10")
        st.dataframe(hot, use_container_width=True)
        st.subheader("各品类内容量 vs 曝光")
        cat2 = load("SELECT category, exposures FROM ads_category_stat ORDER BY exposures DESC")
        st.bar_chart(cat2.set_index("category"))

    with tab3:
        st.subheader("全天时段曝光分布（识别晚高峰）")
        hourly = load("SELECT hour, exposures, ctr FROM ads_hourly ORDER BY hour")
        st.line_chart(hourly.set_index("hour")["exposures"])
        st.subheader("用户次留（按首日活跃群）")
        ret = load("SELECT cohort_date, cohort_active, d1_retention, d3_retention, d7_retention FROM ads_retention")
        st.dataframe(ret.head(10), use_container_width=True)

    with tab4:
        st.subheader("给指定用户做推荐（输入 user_id）")
        uid = st.text_input("输入 user_id（示例可在下方推荐结果表找）", value="")
        if uid:
            rec = load(f"SELECT rank, video_id, category, score, user_tags FROM ads_rec_result WHERE user_id='{uid}' ORDER BY rank")
            prof = load(f"SELECT top_categories, ctr, complete_rate FROM ads_user_profile WHERE user_id='{uid}'")
            if not rec.empty:
                tags = rec["user_tags"].iloc[0] if "user_tags" in rec else ""
                st.write(f"**用户画像标签**：{tags}")
                st.dataframe(rec, use_container_width=True)
            else:
                st.warning("该用户暂无推荐结果（可能正样本不足），换一个试试。")
        st.subheader("示例推荐结果（部分用户）")
        demo = load("SELECT user_id, rank, video_id, category, score FROM ads_rec_result ORDER BY user_id, rank LIMIT 60")
        st.dataframe(demo, use_container_width=True)


if __name__ == "__main__":
    main()
