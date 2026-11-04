"""
用户画像层：把"品类偏好"翻译成可读的兴趣标签。
（底层偏好分来自数仓 dws_user_category_pref，本模块只做标签化，便于看板展示与简历讲解。）
"""
import pandas as pd

LABEL_MAP = {
    "美食": "美食爱好者", "游戏": "硬核游戏党", "搞笑": "欢乐吃瓜人", "知识": "知识青年",
    "舞蹈": "舞蹈控", "音乐": "音乐迷", "宠物": "铲屎官", "运动": "运动达人",
    "时政": "时事观察员", "美妆": "美妆星人", "旅行": "旅行家", "剧情": "剧迷",
}


def user_profile_labels(top_categories_str: str):
    cats = [c for c in str(top_categories_str).split(",") if c]
    return [LABEL_MAP.get(c, f"{c}爱好者") for c in cats[:3]]
