"""
全局配置：数据规模、维度取值、路径、模型参数。
所有模块统一从这里读取，方便一键调参复现。
"""
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent            # 项目根目录（short-video-recsys/）
RAW_DIR = ROOT / "data" / "raw"                  # 生成的原始数据（CSV）
WAREHOUSE_DIR = ROOT / "data" / "warehouse"      # 数仓（SQLite）
DB_PATH = WAREHOUSE_DIR / "recsys.db"            # 数仓文件

# ---------- 随机种子（保证可复现） ----------
RANDOM_SEED = 2026

# ---------- 数据规模 ----------
N_USERS = 5000          # 用户数
N_VIDEOS = 2000         # 视频数
N_DAYS = 14             # 模拟天数
START_DATE = "2026-10-20"  # 模拟起始日期
N_BOT_USERS = 25        # 注入的"机器人"用户数（异常行为，用于检验数据质量）

# ---------- 维度取值 ----------
CATEGORIES = ["搞笑", "美食", "游戏", "舞蹈", "知识", "剧情",
             "音乐", "宠物", "运动", "时政", "美妆", "旅行"]
# 每个品类对应的内容标签池（用于内容推荐 / 用户画像）
TAGS_BY_CATEGORY = {
    "搞笑":   ["脱口秀", "恶搞", "段子", "反转"],
    "美食":   ["探店", "教程", "家常菜", "测评"],
    "游戏":   ["手游", "攻略", "电竞", "主机"],
    "舞蹈":   ["街舞", "国风", "翻跳", "编舞"],
    "知识":   ["科普", "职场", "财经", "学习"],
    "剧情":   ["微短剧", "情感", "悬疑", "正能量"],
    "音乐":   ["翻唱", "原创", "乐器", "rap"],
    "宠物":   ["萌宠", "猫咪", "狗狗", "救助"],
    "运动":   ["健身", "篮球", "跑步", "户外"],
    "时政":   ["热点", "评论", "民生", "国际"],
    "美妆":   ["护肤", "妆教", "测评", "穿搭"],
    "旅行":   ["Vlog", "攻略", "风景", "穷游"],
}
CITY_TIERS = ["一线", "新一线", "二线", "三线", "其他"]
AGE_GROUPS = ["18-24", "25-30", "31-40", "40+"]
GENDERS = ["男", "女"]
REG_CHANNELS = ["应用商店", "短视频广告", "社交分享", "预装"]
DEVICES = ["iOS", "Android"]

# ---------- 行为事件 ----------
# 漏斗顺序：曝光(impression) -> 点击(click) -> 播放(play) -> 完播(complete) -> 互动(like/comment/share/follow)
EVENT_TYPES = ["impression", "click", "play", "complete", "like", "comment", "share", "follow"]
# 各事件的转化概率（条件概率，体现真实漏斗衰减）
FUNNEL_PROB = {
    "click":    0.32,   # 曝光->点击 CTR
    "play":     0.85,   # 点击->播放
    "complete": 0.55,   # 播放->完播
    "like":     0.18,   # 完播->点赞
    "comment":  0.05,   # 完播->评论
    "share":    0.04,   # 完播->转发
    "follow":   0.03,   # 完播->关注作者
}
# 正反馈信号（用于推荐系统标签为正样本）
POSITIVE_EVENTS = {"like", "comment", "share", "follow", "complete"}

# ---------- 数据质量"故意注入"的脏数据比例 ----------
DIRTY_DUP_RATIO = 0.01      # 重复事件
DIRTY_NULL_USER = 0.005     # 空 user_id
DIRTY_LATE_RATIO = 0.02     # 乱序/迟到时间戳

# ---------- 推荐系统参数 ----------
REC_K = 10                  # 每个用户推荐条数
RECALL_CANDIDATES = 100      # 召回阶段候选集大小
TEST_RATIO = 0.2            # 评估切分比例
RANK_MODEL = "gbdt"         # 排序模型：logistic / gbdt
