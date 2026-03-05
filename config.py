import os

# OpenAI API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-czbxukovtswhtcqobycfwjcwwqfrwcvmssejhrtvcfghrdvu")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3.2")

# 输入数据配置
INPUT_FILE = "kuairec_caption_category.csv"
TITLE_COLUMN = "caption"

# 输出配置
OUTPUT_DIR = "output"
OUTPUT_FILE = "events.json"

# 关键词缓存文件（以标题为 key，避免重复调用 API）
KEYWORDS_CACHE_FILE = "output/keywords_cache.json"

# 向量化配置
SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 聚类配置
CLUSTERING_METHOD = "kmeans"   # "kmeans" 或 "dbscan"
N_CLUSTERS = None              # None 表示自动确定聚类数量
MIN_CLUSTER_SIZE = 15           # DBSCAN 最小簇大小
DBSCAN_EPS = 0.15              # None 表示自动用 k 距离图肘部法则确定 eps；也可手动指定如 0.15

# KMeans 自动确定聚类数量的范围
MIN_K = 100
MAX_K = 800
STEP_K = 100

# 关键词提取批次大小（每批发送给 OpenAI 的标题数量）
KEYWORD_BATCH_SIZE = 20

# 事件归纳时每个簇最多取多少标题用于总结
MAX_TITLES_PER_CLUSTER = 15

# 簇至少需要多少条标题才会进行事件归纳（不足则跳过）
MIN_TITLES_FOR_EVENT = 4

# 按轮廓系数排序后，最多归纳多少个事件
MAX_EVENTS_TO_SUMMARIZE = 30
