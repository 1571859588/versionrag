import os

# Use absolute path based on this file's location, so it works regardless of CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "..", "data", "db")
os.makedirs(_DATA_DIR, exist_ok=True)   # auto-create if missing

KNOWLEDGE_GRAPH_PATH = os.path.join(_DATA_DIR, "knowledge_graph_index.pkl")
MILVUS_DB_PATH = os.path.join(_DATA_DIR, "milvus.db")
MILVUS_COLLECTION_NAME_BASELINE = "baseline_collection"
MILVUS_COLLECTION_NAME_VERSIONRAG = "VersionRAG_collection"
MILVUS_MAX_TOKEN_COUNT = 512 # Maximum tokens per chunk
MILVUS_META_ATTRIBUTE_TEXT = "text"
MILVUS_META_ATTRIBUTE_PAGE = "page"
MILVUS_META_ATTRIBUTE_FILE = "file"
MILVUS_META_ATTRIBUTE_CATEGORY = "category"
MILVUS_META_ATTRIBUTE_DOCUMENTATION = "documentation"
MILVUS_META_ATTRIBUTE_VERSION = "version"
MILVUS_META_ATTRIBUTE_TYPE = "type" # file / node
MILVUS_BASELINE_SOURCE_COUNT = 15

# LLM mode: 'openai' | 'groq' | 'offline'
# Override via LLM_MODE env var (e.g. 'openai' to use local vLLM OpenAI-compatible API)
LLM_MODE = os.getenv('LLM_MODE', 'openai')
LLM_OFFLINE_MODEL = os.getenv('LLM_OFFLINE_MODEL', '')  # local lmstudio model name

# Embedding model & dimensions — configurable for local vLLM embedding service
EMBEDDING_MODEL = os.getenv('VERSIONRAG_EMBEDDING_MODEL', 'text-embedding-3-small')
EMBEDDING_DIMENSIONS = int(os.getenv('VERSIONRAG_EMBEDDING_DIM', '512'))

BASELINE_MODEL = "Baseline"
KG_MODEL = "GraphRAG"
VERSIONRAG_MODEL = "VersionRAG"
AVAILABLE_MODELS = [BASELINE_MODEL, KG_MODEL, VERSIONRAG_MODEL]
