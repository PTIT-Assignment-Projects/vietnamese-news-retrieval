from pathlib import Path

APP_NAME = "vietnamese_news_ir_system"
CONFIG_PATH = "configs"
CONFIG_FILE_NAME = "sample_config.json"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
CORPUS_PATH = DATA_DIR / "corpus.parquet"

CORPUS_JSON_PATH = DATA_DIR / "corpus.json"
TEST_QUERIES_JSON_PATH = DATA_DIR / "test_queries.json"

# COLUMNS
# TRAIN_DATA
QUESTION_COLUMN = "question"
CONTEXT_LIST_COLUMN = "context_list"
QID_COLUMN = "qid"
CID_COLUMN = "cid"

# CORPUS_DATA
TEXT_COLUMN = "text"


# SPARK
OVERWRITE_WRITE_MODE = "overwrite"