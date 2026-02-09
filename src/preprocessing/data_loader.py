from pathlib import Path

from pyspark.sql import DataFrame

from src.dependencies.constant import STOPWORDS_DICT_PATH
from src.dependencies.spark import SparkIRSystem
def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return {
            line.strip().replace(" ", "_").lower()
            for line in f
            if line.strip()
        }

def load_data(spark: SparkIRSystem, file_path: Path | str = None) -> DataFrame:
    path = str(file_path) if file_path else None
    df = spark.spark_sess.read.parquet(path)
    return df
vietnamese_stopwords = load_stopwords(str(STOPWORDS_DICT_PATH))