import os
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

def save_data(df: DataFrame, file_path: Path | str) -> None:
    from src.dependencies.constant import OVERWRITE_WRITE_MODE
    df.write.mode(OVERWRITE_WRITE_MODE).parquet(str(file_path))

def reduce_dataset(spark: SparkIRSystem, limit: int = 10000) -> None:
    from src.dependencies.constant import CORPUS_PATH, TRAIN_PATH, TEST_PATH, CID_COLUMN
    from pyspark.sql.functions import expr
    import shutil
    
    print(f"Limiting corpus to {limit} rows...")
    corpus_df = load_data(spark, CORPUS_PATH)
    corpus_reduced = corpus_df.limit(limit)
    
    # Save to temp and move back to avoid read/write conflicts
    temp_corpus = str(CORPUS_PATH) + "_temp"
    save_data(corpus_reduced, temp_corpus)
    
    print(f"Filtering train/test data (CID > {limit})...")
    for path in [TRAIN_PATH, TEST_PATH]:
        df = load_data(spark, path)
        # Handle cid as an ARRAY<BIGINT>
        df_reduced = df.withColumn(CID_COLUMN, expr(f"filter({CID_COLUMN}, x -> x <= {limit})")) \
                       .filter(expr(f"size({CID_COLUMN}) > 0"))
        temp_path = str(path) + "_temp"
        save_data(df_reduced, temp_path)
    
    # Clean up and replace
    def safe_remove(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    for path in [CORPUS_PATH, TRAIN_PATH, TEST_PATH]:
        if os.path.exists(path):
            safe_remove(path)
        shutil.move(str(path) + "_temp", path)
        
    print("Dataset reduction complete.")

vietnamese_stopwords = load_stopwords(str(STOPWORDS_DICT_PATH))