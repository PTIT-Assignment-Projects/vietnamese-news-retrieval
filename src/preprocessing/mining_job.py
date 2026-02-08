from pathlib import Path

from pyspark.sql import DataFrame

from src.dependencies.constant import TRAIN_PATH
from src.dependencies.spark import SparkIRSystem


def load_data(spark: SparkIRSystem, file_path: Path | str = None) -> DataFrame:
    path = str(file_path) if file_path else None
    df = spark.spark_sess.read.parquet(path)
    return df