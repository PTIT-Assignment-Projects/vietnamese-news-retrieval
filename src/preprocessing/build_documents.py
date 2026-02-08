from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from src.dependencies.constant import QID_COLUMN, OVERWRITE_WRITE_MODE, QUESTION_COLUMN


def build_documents_json(file_path: Path, df: DataFrame) -> None:
    path = str(file_path)
    df.write.mode(OVERWRITE_WRITE_MODE).json(path)

def build_test_queries(file_path: Path, df: DataFrame) -> None:
    df2 = df.select(col(QID_COLUMN), col(QUESTION_COLUMN))
    df2.write.mode(OVERWRITE_WRITE_MODE).json(str(file_path))