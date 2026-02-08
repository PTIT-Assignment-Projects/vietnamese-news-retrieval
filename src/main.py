from src.dependencies.spark import SparkIRSystem
from src.preprocessing.mining_job import load_data


def main():
    spark = SparkIRSystem()
    load_data(spark, "../data/corpus.parquet")
    load_data(spark, "../data/train.parquet")
    load_data(spark, "../data/test.parquet")

if __name__ == "__main__":
    main()