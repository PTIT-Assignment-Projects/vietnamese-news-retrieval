from src.dependencies.spark import SparkIRSystem
from src.dependencies.constant import CORPUS_PATH, TRAIN_PATH, TEST_PATH, CID_COLUMN, OVERWRITE_WRITE_MODE
from src.preprocessing.data_loader import load_data
import os

def reduce_data():
    spark = SparkIRSystem()
    
    print("Reducing corpus...")
    # 1. Load and limit corpus
    corpus_df = load_data(spark, CORPUS_PATH)
    # If the user wants cid <= 10000 specifically:
    corpus_reduced = corpus_df.filter(corpus_df[CID_COLUMN] <= 15000)
    # But they said "first 10000 data", so let's use limit.
    # corpus_reduced = corpus_df.limit(10000)

    # Save back corpus
    # Note: Spark can't easily overwrite the same path it's reading from in some versions
    # without caching or writing to a temp location first.
    temp_corpus_path =  "_temp" + str(CORPUS_PATH)
    corpus_reduced.write.mode(OVERWRITE_WRITE_MODE).parquet(temp_corpus_path)
    
    print("Reducing train data...")
    # 2. Load and filter train data
    from pyspark.sql.functions import expr
    train_df = load_data(spark, TRAIN_PATH)

    train_reduced = train_df.withColumn(CID_COLUMN, expr(f"filter({CID_COLUMN}, x -> x <= 15000)")) \
                            .filter(expr(f"size({CID_COLUMN}) > 0"))
    
    temp_train_path =  "_temp" + str(TRAIN_PATH)
    train_reduced.write.mode(OVERWRITE_WRITE_MODE).parquet(temp_train_path)
    
    print("Reducing test data...")
    # 3. Load and filter test data
    test_df = load_data(spark, TEST_PATH)
    test_reduced = test_df.withColumn(CID_COLUMN, expr(f"filter({CID_COLUMN}, x -> x <= 10000)")) \
                           .filter(expr(f"size({CID_COLUMN}) > 0"))
    
    temp_test_path =  "_temp" + str(TEST_PATH)
    test_reduced.write.mode(OVERWRITE_WRITE_MODE).parquet(temp_test_path)
    
    spark.stop_session()

    # Move temp files to original locations
    import shutil
    def safe_remove(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    for path, temp in [
        (CORPUS_PATH, temp_corpus_path),
        (TRAIN_PATH, temp_train_path),
        (TEST_PATH, temp_test_path)
    ]:
        if os.path.exists(path):
            safe_remove(path)
        shutil.move(temp, path)
    
    print("Done! Data reduced to first 10,000 documents and filtered train/test sets.")

if __name__ == "__main__":
    reduce_data()
