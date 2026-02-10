from src.preprocessing.inverted_index import SparkInvertedIndexIR


def main():
    spark = SparkInvertedIndexIR()
    # spark.load_corpus()
    
    # Load and print the pickle file for verification
    import pickle
    from src.dependencies.constant import UTIL_DIR

    pickle_path = UTIL_DIR / "document_corpus.pkl"
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
            print(f"\nSuccessfully loaded pickle file from {pickle_path}")
            print(f"Total documents in pickle: {len(data)}")
            print("First 5 entries:")
            for cid, tokens in list(data.items())[:5]:
                print(f"CID {cid}: {str(tokens)[:100]}...")
    else:
        print(f"Pickle file not found at {pickle_path}")

    # Print length of train and test data
    from src.dependencies.constant import TRAIN_PATH, TEST_PATH
    train_df = spark.spark_sess.read.parquet(str(TRAIN_PATH))
    test_df = spark.spark_sess.read.parquet(str(TEST_PATH))
    print(f"\nTrain data length: {train_df.count()}")
    print(f"Test data length: {test_df.count()}")

if __name__ == "__main__":
    main()