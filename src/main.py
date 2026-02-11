from src.preprocessing.positional_index import SparkInvertedIndexIR


def main():
    spark = SparkInvertedIndexIR()
    # spark.load_corpus()
    spark.load_preprocessed_corpus()
    # spark.build_index()
    spark.load_built_index()
    # spark.compute_tf()
    spark.load_computed_tf()
    # spark.compute_idf()
    spark.load_computed_idf()
    # spark.compute_tfidf()
    spark.load_computed_tf_idf()
if __name__ == "__main__":
    main()