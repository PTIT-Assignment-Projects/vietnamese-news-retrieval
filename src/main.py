from src.preprocessing.positional_index import SparkInvertedIndexIR


def main():
    spark = SparkInvertedIndexIR()
    # spark.load_corpus()
    spark.load_preprocessed_corpus()
    # spark.build_index()
    spark.load_built_index()

if __name__ == "__main__":
    main()