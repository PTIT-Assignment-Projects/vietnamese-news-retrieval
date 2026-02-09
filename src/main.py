from src.preprocessing.inverted_index import SparkInvertedIndexIR


def main():
    spark = SparkInvertedIndexIR()
    spark.load_corpus()

if __name__ == "__main__":
    main()