
from src.dependencies.constant import CID_COLUMN, CONTEXT_LIST_COLUMN, TEXT_COLUMN, CORPUS_PATH
from src.dependencies.spark import SparkIRSystem
from src.preprocessing.data_loader import load_data
from src.preprocessing.preprocessing import tokenize_words


class SparkInvertedIndexIR(SparkIRSystem):
    def __init__(self):
        super().__init__()
        self.documents = {}
        self.vocabulary = {}
        self.path = CORPUS_PATH
        self.doc_count = 10
        self.term_doc_matrix = {} # {term: [d1, d2, ...]}
        self.tf_matrix = {}
        self.idf_matrix = {}
        self.tf_idf_matrix = {}
        self.doc_lengths = {} # {doc: length}
        self.normalized_tfidf = {} # {term: {doc: value}}

    def load_corpus(self) -> None:
        from pyspark.sql.functions import pandas_udf
        from src.preprocessing.preprocessing import tokenize_batch_logic

        # Register the UDF lazily here
        # This guarantees a Session exists because __init__ was already called
        tokenize_batch_udf = pandas_udf(tokenize_batch_logic, "string")
        
        df = load_data(self, self.path)
        
        # Apply vectorized tokenization (Pandas UDF)
        # This processes data in large chunks, significantly reducing IPC overhead
        processed_df = df.select(
            CID_COLUMN, 
            tokenize_batch_udf(df[TEXT_COLUMN]).alias("tokens")
        )
        
        # Collect as a map. 
        # For large datasets, self.documents should ideally remain a DataFrame/RDD
        # but for the current architecture, we collect it back to the driver.
        self.documents = processed_df.rdd.map(lambda x: (x[0], x[1])).collectAsMap()
        
        print(f"Loaded {len(self.documents)} documents.")

