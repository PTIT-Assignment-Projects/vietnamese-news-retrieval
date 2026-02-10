import math
import pickle
from collections import defaultdict

from src.dependencies.constant import CID_COLUMN, TEXT_COLUMN, CORPUS_PATH, UTIL_DIR, TERM_DOC_MATRIX_PICKLE_FILE, \
    TF_MATRIX_PICKLE_FILE, IDF_MATRIX_PICKLE_FILE
from src.dependencies.spark import SparkIRSystem
from src.preprocessing.data_loader import load_data


class SparkInvertedIndexIR(SparkIRSystem):
    def __init__(self):
        super().__init__()
        self.documents = {}
        self.vocabulary = set()
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
        self.doc_count = len(self.documents)
        from src.dependencies.constant import UTIL_DIR
        pickle_path = UTIL_DIR / "document_corpus.pkl"
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pickle_path, "wb") as file:
            pickle.dump(self.documents, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Loaded {df.count()} documents.")

    def load_preprocessed_corpus(self) -> None:
        pickle_path = UTIL_DIR / "document_corpus.pkl"
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pickle_path, "rb") as file:
            self.documents = pickle.load(file)
        self.doc_count = len(self.documents)


    def build_index(self):
        print("Building positional index...")
        docs_rdd = self.context.parallelize(list(self.documents.items()))
        def tokenize_doc(item):
            doc_id, content = item
            tokens = content.lower().split()
            result = []
            for position, token in enumerate(tokens):
                result.append((token, doc_id, position))
            return result
        # flatMap to create all (term, doc, pos) tuples
        positional_index_rdd = docs_rdd.flatMap(tokenize_doc)

        # group by term: {term: [(doc,pos),...]}
        grouped_by_term = positional_index_rdd.map(
            lambda x: (x[0], (x[1], x[2]))
        ).groupByKey()\
            .map(lambda x: (x[0], list(x[1])))

        # build vocabulary and position map
        index_data = grouped_by_term.collect()
        for term, positions in index_data:
            self.vocabulary.add(term)
            # group positions by document
            doc_positions = defaultdict(list)
            for doc, pos in positions:
                doc_positions[doc].append(pos)
            self.term_doc_matrix[term] = dict(doc_positions)
        print(f"Vocabulary size: {len(self.vocabulary)}")
        pickle_path = UTIL_DIR / TERM_DOC_MATRIX_PICKLE_FILE
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pickle_path, "wb") as file:
            pickle.dump(self.term_doc_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
        return self.term_doc_matrix
    def load_built_index(self):
        pickle_path = UTIL_DIR / TERM_DOC_MATRIX_PICKLE_FILE
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pickle_path, "rb") as file:
            self.term_doc_matrix = pickle.load(file)
            self.vocabulary = set(self.term_doc_matrix.keys())
    def compute_tf(self):
        print("Start compute TF")
        terms = list(self.vocabulary)
        if not terms:
            print("Empty vocabulary. Build the index first")
            return {}
        # Broadcast the matrix to avoid shipping it with every task closure
        matrix_bc = self.context.broadcast(self.term_doc_matrix)

        def compute_term_tf(term):
            # Access matrix from broadcast variable
            matrix = matrix_bc.value
            term_tf = {}
            if term in matrix:
                for doc_id, positions in matrix[term].items():
                    term_tf[doc_id] = len(positions)
            return term, term_tf

        terms_rdd = self.context.parallelize(terms)
        tf_results = terms_rdd.map(compute_term_tf).collect() # list of output - a tuple : (term, term_df)
        self.tf_matrix = dict(tf_results) # key: term, value: term_df
        pickle_path = UTIL_DIR / TF_MATRIX_PICKLE_FILE
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as file:
            pickle.dump(self.tf_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
        return self.tf_matrix
    def load_computed_tf(self):
        pickle_path = UTIL_DIR / TF_MATRIX_PICKLE_FILE
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "rb") as file:
            self.tf_matrix = pickle.load(file)

    def compute_idf(self):
        print("Start compute idf")
        if not self.tf_matrix:
            self.compute_tf()

        N = self.doc_count
        # Broadcast tf_matrix to avoid shipping it with every task closure
        tf_matrix_bc = self.context.broadcast(self.tf_matrix)

        terms_rdd = self.context.parallelize(list(self.vocabulary))
        # map
        def compute_term_idf(term):
            # Access tf_matrix from broadcast variable and use local N
            tf_matrix = tf_matrix_bc.value
            idf_value = 0.0
            if term in tf_matrix:
                df = len(tf_matrix[term])
                idf_value = math.log10((1 + N) / (1 + df)) + 1
            return term, idf_value
        # intermediate
        idf_results = terms_rdd.map(compute_term_idf).collect()
        self.idf_matrix = dict(idf_results)  # key: term, value: idf_value
        pickle_path = UTIL_DIR / IDF_MATRIX_PICKLE_FILE
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as file:
            pickle.dump(self.idf_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
        return self.idf_matrix

    def load_computed_idf(self):
        pickle_path = UTIL_DIR / IDF_MATRIX_PICKLE_FILE
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "rb") as file:
            self.idf_matrix = pickle.load(file)

