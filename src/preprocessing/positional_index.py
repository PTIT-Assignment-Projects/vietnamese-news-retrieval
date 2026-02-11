import math
from collections import defaultdict

from src.dependencies.constant import CID_COLUMN, TEXT_COLUMN, CORPUS_PATH, UTIL_DIR, DOCUMENT_CORPUS_PICKLE_PATH, \
    TERM_DOC_MATRIX_PICKLE_PATH, TF_MATRIX_PICKLE_PATH, IDF_MATRIX_PICKLE_PATH, \
    TF_IDF_MATRIX_PICKLE_PATH, DOC_LENGTH_PICKLE_PATH, NORMALIZED_TFIDF_PICKLE_PATH, L2_PAIR_PICKLE_PATH
from src.dependencies.spark import SparkIRSystem
from src.preprocessing.data_loader import load_data
from src.preprocessing.preprocessing import process_text
from src.util.pickle_handling import load_pickle_file, save_to_pickle_file


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
        self.pair_distance = {}

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
        save_to_pickle_file(self.documents, DOCUMENT_CORPUS_PICKLE_PATH)
        print(f"Loaded {df.count()} documents.")

    def load_preprocessed_corpus(self) -> None:
        self.documents = load_pickle_file(DOCUMENT_CORPUS_PICKLE_PATH)
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
        save_to_pickle_file(self.term_doc_matrix, TERM_DOC_MATRIX_PICKLE_PATH)
        return self.term_doc_matrix
    def load_built_index(self):
        self.term_doc_matrix = load_pickle_file(TERM_DOC_MATRIX_PICKLE_PATH)
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
        save_to_pickle_file(self.tf_matrix, TF_MATRIX_PICKLE_PATH)
        return self.tf_matrix
    def load_computed_tf(self):
        self.tf_matrix = load_pickle_file(TF_MATRIX_PICKLE_PATH)

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
        save_to_pickle_file(self.idf_matrix, IDF_MATRIX_PICKLE_PATH)
        return self.idf_matrix

    def load_computed_idf(self):
        self.idf_matrix = load_pickle_file(IDF_MATRIX_PICKLE_PATH)

    def compute_tfidf(self):
        print("Start compute TF-IDF")
        if not self.tf_matrix:
            self.load_computed_tf()
            if not self.tf_matrix:
                self.compute_tf()
        if not self.idf_matrix:
            self.load_computed_idf()
            if not self.idf_matrix:
                self.compute_idf()

        tf_matrix_bc = self.context.broadcast(self.tf_matrix)
        idf_matrix_bc = self.context.broadcast(self.idf_matrix)

        terms_rdd = self.context.parallelize(list(self.vocabulary))

        def compute_term_tfidf(term):
            result = {}
            tf_matrix = tf_matrix_bc.value
            idf_matrix = idf_matrix_bc.value
            
            if term in tf_matrix:
                tf_dict = tf_matrix[term]
                idf_value = idf_matrix.get(term, 0)
                # Only iterate over documents that actually contain this term
                for doc_id, tf_val in tf_dict.items():
                    result[doc_id] = tf_val * idf_value
            return term, result # word : {doc_id: tfidf}

        tfidf_results = terms_rdd.map(compute_term_tfidf).collect()
        self.tf_idf_matrix = dict(tfidf_results)
        save_to_pickle_file(self.tf_idf_matrix, TF_IDF_MATRIX_PICKLE_PATH)
        return self.tf_idf_matrix
    def load_computed_tf_idf(self):
        self.tf_idf_matrix = load_pickle_file(TF_IDF_MATRIX_PICKLE_PATH)

    def compute_doc_lengths(self):
        print("Computing document lengths (L2 norm)...")
        if not self.tf_idf_matrix:
            self.load_computed_tf_idf()
            if not self.tf_idf_matrix:
                self.compute_tfidf()

        # Convert matrix items to RDD: RDD[(term, {doc_id: weight})]
        # Use short-circuit: if we already have it in memory, skip expensive RDD creation if not needed
        # but here we use Spark as requested.
        tfidf_rdd = self.context.parallelize(list(self.tf_idf_matrix.items()))
        
        # Calculate sum of squares per document: RDD[(doc_id, weight^2)] -> aggregate by doc_id
        doc_sq_sums = tfidf_rdd.flatMap(lambda x: [(doc_id, weight**2) for doc_id, weight in x[1].items()]) \
                               .reduceByKey(lambda a, b: a + b) \
                               .mapValues(math.sqrt) \
                               .collectAsMap()
        
        self.doc_lengths = dict(doc_sq_sums)
        print(f"Computed lengths for {len(self.doc_lengths)} documents.")
        save_to_pickle_file(self.doc_lengths, DOC_LENGTH_PICKLE_PATH)
        return self.doc_lengths
    def load_computed_doc_lengths(self):
        self.doc_lengths = load_pickle_file(DOC_LENGTH_PICKLE_PATH)

    def compute_normalize_tfidf(self):
        print("Starting compute Normalize TF-IDF...")
        if not self.tf_idf_matrix:
            self.load_computed_tf_idf()
            if not self.tf_idf_matrix:
                self.compute_tfidf()
        if not self.doc_lengths:
            self.load_computed_doc_lengths()
            if not self.doc_lengths:
                self.compute_doc_lengths()

        # Optimize: instead of broadcasting matrix, parallelize it
        tfidf_rdd = self.context.parallelize(list(self.tf_idf_matrix.items()))
        lengths_bc = self.context.broadcast(self.doc_lengths)

        def normalize_term_optimized(item):
            term, doc_weights = item
            lengths = lengths_bc.value
            return term, {doc_id: weight / lengths.get(doc_id, 1.0) for doc_id, weight in doc_weights.items()}

        results = tfidf_rdd.map(normalize_term_optimized).collect()
        self.normalized_tfidf = dict(results)
        print("Normalization complete.")
        save_to_pickle_file(self.normalized_tfidf, NORMALIZED_TFIDF_PICKLE_PATH)
        return self.normalized_tfidf

    def load_computed_normalized_tfidf(self):
        self.normalized_tfidf = load_pickle_file(NORMALIZED_TFIDF_PICKLE_PATH)


    def retrieve(self, query_str, k=10):
        print(f"Ranking documents for query: '{query_str}'")
        if not self.tf_idf_matrix: self.load_computed_tf_idf()
        if not self.doc_lengths: self.load_computed_doc_lengths()

        # 1. Query Processing
        processed_query = process_text(query_str)
        print(f"  [STEP 2] Preprocessed Query: '{processed_query}'")
        
        query_tokens = processed_query.split()
        tf_q = defaultdict(int)
        for t in query_tokens:
            if t in self.vocabulary: tf_q[t] += 1
        
        matched_tokens = list(tf_q.keys())
        print(f"  [STEP 3] Matched Vocabulary Tokens: {matched_tokens}")
        
        if not tf_q: 
            print("  [INFO] No matched tokens found in vocabulary.")
            return []

        q_vec = {}
        q_len_sq = 0
        for term, count in tf_q.items():
            weight = (1 + math.log10(count)) * self.idf_matrix.get(term, 0)
            q_vec[term] = weight
            q_len_sq += weight ** 2
        q_len = math.sqrt(q_len_sq)
        print(f"  [STEP 4] Query Vector L2 Norm: {q_len:.6f}")

        # 2. Parallel Scoring
        relevant_terms = matched_tokens
        relevant_data = [(t, self.tf_idf_matrix[t]) for t in relevant_terms if t in self.tf_idf_matrix]
        
        if not relevant_data: return []
        
        q_vec_bc = self.context.broadcast(q_vec)
        q_len_bc = self.context.broadcast(q_len)
        lengths_bc = self.context.broadcast(self.doc_lengths)

        scores = self.context.parallelize(relevant_data) \
            .flatMap(lambda x: [(doc_id, x[1][doc_id] * q_vec_bc.value[x[0]]) for doc_id in x[1]]) \
            .reduceByKey(lambda a, b: a + b) \
            .map(lambda x: (x[0], x[1] / (q_len_bc.value * lengths_bc.value.get(x[0], 1.0)))) \
            .takeOrdered(k, key=lambda x: -x[1])

        return scores

