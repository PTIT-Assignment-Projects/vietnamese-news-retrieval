import math
from collections import defaultdict

from src.dependencies.constant import CID_COLUMN, TEXT_COLUMN, CORPUS_PATH, UTIL_DIR, DOCUMENT_CORPUS_PICKLE_PATH, \
    TERM_DOC_MATRIX_PICKLE_PATH, TF_MATRIX_PICKLE_PATH, IDF_MATRIX_PICKLE_PATH, \
    TF_IDF_MATRIX_PICKLE_PATH, DOC_LENGTH_PICKLE_PATH, NORMALIZED_TFIDF_PICKLE_PATH, L2_PAIR_PICKLE_PATH, BM25_STATS_PICKLE_PATH
from src.dependencies.spark import SparkIRSystem
from src.preprocessing.data_loader import load_data
from src.preprocessing.preprocessing import process_text, get_dependencies
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
        self.doc_token_counts = {} # {doc: total_tokens}
        self.avg_doc_length = 0.0
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

    def compute_bm25_stats(self):
        print("Computing BM25 statistics...")
        if not self.term_doc_matrix:
            self.load_built_index()

        # Compute document token counts
        # {term: {doc_id: [positions]}}
        doc_counts = defaultdict(int)
        for term, docs in self.term_doc_matrix.items():
            for doc_id, positions in docs.items():
                doc_counts[doc_id] += len(positions)
        
        self.doc_token_counts = dict(doc_counts)
        if self.doc_token_counts:
            self.avg_doc_length = sum(self.doc_token_counts.values()) / len(self.doc_token_counts)
        else:
            self.avg_doc_length = 0.0
        
        save_to_pickle_file({
            'doc_token_counts': self.doc_token_counts,
            'avg_doc_length': self.avg_doc_length
        }, BM25_STATS_PICKLE_PATH)
        print(f"Average document length: {self.avg_doc_length:.2f}")
        return self.doc_token_counts, self.avg_doc_length

    def load_bm25_stats(self):
        data = load_pickle_file(BM25_STATS_PICKLE_PATH)
        if data:
            self.doc_token_counts = data.get('doc_token_counts', {})
            self.avg_doc_length = data.get('avg_doc_length', 0.0)
        return self.doc_token_counts, self.avg_doc_length

    def retrieve_bm25(self, query_str, k=10, k1=1.2, b=0.75):
        """
        Rank documents using BM25.
        score(D, Q) = sum( IDF(qi) * [f(qi, D) * (k1 + 1)] / [f(qi, D) + k1 * (1 - b + b * |D|/avgdl)] )
        """
        print(f"Ranking documents using BM25 for query: '{query_str}'")
        if not self.tf_matrix: self.load_computed_tf()
        if not self.doc_token_counts: self.compute_bm25_stats()
        
        # 1. Query Processing
        processed_query = process_text(query_str)
        query_tokens = processed_query.split()
        
        tf_q = defaultdict(int)
        for t in query_tokens:
            if t in self.vocabulary: tf_q[t] += 1
        
        if not tf_q: return []

        # 2. Parallel Scoring
        relevant_terms = list(tf_q.keys())
        # We need term frequencies for each document and BM25 IDF
        # Standard BM25 IDF: log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        # Using simplified IDF from self.idf_matrix for consistency or standard BM25 IDF
        
        N = self.doc_count
        relevant_data = []
        for term in relevant_terms:
            if term in self.tf_matrix:
                df = len(self.tf_matrix[term])
                # BM25 IDF
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                relevant_data.append((term, self.tf_matrix[term], idf))

        if not relevant_data: return []

        avgdl_bc = self.context.broadcast(self.avg_doc_length)
        doc_token_counts_bc = self.context.broadcast(self.doc_token_counts)
        k1_bc = self.context.broadcast(k1)
        b_bc = self.context.broadcast(b)

        scores = self.context.parallelize(relevant_data) \
            .flatMap(lambda x: [
                (doc_id, x[2] * (tf * (k1_bc.value + 1)) / (tf + k1_bc.value * (1 - b_bc.value + b_bc.value * doc_token_counts_bc.value.get(doc_id, 0) / avgdl_bc.value)))
                for doc_id, tf in x[1].items()
            ]) \
            .reduceByKey(lambda a, b: a + b) \
            .takeOrdered(k, key=lambda x: -x[1])

        return scores

    def expand_query_prf(self, query_str, top_docs, num_expansion_terms=5):
        """
        Extract top terms from the highest ranked documents to expand the query. (Driver/Local)
        """
        term_freqs = defaultdict(int)
        # Using dependencies directly
        _, stopwords, _ = get_dependencies()
        
        relevant_tokens = []
        for doc_id in top_docs:
            content = self.documents.get(doc_id, "")
            tokens = content.split()
            relevant_tokens.extend([t for t in tokens if t not in stopwords and len(t) > 1])
        
        for t in relevant_tokens:
            term_freqs[t] += 1
        
        # Sort by frequency and take top N expansion terms
        # Don't expand with terms already in query if possible (optional)
        query_terms = set(process_text(query_str).split())
        new_terms = [t for t, freq in sorted(term_freqs.items(), key=lambda x: x[1], reverse=True) 
                     if t not in query_terms]
        
        expansion = new_terms[:num_expansion_terms]
        return query_str + " " + " ".join(expansion)

    def retrieve_bm25_prf(self, query_str, k=10, prf_k=3, expansion_terms=3):
        """BM25 with Pseudo-Relevance Feedback expansion"""
        # 1. First pass
        initial_results = self.retrieve_bm25(query_str, k=prf_k)
        if not initial_results:
            return self.retrieve_bm25(query_str, k=k)
        
        # 2. Expand
        top_ids = [r[0] for r in initial_results]
        expanded_query = self.expand_query_prf(query_str, top_ids, num_expansion_terms=expansion_terms)
        print(f"  [PRF] Expanded Query: {expanded_query}")
        
        # 3. Second pass
        return self.retrieve_bm25(expanded_query, k=k)

    def retrieve(self, query_str, k=10):
        """
        Rank documents using the Core Algorithm: 
        Intersection filtering (documents must contain all query terms) followed by Cosine Similarity.
        """
        print(f"Ranking documents for query: '{query_str}'")
        if not self.tf_idf_matrix: self.load_computed_tf_idf()
        if not self.doc_lengths: self.load_computed_doc_lengths()
        if not self.idf_matrix: self.load_computed_idf()

        # 1. Query Processing & Tokenization
        processed_query = process_text(query_str)
        query_tokens = processed_query.split()
        
        # 2. Identify Match Set (Union: any doc containing at least one term)
        matching_docs = set()
        valid_query_tokens = []
        
        for token in query_tokens:
            if token in self.term_doc_matrix:
                valid_query_tokens.append(token)
                # Union: collect all docs containing any of the valid tokens
                matching_docs.update(self.term_doc_matrix[token].keys())
        
        print(f"  [STEP 2] Preprocessed Tokens: {valid_query_tokens}")
        print(f"  [STEP 3] Document Filter: {len(matching_docs)} documents match at least one term.")

        if not matching_docs or not valid_query_tokens:
            return []

        # 3. Compute Query Vector (Keep existing smoothing/log logic)
        tf_q = defaultdict(int)
        for t in valid_query_tokens:
            tf_q[t] += 1
            
        q_vec = {}
        q_len_sq = 0
        for term, count in tf_q.items():
            # weight = (1 + log10(tf)) * idf
            weight = (1 + math.log10(count)) * self.idf_matrix.get(term, 0)
            q_vec[term] = weight
            q_len_sq += weight ** 2
        q_len = math.sqrt(q_len_sq)
        print(f"  [STEP 4] Query Vector L2 Norm: {q_len:.6f}")

        # 4. Parallel Scoring on Filtered Set
        relevant_data = [(t, self.tf_idf_matrix[t]) for t in tf_q.keys() if t in self.tf_idf_matrix]
        
        # Broadcast the intersection set and query metadata
        matching_docs_bc = self.context.broadcast(matching_docs)
        q_vec_bc = self.context.broadcast(q_vec)
        q_len_bc = self.context.broadcast(q_len)
        lengths_bc = self.context.broadcast(self.doc_lengths)

        scores = self.context.parallelize(relevant_data) \
            .flatMap(lambda x: [
                (doc_id, x[1][doc_id] * q_vec_bc.value[x[0]]) 
                for doc_id in x[1] 
                if doc_id in matching_docs_bc.value   # Only score docs in the intersection
            ]) \
            .reduceByKey(lambda a, b: a + b) \
            .map(lambda x: (x[0], x[1] / (q_len_bc.value * lengths_bc.value.get(x[0], 1.0)))) \
            .takeOrdered(k, key=lambda x: -x[1])

        return scores

