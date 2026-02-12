import math
import numpy as np
from collections import defaultdict
from src.preprocessing.positional_index import SparkInvertedIndexIR
from src.preprocessing.preprocessing import process_text
import json
from pathlib import Path
from src.dependencies.constant import TRAIN_PATH, TEST_PATH, QUESTION_COLUMN, CID_COLUMN, UTIL_DIR

# --- METRIC FUNCTIONS ---

def calculate_precision_at_k(retrieved_ids, true_set, k):
    if k == 0: return 0.0
    k_retrieved = retrieved_ids[:k]
    hits = len([r_id for r_id in k_retrieved if r_id in true_set])
    return hits / k

def calculate_recall_at_k(retrieved_ids, true_set, k):
    if not true_set: return 0.0
    k_retrieved = retrieved_ids[:k]
    hits = len([r_id for r_id in k_retrieved if r_id in true_set])
    return hits / len(true_set)

def calculate_mrr(retrieved_ids, true_set):
    for i, r_id in enumerate(retrieved_ids):
        if r_id in true_set:
            return 1.0 / (i + 1)
    return 0.0

def calculate_ap(retrieved_ids, true_set):
    if not true_set: return 0.0
    ap = 0.0
    hits = 0
    for i, r_id in enumerate(retrieved_ids):
        if r_id in true_set:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(true_set)

def calculate_ndcg_at_k(retrieved_ids, true_set, k):
    if not true_set: return 0.0
    k_retrieved = retrieved_ids[:k]
    dcg = sum([1.0 / math.log2(i + 2) for i, r_id in enumerate(k_retrieved) if r_id in true_set])
    idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(true_set), k))])
    return dcg / idcg if idcg > 0 else 0.0

# --- SPARK WORKER LOGIC ---

def evaluate_query_batch(row, tf_idf_bc, idf_bc, lengths_bc, vocab_bc, k_vals, qid_col, method="tfidf", bm25_stats_bc=None, tf_matrix_bc=None, N_bc=None):
    """
    This function runs on Spark Executors. It performs retrieval locally using 
    broadcasted data instead of triggering sub-spark jobs.
    """
    query_str = row[QUESTION_COLUMN]
    ground_truth = row[CID_COLUMN]
    qid = row[qid_col] if qid_col in row else None
    if ground_truth is None: return None
    
    # Ground truth handle
    true_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
    
    # 1. Preprocess query locally
    processed_query = process_text(query_str)
    q_tokens = processed_query.split()
    
    vocab = vocab_bc.value
    valid_query_tokens = [t for t in q_tokens if t in vocab]
    
    if not valid_query_tokens: return None

    doc_scores = defaultdict(float)
    
    if method == "tfidf":
        # Union logic to match BM25 sample size coverage
        matching_docs = set()
        tf_q = defaultdict(int)
        for t in q_tokens:
            if t in vocab:
                tf_q[t] += 1
                doc_list = set(tf_idf_bc.value.get(t, {}).keys())
                matching_docs.update(doc_list) # Any doc with any query term
        
        if not matching_docs: return None

        # Build Query Vector
        q_vec = {}
        q_len_sq = 0
        idf_matrix = idf_bc.value
        for term, count in tf_q.items():
            weight = (1 + math.log10(count)) * idf_matrix.get(term, 0.0)
            q_vec[term] = weight
            q_len_sq += weight ** 2
        q_len = math.sqrt(q_len_sq)

        # Compute Cosine Similarity Locally
        tfidf_mat = tf_idf_bc.value
        for term, q_weight in q_vec.items():
            if term in tfidf_mat:
                for doc_id, d_weight in tfidf_mat[term].items():
                    if doc_id in matching_docs:
                        doc_scores[doc_id] += q_weight * d_weight
        
        # Normalize scores
        lengths = lengths_bc.value
        final_scores = []
        for doc_id, dot_prod in doc_scores.items():
            score = dot_prod / (q_len * lengths.get(doc_id, 1.0))
            if score > 0:
                final_scores.append((doc_id, score))
                
    elif method == "bm25":
        k1, b = 1.2, 0.75
        avgdl = bm25_stats_bc.value['avg_doc_length']
        doc_counts = bm25_stats_bc.value['doc_token_counts']
        tf_matrix = tf_matrix_bc.value
        N = N_bc.value
        
        for token in valid_query_tokens:
            if token in tf_matrix:
                df = len(tf_matrix[token])
                # BM25 IDF
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                for doc_id, tf in tf_matrix[token].items():
                    doc_len = doc_counts.get(doc_id, 0)
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
                    doc_scores[doc_id] += score
        
        final_scores = list(doc_scores.items())

    # Rank & Get Top K
    max_k = max(k_vals)
    final_scores.sort(key=lambda x: x[1], reverse=True)
    retrieved_ids = [s[0] for s in final_scores[:max_k]]

    # 4. Calculate Batch Metrics
    results = {
        'mrr': calculate_mrr(retrieved_ids, true_set),
        'map': calculate_ap(retrieved_ids, true_set),
    }
    for k in k_vals:
        results[f'p@{k}'] = calculate_precision_at_k(retrieved_ids, true_set, k)
        results[f'r@{k}'] = calculate_recall_at_k(retrieved_ids, true_set, k)
        results[f'ndcg@{k}'] = calculate_ndcg_at_k(retrieved_ids, true_set, k)
        
    return {
        'qid': qid,
        'query': query_str,
        'metrics': results,
        'retrieved': retrieved_ids
    }

def run_evaluation_for_method(spark_ir, queries_rdd, method, tf_idf_bc, idf_bc, lengths_bc, vocab_bc, bm25_stats_bc, tf_matrix_bc, N_bc, k_values, qid_col):
    print(f"Starting parallel evaluation for {method}...")
    metrics_rdd = queries_rdd.map(
        lambda row: evaluate_query_batch(row, tf_idf_bc, idf_bc, lengths_bc, vocab_bc, k_values, qid_col, method, bm25_stats_bc, tf_matrix_bc, N_bc)
    ).filter(lambda x: x is not None)

    results_list = metrics_rdd.collect()
    num_queries = len(results_list)

    if num_queries == 0:
        print(f"No valid queries found for {method} evaluation.")
        return None

    # Save results
    output_dir = UTIL_DIR / "evaluation_results" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in results_list:
        qid = item['qid'] or f"query_{hash(item['query'])}"
        with open(output_dir / f"{qid}.json", 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

    # Calculate means
    summary = defaultdict(float)
    for item in results_list:
        m = item['metrics']
        for key, val in m.items():
            summary[key] += val / num_queries
            
    return summary, num_queries

def evaluate():
    print("Initializing Parallel Evaluation Pipeline...")
    spark_ir = SparkInvertedIndexIR()
    
    # Load all components to Driver
    spark_ir.load_preprocessed_corpus()
    spark_ir.load_built_index()
    spark_ir.load_computed_tf_idf()
    spark_ir.load_computed_doc_lengths()
    spark_ir.load_computed_idf()
    spark_ir.load_computed_tf()
    spark_ir.load_bm25_stats()
    if not spark_ir.doc_token_counts:
        spark_ir.compute_bm25_stats()

    # Broadcast
    print("Broadcasting matrices to workers...")
    tf_idf_bc = spark_ir.context.broadcast(spark_ir.tf_idf_matrix)
    idf_bc = spark_ir.context.broadcast(spark_ir.idf_matrix)
    lengths_bc = spark_ir.context.broadcast(spark_ir.doc_lengths)
    vocab_bc = spark_ir.context.broadcast(spark_ir.vocabulary)
    bm25_stats_bc = spark_ir.context.broadcast({
        'avg_doc_length': spark_ir.avg_doc_length,
        'doc_token_counts': spark_ir.doc_token_counts
    })
    tf_matrix_bc = spark_ir.context.broadcast(spark_ir.tf_matrix)
    N_bc = spark_ir.context.broadcast(spark_ir.doc_count)

    k_values = [1, 5, 10]
    from src.dependencies.constant import QID_COLUMN as QID_COL
    
    train_df = spark_ir.spark_sess.read.parquet(str(TRAIN_PATH)).select(QUESTION_COLUMN, CID_COLUMN, QID_COL)
    test_df = spark_ir.spark_sess.read.parquet(str(TEST_PATH)).select(QUESTION_COLUMN, CID_COLUMN, QID_COL)
    combined_df = train_df.union(test_df)
    
    num_workers = spark_ir.context.defaultParallelism * 4
    queries_rdd = combined_df.rdd.repartition(num_workers).cache()

    all_summaries = {}
    for method in ["tfidf", "bm25"]:
        res = run_evaluation_for_method(spark_ir, queries_rdd, method, tf_idf_bc, idf_bc, lengths_bc, vocab_bc, bm25_stats_bc, tf_matrix_bc, N_bc, k_values, QID_COL)
        if res:
            summary, num_q = res
            all_summaries[method] = summary
            
            # Save specific report
            report = generate_report_text(method, summary, num_q, k_values)
            report_path = UTIL_DIR / f"evaluation_summary_{method}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(report)

    print("Total Evaluation Complete.")

def generate_report_text(method, summary, num_queries, k_values):
    report = []
    report.append("="*60)
    report.append(f"EVALUATION RESULTS: {method.upper()} ({num_queries} queries)")
    report.append("="*60)
    report.append(f"MRR: {summary['mrr']:.4f}")
    report.append(f"MAP: {summary['map']:.4f}")
    report.append("-" * 60)
    report.append(f"{'K':<10} | {'Precision':<10} | {'Recall':<10} | {'NDCG':<10}")
    report.append("-" * 60)
    for k in k_values:
        p, r, n = summary[f'p@{k}'], summary[f'r@{k}'], summary[f'ndcg@{k}']
        report.append(f"{k:<10} | {p:<10.4f} | {r:<10.4f} | {n:<10.4f}")
    report.append("="*60)
    return "\n".join(report)

if __name__ == "__main__":
    evaluate()

