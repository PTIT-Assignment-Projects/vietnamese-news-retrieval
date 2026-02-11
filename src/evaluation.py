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

def evaluate_query_batch(row, tf_idf_bc, idf_bc, lengths_bc, vocab_bc, k_vals, qid_col):
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
    tf_q = defaultdict(int)
    for t in q_tokens:
        if t in vocab:
            tf_q[t] += 1
    
    if not tf_q: return None

    # 2. Build Query Vector
    q_vec = {}
    q_len_sq = 0
    idf_matrix = idf_bc.value
    for term, count in tf_q.items():
        weight = (1 + math.log10(count)) * idf_matrix.get(term, 0.0)
        q_vec[term] = weight
        q_len_sq += weight ** 2
    q_len = math.sqrt(q_len_sq)

    # 3. Compute Cosine Similarity Locally
    doc_scores = defaultdict(float)
    tfidf_mat = tf_idf_bc.value
    for term, q_weight in q_vec.items():
        if term in tfidf_mat:
            for doc_id, d_weight in tfidf_mat[term].items():
                doc_scores[doc_id] += q_weight * d_weight
    
    # Normalize scores
    lengths = lengths_bc.value
    final_scores = []
    for doc_id, dot_prod in doc_scores.items():
        score = dot_prod / (q_len * lengths.get(doc_id, 1.0))
        if score > 0:
            final_scores.append((doc_id, score))
    
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

def evaluate():
    print("Initializing Parallel Evaluation Pipeline...")
    spark_ir = SparkInvertedIndexIR()
    
    # Load all components to Driver first
    spark_ir.load_preprocessed_corpus()
    spark_ir.load_built_index()
    spark_ir.load_computed_tf_idf()
    spark_ir.load_computed_doc_lengths()
    spark_ir.load_computed_idf()

    # Broadcast components to Executors
    print("Broadcasting matrices to workers...")
    tf_idf_bc = spark_ir.context.broadcast(spark_ir.tf_idf_matrix)
    idf_bc = spark_ir.context.broadcast(spark_ir.idf_matrix)
    lengths_bc = spark_ir.context.broadcast(spark_ir.doc_lengths)
    vocab_bc = spark_ir.context.broadcast(spark_ir.vocabulary)

    k_values = [1, 5, 10]
    
    from src.dependencies.constant import QID_COLUMN as QID_COL
    print(f"Reading datasets from {TRAIN_PATH} and {TEST_PATH}...")
    train_df = spark_ir.spark_sess.read.parquet(str(TRAIN_PATH)).select(QUESTION_COLUMN, CID_COLUMN, QID_COL)
    test_df = spark_ir.spark_sess.read.parquet(str(TEST_PATH)).select(QUESTION_COLUMN, CID_COLUMN, QID_COL)
    
    # Union both datasets for a larger sample size since no training is involved
    combined_df = train_df.union(test_df)
    
    # repartition to ensure high parallelism
    num_workers = spark_ir.context.defaultParallelism * 4
    queries_rdd = combined_df.rdd.repartition(num_workers)

    print(f"Starting parallel evaluation on {combined_df.count()} queries...")
    metrics_rdd = queries_rdd.map(
        lambda row: evaluate_query_batch(row, tf_idf_bc, idf_bc, lengths_bc, vocab_bc, k_values, QID_COL)
    ).filter(lambda x: x is not None)

    # Collect and aggregate results
    results_list = metrics_rdd.collect()
    num_queries = len(results_list)

    if num_queries == 0:
        print("No valid queries found for evaluation.")
        return

    # Create output directory
    output_dir = UTIL_DIR / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    print(f"Saving individual query results to {output_dir}...")
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

    # Final Output preparation
    report = []
    report.append("="*60)
    report.append(f"PARALLEL EVALUATION RESULTS ({num_queries} queries)")
    report.append("="*60)
    report.append(f"Mean Reciprocal Rank (MRR): {summary['mrr']:.4f}")
    report.append(f"Mean Average Precision (MAP): {summary['map']:.4f}")
    report.append("-" * 60)
    report.append(f"{'K':<10} | {'Precision':<10} | {'Recall':<10} | {'NDCG':<10}")
    report.append("-" * 60)
    
    for k in k_values:
        p = summary[f'p@{k}']
        r = summary[f'r@{k}']
        n = summary[f'ndcg@{k}']
        report.append(f"{k:<10} | {p:<10.4f} | {r:<10.4f} | {n:<10.4f}")
    
    report.append("="*60)
    
    # Print to console
    final_report = "\n".join(report)
    print(final_report)
    
    # Save summary report
    summary_path = UTIL_DIR / "evaluation_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    print(f"Summary saved to {summary_path}")
    print("Evaluation Complete.")

if __name__ == "__main__":
    evaluate()
