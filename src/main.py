from src.preprocessing.positional_index import SparkInvertedIndexIR


def main():
    print("Initializing IR System...")
    spark = SparkInvertedIndexIR()
    
    # Pre-load necessary components
    spark.load_preprocessed_corpus()
    spark.load_built_index()
    spark.load_computed_tf()
    spark.load_computed_idf()
    spark.load_computed_tf_idf()
    spark.load_computed_doc_lengths()
    
    print("\n" + "="*60)
    print("STEP-BY-STEP QUERY RETRIEVAL DEMO")
    print("="*60)
    
    query = "Hành trình khởi nghiệp của các tỷ phú công nghệ"
    print(f"\n[STEP 1] Input Query: '{query}'")
    
    # 1. TF-IDF Results
    print("\n" + "-"*30)
    print("1. VECTOR SPACE MODEL (TF-IDF)")
    print("-"*30)
    tfidf_results = spark.retrieve(query, k=5)
    if tfidf_results:
        for i, (doc_id, score) in enumerate(tfidf_results, 1):
            doc_content = spark.documents.get(doc_id, "No content available")[:100] + "..."
            print(f"{i}. Doc ID: {doc_id} | Score: {score:.6f}")
            print(f"   Excerpt: {doc_content}")
    else:
        print("No matching documents found with TF-IDF.")

    # 2. BM25 Results
    print("\n" + "-"*30)
    print("2. BM25 MODEL")
    print("-"*30)
    bm25_results = spark.retrieve_bm25(query, k=5)
    if bm25_results:
        for i, (doc_id, score) in enumerate(bm25_results, 1):
            doc_content = spark.documents.get(doc_id, "No content available")[:100] + "..."
            print(f"{i}. Doc ID: {doc_id} | Score: {score:.6f}")
            print(f"   Excerpt: {doc_content}")
    else:
        print("No matching documents found with BM25.")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60 + "\n")
    
if __name__ == "__main__":
    main()