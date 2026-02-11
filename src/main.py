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
    
    results = spark.retrieve(query, k=5)
    
    if results:
        print("\n[STEP 5] Final Ranking (Top 5):")
        print("-" * 30)
        for i, (doc_id, score) in enumerate(results, 1):
            # Try to get a snippet from the document if available
            doc_content = spark.documents.get(doc_id, "No content available")[:100] + "..."
            print(f"{i}. Doc ID: {doc_id} | Score: {score:.6f}")
            print(f"   Excerpt: {doc_content}\n")
    else:
        print("\nNo matching documents found.")
    
    print("="*60)
    print("DEMO COMPLETE")
    print("="*60 + "\n")
    
if __name__ == "__main__":
    main()