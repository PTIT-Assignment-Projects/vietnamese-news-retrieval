import pandas as pd
from pyspark.sql.functions import pandas_udf
from typing import Iterator

# Global cache to load dependencies only once per worker process
_TOKENIZER = None
_STOPWORDS = None

def get_dependencies():
    global _TOKENIZER, _STOPWORDS
    if _TOKENIZER is None:
        from underthesea import word_tokenize
        _TOKENIZER = word_tokenize
    if _STOPWORDS is None:
        from src.preprocessing.data_loader import vietnamese_stopwords
        _STOPWORDS = vietnamese_stopwords
    return _TOKENIZER, _STOPWORDS


def tokenize_batch_logic(content_series: pd.Series) -> pd.Series:
    word_tokenize, stopwords = get_dependencies()

    def process_text(text: str) -> str:
        if text is None: return ""
        words = text.lower()
        tokens = word_tokenize(words, format="text", use_token_normalize=True).split()
        return " ".join([t for t in tokens if t not in stopwords])

    return content_series.apply(process_text)

# Keep the original function for compatibility but optimize it
def tokenize_words(text: str) -> str:
    word_tokenize, stopwords = get_dependencies()
    if text is None:
        return ""
    words = text.lower()
    tokens = word_tokenize(words, format="text", use_token_normalize=True).split()
    cleaned = [t for t in tokens if t not in stopwords]
    return " ".join(cleaned)