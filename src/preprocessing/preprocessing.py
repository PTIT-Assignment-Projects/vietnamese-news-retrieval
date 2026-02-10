import re

import pandas as pd

# Global cache to load dependencies only once per worker process
_TOKENIZER = None
_STOPWORDS = None
_WORD_NORMALIZE = None

def get_dependencies():
    global _TOKENIZER, _STOPWORDS, _WORD_NORMALIZE
    if _TOKENIZER is None:
        from underthesea import word_tokenize
        _TOKENIZER = word_tokenize
    if _STOPWORDS is None:
        from src.preprocessing.data_loader import vietnamese_stopwords
        _STOPWORDS = vietnamese_stopwords
    if _WORD_NORMALIZE is None:
        from underthesea import text_normalize
        _WORD_NORMALIZE = text_normalize
    return _TOKENIZER, _STOPWORDS, _WORD_NORMALIZE


def tokenize_batch_logic(content_series: pd.Series) -> pd.Series:
    word_tokenize, stopwords, text_normalize_underthesea = get_dependencies()

    def process_text(text: str) -> str:
        if text is None: return ""
        words = text_normalize_underthesea(text)
        words = words.lower()
        text = words.replace("\ufffd", " ")
        tokens = word_tokenize(text, format="text", use_token_normalize=True).split()
        valid_pattern = re.compile(
            r'^[a-z0-9_àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\.\-]+$')
        cleaned_tokens = []
        for t in tokens:
            # Only keep tokens that match our valid character set
            if not valid_pattern.match(t):
                continue

            # Finally, check for stopwords and length
            if t not in stopwords and len(t) > 1:
                cleaned_tokens.append(t)
        return " ".join(cleaned_tokens)

    return content_series.apply(process_text)