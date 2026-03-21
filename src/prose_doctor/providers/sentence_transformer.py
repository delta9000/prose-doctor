"""Sentence-transformer provider -- all-MiniLM-L6-v2 (384d)."""


def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")
