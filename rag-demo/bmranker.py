from rank_bm25 import BM25Okapi
import difflib

def build_vocab(texts):
    tokenized_corpus = [[w.strip(".,") for w in text.lower().split()] for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    vocab = set(w for doc in tokenized_corpus for w in doc)
    return vocab, bm25

def fuzzy_query(query, vocab):
    return " ".join([
        difflib.get_close_matches(word, vocab, n=1, cutoff=0.7)[0] if difflib.get_close_matches(word, vocab, n=1, cutoff=0.7) else word
        for word in query.lower().split()
    ])

def rank(texts, query, vocab, bm25, top_k=3):
    fuzzy = fuzzy_query(query, vocab)
    tokenized_query = fuzzy.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indecies = sorted(range(len(bm25_scores)), key=lambda i:bm25_scores[i], reverse=True)[:top_k]
    return [(texts[idx], bm25_scores[idx], idx) for idx in top_indecies]