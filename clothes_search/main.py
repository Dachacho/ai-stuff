from datasets import load_dataset
from embeddings import encode_texts
from db import get_client, create_collection, upload_embeddings, search
from bmranker import build_vocab, rank
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def load_and_search():
    ds = load_dataset("ghoumrassi/clothes_sample", split="train[:990]")

    texts = []
    payloads = []

    for idx, item in enumerate(ds):
        texts.append(item["text"])
        payloads.append({
            "text": item["text"],
            "ds_idx": int(idx)
        })

    vocab, bm25 = build_vocab(texts)
    embeddings = encode_texts(texts)
    client = get_client()
    create_collection(client, "db", embeddings.shape[1])
    upload_embeddings(client, "db", embeddings, payloads)

    query = "red jaleckt"
    query_vec = encode_texts([query])[0]

    bm25_results = rank(texts, query, vocab, bm25, top_k=5)
    print("\nBM25 candidates:")
    candidate_indices = [idx for _, _, idx in bm25_results]
    for text, score, idx in bm25_results:
        print(f"BM25 Score: {score:.2f}")
        print("Text:", text)
        print("-" * 40)
    
    candidate_texts = [texts[idx] for idx in candidate_indices]
    candidate_embeddings = encode_texts(candidate_texts)
    similarities = [cosine_similarity(query_vec, emb) for emb in candidate_embeddings]
    reranked = sorted(zip(candidate_texts, similarities, candidate_indices), key=lambda x: x[1], reverse=True)


    print("\nSemantic reranked BM25 candidates:")
    for text, sim, idx in reranked:
        print(f"Cosine similarity: {sim:.3f}")
        print("Text:", text)
        img = ds[idx]["image"]
        print("Image object:", img)
        print("-" * 40)

    
    # results = search(client, "db", query_vec, top_k=3)
    # print("\nsemantic search results for: ", query)
    # for hit in results:
    #     idx = hit.payload["ds_idx"]
    #     print(f"Score: {hit.score:.3f}")
    #     print("image:", hit.payload["text"])
    #     img = ds[idx]["image"]
    #     print("image object: ", img)
    #     print("-" * 40)

if __name__ == "__main__":
    load_and_search()
