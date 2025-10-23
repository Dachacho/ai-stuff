from datasets import load_dataset
from embeddings import encode_texts
from db import get_client, create_collection, upload_embeddings, search
from spellcheck import spellcheck, build_custom_spellchecker

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
    
    embeddings = encode_texts(texts)
    
    client = get_client()
    create_collection(client, "db", embeddings.shape[1])
    upload_embeddings(client, "db", embeddings, payloads)

    spell = build_custom_spellchecker(texts)

    query = "black sneacker"
    corrected_query = spellcheck(query, spell)
    print(f"Corrected query: {corrected_query}")
    query_vec = encode_texts([corrected_query])[0]
    results = search(client, "db", query_vec, top_k=3)
    print("\nsemantic search results for: ", query)
    for hit in results:
        idx = hit.payload["ds_idx"]
        print(f"Score: {hit.score:.3f}")
        print("image:", hit.payload["text"])
        img = ds[idx]["image"]
        print("image object: ", img)
        print("-" * 40)

if __name__ == "__main__":
    load_and_search()
