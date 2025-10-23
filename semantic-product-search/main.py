from datasets import load_dataset
from embeddings import encode_texts
from database import get_client, create_collection, upload_embeddings, search

def load_and_print_products():
    dataset = load_dataset("ag_news", split="train[:1000]")
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    texts = []
    payloads = []
    for item in dataset:
        texts.append(item["text"])
        payloads.append({
            "text": item["text"],
            "category": label_names[item["label"]]
        })
    
    embeddings = encode_texts(texts)
    print("embeddings shape: ", embeddings.shape)
    print("first embedding vector (turnicated): ", embeddings[0][:5])

    client = get_client()
    create_collection(client, "poopoo", embeddings.shape[1])
    upload_embeddings(client, "poopoo", embeddings, payloads)

    query = "stock market and oil prices"
    query_vec = encode_texts([query])[0]
    results = search(client, "poopoo", query_vec, top_k=3)
    print("\nsemantic search results for: ", query)
    for hit in results:
        print(f"Score: {hit.score:.3f}")
        print("Text:", hit.payload["text"])
        print("Category:", hit.payload["category"])
        print("-" * 40)

if __name__ == "__main__":
    load_and_print_products()