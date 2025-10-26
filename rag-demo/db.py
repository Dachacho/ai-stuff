from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

def get_client(host="localhost", port=6333):
    return QdrantClient(host=host, port=port)

def create_collection(client, collection_name, vector_size):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

def upload_embeddings(client, collection_name, embeddings, payloads):
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload=payloads[i])
        for i in range(len(embeddings))
    ]
    client.upsert(collection_name=collection_name, points=points)

# needed this because it was nearly 4k chunks and single request couldn't handle it
def upload_embeddings_in_batches(client, collection_name, embeddings, payloads, batch_size=200):
    total = len(embeddings)
    for i in range(0, total, batch_size):
        batch_embeds = embeddings[i:i+batch_size]
        batch_payloads = payloads[i:i+batch_size]
        upload_embeddings(client, collection_name, batch_embeds, batch_payloads)
        print(f"Uploaded {min(i+batch_size, total)}/{total} embeddings")

def search(client, collection_name, query_vector, top_k=3):
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )
    return hits

def get_context_for_query(query, client, collection_name, model_name="all-MiniLm-L6-v2", top_k=3):
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query], convert_to_numpy=True)[0]
    results = search(client, collection_name, query_vec, top_k=top_k)
    context_chunks = [hit.payload["text"] for hit in results]
    return "\n".join(context_chunks)