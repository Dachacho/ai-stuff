from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

def get_client(host="localhost", port=6333):
    return QdrantClient(host=host, port=port)

def create_collection(client, collection_name, vector_size):
    try:
        client.delete(collection_name)
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

def search(client, collection_name, query_vector, top_k=3):
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )
    return hits