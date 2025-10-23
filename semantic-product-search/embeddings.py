from sentence_transformers import SentenceTransformer
import numpy as np

def encode_texts(texts, model_name="all-MiniLm-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(embeddings)