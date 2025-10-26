import pdfplumber
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from db import get_client, create_collection, upload_embeddings_in_batches, search

def create_string_from_pdf():
    with pdfplumber.open('csharp-in-a-nutshell.pdf') as pdf:
        full_text = ''

        for page in pdf.pages:
            text = page.extract_text()
            full_text += text + ' '
    
        return full_text

def create_chunks(text, max_chunk_size=1000, overlap_sentences=2):
    sentences = re.split(r'(?<=[.!?])\s+', text) 

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            current_chunk = current_chunk[-overlap_sentences:]
            current_size = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def encode_chunks(texts, model_name="all-MiniLm-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(embeddings)

def main():
    text = create_string_from_pdf()
    print("PDF extracted")

    chunks = (create_chunks(text))
    print(f"text chunked into {len(chunks)} chunks")

    embeddings = encode_chunks(chunks)

    payloads = [{"text": chunk, "chunk_idx": idx} for idx, chunk in enumerate(chunks)]

    client = get_client()
    create_collection(client, "rag_demo", embeddings.shape[1])
    upload_embeddings_in_batches(client, "rag_demo", embeddings, payloads, batch_size=200)
    print("embeddings uploaded to db")

if __name__ == "__main__":
    main()