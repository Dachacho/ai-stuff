import streamlit as st
from datasets import load_dataset
from embeddings import encode_texts
from db import get_client, search

@st.cache_resource
def load_resources():
    ds = load_dataset("ghoumrassi/clothes_sample", split="train[:990]")
    client = get_client()
    return ds, client

ds, client = load_resources()
COLLECTION = "db"

st.title("Clothes Semantic Search Demo")

query = st.text_input("enter your search")

if query:
    query_vec = encode_texts([query])[0]
    results = search(client, COLLECTION, query_vec, top_k=5)
    st.subheader("Results")
    for hit in results:
        idx = hit.payload["ds_idx"]
        st.image(ds[idx]["image"], 
                 caption=f"Score: {hit.score:.3f}\n{hit.payload['text']}", 
                 use_column_width=True)