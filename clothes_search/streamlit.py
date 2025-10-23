import streamlit as st
from datasets import load_dataset
from embeddings import encode_texts
from db import get_client, search
from spellcheck import spellcheck, build_custom_spellchecker

@st.cache_resource
def load_resources():
    ds = load_dataset("ghoumrassi/clothes_sample", split="train[:990]")
    client = get_client()
    texts = [item["text"] for item in ds]
    return ds, client, texts

ds, client, texts = load_resources()
COLLECTION = "db"

spell = build_custom_spellchecker(texts)

st.title("Clothes Semantic Search Demo")

query = st.text_input("enter your search")

if query:
    corrected_query = spellcheck(query, spell)
    query_vec = encode_texts([corrected_query])[0]
    results = search(client, COLLECTION, query_vec, top_k=5)
    st.subheader("Results")
    for hit in results:
        idx = hit.payload["ds_idx"]
        st.image(ds[idx]["image"], 
                 caption=f"Score: {hit.score:.3f}\n{hit.payload['text']}", 
                 use_container_width=True)