import streamlit as st
from db import get_client, get_context_for_query
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
hf_client = InferenceClient(
    provider="cerebras",
    api_key=os.environ["HUGGING_FACE_TOKEN"],
)

st.title("C# in a Nutshell RAG Q&A")

query = st.text_input("Ask a question about C#:")

if st.button("Get Answer") and query.strip():
    with st.spinner("Retrieving answer..."):
        client = get_client()
        context = get_context_for_query(query, client, "rag_demo", top_k=3)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
        ]
        completion = hf_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content
    st.markdown("### Answer")
    st.write(answer)
    with st.expander("Show retrieved context"):
        st.write(context)