import streamlit as st
import pdfplumber
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
import cohere

# Load Embedding and Re-ranking Models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Initialize Cohere API
cohere_api_key = "LC3f55pXtlMb5DAi4pW9PM57HxhfhMX0H89Sg6Xr"
cohere_client = cohere.Client(cohere_api_key)

# Initialize Pinecone
pinecone_api_key = "pcsk_r5BQ5_8441hr1oD14nMzUKaLyhHxCLeotbVuNq1AwUhEWSGvkESctK3CrZ38LoEjW9CuZ"
pinecone_instance = Pinecone(api_key=pinecone_api_key)
index_name = "financial-qa"
index = pinecone_instance.Index(index_name)


def extract_pnl_data(pdf_file):
    """Extracts P&L data from a PDF file."""
    data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if len(row) >= 2 and row[0] and row[-1]:
                        try:
                            description = row[0].strip()
                            value = float(row[-1].replace(',', '').replace('$', ''))
                            data.append((description, value))
                        except ValueError:
                            continue
    return pd.DataFrame(data, columns=["Description", "Value"])


def store_embeddings(df):
    vectors = []
    for idx, row in df.iterrows():
        vector = embedding_model.encode(row["Description"]).tolist()
        metadata = {"description": row["Description"], "value": row["Value"]}
        vectors.append((f"entry_{idx}", vector, metadata))
    index.upsert(vectors)


def query_financial_data(query, top_k=5):
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    descriptions = [match.metadata["description"] for match in results.matches]
    rerank_scores = reranker.predict([(query, desc) for desc in descriptions])
    ranked_results = sorted(zip(results.matches, rerank_scores), key=lambda x: x[1], reverse=True)
    context = "\n".join([f"{match.metadata['description']}: {match.metadata['value']}" for match, _ in ranked_results])
    return generate_response(query, context)


def generate_response(query, context):
    """Generates a response using Cohere."""
    prompt = f"""You are a financial analyst. Answer based on the given data with a structured and detailed numerical breakdown. 
    Ensure that all amounts are correctly formatted in crores and presented clearly. 
    If comparisons are required, explicitly mention the difference and percentage change. 
    Use proper financial terminology and ensure clarity in every component. 
    The response should be structured with tables, bullet points, and headings where necessary for easy readability. 
    Avoid incomplete or fragmented text; provide well-formed and complete sentences.
    
    Context:
    {context}

    Question: {query}
    Answer:"""
    response = cohere_client.generate(model="command", prompt=prompt, max_tokens=200, temperature=0.7)
    return response.generations[0].text.strip()


# Streamlit UI
st.title("Financial QA Bot")
uploaded_file = st.file_uploader("Upload a P&L Statement PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        df = extract_pnl_data(uploaded_file)
        store_embeddings(df)
        st.success("Financial data extracted and stored successfully!")
        st.write(df)

query = st.text_input("Enter your financial question")
if st.button("Get Answer") and query:
    with st.spinner("Retrieving and generating response..."):
        response = query_financial_data(query)
        st.write("**Answer:**", response)
