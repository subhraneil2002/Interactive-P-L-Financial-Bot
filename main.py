import os
import streamlit as st
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, ServerlessSpec
import cohere
from dotenv import load_dotenv

# --- Load API Keys securely ---
load_dotenv()  # Load environment variables from a .env file

cohere_api_key = os.getenv("COHERE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")  # fallback to default

# Ensure that the API keys are loaded properly
print(f"Cohere API Key: {cohere_api_key}")
print(f"Pinecone API Key: {pinecone_api_key}")

# --- Initialize Models ---
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Initialize Cohere ---
cohere_client = cohere.Client(cohere_api_key)

# --- Initialize Pinecone ---
pc = Pinecone(api_key=pinecone_api_key)

index_name = "financial-qa"

# Check if the index exists, create if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Correct dimension for MiniLM-L6-v2
        metric="cosine",  # You can also use "euclidean"
    )

# Use the created index
index = pc.Index(index_name)


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
                            value = float(row[-1].replace(',', '').replace('$', '').strip())
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
    matches = results.get("matches", [])
    if not matches:
        return "No relevant data found."

    descriptions = [match['metadata']["description"] for match in matches]
    rerank_scores = reranker.predict([(query, desc) for desc in descriptions])
    ranked = sorted(zip(matches, rerank_scores), key=lambda x: x[1], reverse=True)

    context_lines = [
        f"{match['metadata']['description']}: {match['metadata']['value']}"
        for match, _ in ranked
    ]
    context = "\n".join(context_lines[:top_k])

    return generate_response(query, context)


def generate_response(query, context):
    """Generates a response using Cohere."""
    prompt = f"""You are a financial analyst. Answer based on the given data with a structured and detailed numerical breakdown.
Ensure that all amounts are correctly formatted in crores and presented clearly.
If comparisons are required, explicitly mention the difference and percentage change.
Use proper financial terminology and ensure clarity in every component.
Provide well-formed and complete sentences.

Context:
{context}

Question: {query}
Answer:"""

    # Truncate prompt if needed
    if len(prompt) > 3000:
        prompt = prompt[:3000]

    response = cohere_client.generate(model="command", prompt=prompt, max_tokens=200, temperature=0.7)
    return response.generations[0].text.strip()


# --- Streamlit UI ---
st.title("📊 Financial QA Bot")
uploaded_file = st.file_uploader("Upload a P&L Statement PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Processing PDF..."):
        df = extract_pnl_data(uploaded_file)
        if df.empty:
            st.error("No valid financial data found in the uploaded PDF.")
        else:
            store_embeddings(df)
            st.success("✅ Financial data extracted and stored successfully!")
            st.dataframe(df)

query = st.text_input("💬 Enter your financial question")
if st.button("Get Answer") and query:
    with st.spinner("🔍 Retrieving and generating response..."):
        response = query_financial_data(query)
        st.markdown(f"**Answer:**\n\n{response}")
