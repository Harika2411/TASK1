import os
import PyPDF2
import pdfplumber
from sentence_transformers import SentenceTransformer
import pinecone
import pandas as pd
from openai import OpenAI

# 1. Initialize Configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Pre-trained embedding model from Hugging Face
VECTOR_DB_NAME = "pdf_embeddings"
API_KEY = "your_openai_api_key"  # Replace with your OpenAI key
PINECONE_API_KEY = "your_pinecone_api_key"  # Replace with your Pinecone key
PINECONE_ENV = "your_pinecone_environment"  # Replace with your Pinecone environment

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if VECTOR_DB_NAME not in pinecone.list_indexes():
    pinecone.create_index(VECTOR_DB_NAME, dimension=384)
vector_db = pinecone.Index(VECTOR_DB_NAME)

# 2. Function to Process PDFs
def process_pdf(file_path):
    """Extract and chunk text from PDF."""
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Split into smaller chunks
                for i in range(0, len(text), 500):
                    chunk = text[i:i + 500]
                    chunks.append({"text": chunk, "page": page_num + 1})
    return chunks

# 3. Generate Vector Embeddings
def embed_and_store(chunks, file_name):
    """Generate embeddings and store them in Pinecone."""
    for chunk in chunks:
        embedding = embedding_model.encode(chunk["text"], convert_to_tensor=True).tolist()
        metadata = {"file_name": file_name, "page": chunk["page"]}
        vector_db.upsert([(f"{file_name}-{chunk['page']}", embedding, metadata)])

# 4. Query the Database
def query_database(query, top_k=5):
    """Retrieve relevant chunks based on user query."""
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).tolist()
    results = vector_db.query(query_embedding, top_k=top_k, include_metadata=True)
    return results

# 5. Generate Response using LLM
def generate_response(query, retrieved_chunks):
    """Generate a response using the LLM with retrieved context."""
    context = "\n".join([f"Page {chunk['metadata']['page']}: {chunk['metadata']['text']}" for chunk in retrieved_chunks])
    prompt = f"Answer the following query using the provided context:\n\nContext:\n{context}\n\nQuery:\n{query}\n\nAnswer:"
    response = OpenAI(api_key=API_KEY).completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response["choices"][0]["text"]

# 6. Main Flow
if __name__ == "__main__":
    # Specify PDF file paths
    pdf_files = ["example1.pdf", "example2.pdf"]

    # Ingest PDFs
    for pdf_file in pdf_files:
        chunks = process_pdf(pdf_file)
        embed_and_store(chunks, os.path.basename(pdf_file))

    # Handle user query
    user_query = "What is the unemployment rate for people with a bachelor's degree?"
    results = query_database(user_query, top_k=5)
    response = generate_response(user_query, results)

    print("Response:", response)
