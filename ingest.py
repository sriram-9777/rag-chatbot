# ingest.py
# ingest.py — fixed imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ Fixed
from sentence_transformers import SentenceTransformer
import chromadb
import os

# 1. Load your documents (PDF or TXT)
def load_documents(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

# 2. Split into chunks
# Why chunks? LLMs have token limits — we can't send 100 pages at once
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # ~500 characters per chunk
        chunk_overlap=50      # 50 char overlap so context isn't lost at edges
    )
    return splitter.split_documents(documents)

# 3. Embed and store in ChromaDB
def store_in_vectordb(chunks):
    # Load a free local embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create ChromaDB client (saves data locally)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("my_knowledge_base")
    
    # Convert each chunk to a vector and store it
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk.page_content).tolist()
        collection.add(
            documents=[chunk.page_content],
            embeddings=[embedding],
            ids=[f"chunk_{i}"],
            metadatas=[{"source": chunk.metadata.get("source", "unknown")}]
        )
    
    print(f"✅ Stored {len(chunks)} chunks in vector database")

# Run ingestion
if __name__ == "__main__":
    docs = load_documents("your_document.txt")  # ✅ changed to txt file
    chunks = split_documents(docs)
    store_in_vectordb(chunks)
    print("Ingestion complete!")