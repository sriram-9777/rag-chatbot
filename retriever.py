# retriever.py
from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_knowledge_base")

def retrieve_relevant_chunks(user_question: str, top_k: int = 3) -> list[str]:
    """
    Convert the question to a vector, search the DB,
    return the top matching text chunks.
    """
    # Embed the user's question
    question_embedding = embedding_model.encode(user_question).tolist()
    
    # Search for the most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    # Return just the text of matching chunks
    chunks = results["documents"][0]
    sources = results["metadatas"][0]
    
    print(f"📚 Found {len(chunks)} relevant chunks")
    return chunks, sources