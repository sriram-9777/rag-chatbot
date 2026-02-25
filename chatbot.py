# chatbot.py
from google import genai
from google.genai import types
import os
from retriever import retrieve_relevant_chunks
from search import google_search
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Conversation history
conversation_history = []

def build_system_prompt(rag_context: str, search_context: str) -> str:
    return f"""You are a powerful AI assistant with access to two knowledge sources:
1. A private knowledge base (RAG documents)
2. Real-time Google Search results

Use BOTH sources to give the most complete, accurate, and up-to-date answer possible.
If the knowledge base has relevant info, prioritize it.
If the question needs current/real-time info, rely more on search results.
If neither source has enough info, use your own general knowledge.
Always be clear, helpful, and accurate.

--- KNOWLEDGE BASE (RAG) ---
{rag_context if rag_context else "No relevant documents found."}

--- GOOGLE SEARCH RESULTS ---
{search_context if search_context else "No search results found."}
"""

def chat(user_message: str):
    # Step 1: Search RAG knowledge base
    try:
        relevant_chunks, sources = retrieve_relevant_chunks(user_message)
        rag_context = "\n\n".join(relevant_chunks)
    except Exception:
        rag_context = ""
        sources = []

    # Step 2: Search Google for real-time info
    print(f"🔍 Searching Google for: {user_message}")
    search_results = google_search(user_message)

    # Step 3: Build system prompt with both contexts
    system_prompt = build_system_prompt(rag_context, search_results)

    # Step 4: Build conversation history
    history_text = ""
    for msg in conversation_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    full_prompt = f"{history_text}User: {user_message}"

    # Step 5: Call Gemini
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=full_prompt
    )

    assistant_reply = response.text

    # Step 6: Save to history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, [s["source"] for s in sources]


# Terminal test
if __name__ == "__main__":
    print("🤖 Smart RAG + Search Chatbot Ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        answer, sources = chat(user_input)
        print(f"\nBot: {answer}")
        if sources:
            print(f"📎 RAG Sources: {sources}\n")