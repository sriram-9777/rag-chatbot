# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import chat
import uvicorn

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def root():
    return {"status": "RAG Chatbot is running! 🚀"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        answer, sources = chat(request.message)
        
        # ✅ Fix — safely convert sources to list of strings
        clean_sources = []
        for s in sources:
            if isinstance(s, dict):
                clean_sources.append(s.get("source", "unknown"))
            else:
                clean_sources.append(str(s))

        return ChatResponse(answer=answer, sources=clean_sources)

    except Exception as e:
        # ✅ Print full error so we can see it in terminal
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)