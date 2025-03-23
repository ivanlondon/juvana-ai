from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from utils.loaders import load_user_documents
from utils.chunking import split_documents
from utils.qa import build_qa_chain
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

app = FastAPI()

# --- CORS config ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
USER_ID = "user_001"
DOMAIN = "immigration"
persist_dir = f"./chroma_store/{USER_ID}/{DOMAIN}"
qa_chain = None  # Global chain object

# --- Init LLM pipeline on startup ---
@app.on_event("startup")
async def init_pipeline():
    global qa_chain
    print("🚀 Initializing LLM pipeline...")
    docs = load_user_documents(USER_ID, DOMAIN)
    chunks = split_documents(docs)
    embedding = OllamaEmbeddings(model="tinyllama")
    Chroma.from_documents(chunks, embedding=embedding, persist_directory=persist_dir)
    qa_chain = build_qa_chain(USER_ID, DOMAIN)
    print("✅ LLM pipeline initialized.")

# --- Main API Endpoint ---
@app.post("/query")
async def query(request: Request):
    global qa_chain
    if qa_chain is None:
        return {"result": "LLM pipeline is not ready yet."}

    body = await request.json()
    user_question = body.get("query")

    if not user_question:
        return {"result": "No query provided."}

    print(f"🔎 Received query: {user_question}")
    result = qa_chain.invoke(user_question)
    return {"result": result['result']}
