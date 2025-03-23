from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

def build_qa_chain(user_id: str, domain: str = "immigration"):
    persist_dir = f"./chroma_store/{user_id}/{domain}"
    embedding = OllamaEmbeddings(model="tinyllama")

    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = vectordb.as_retriever()

    llm = OllamaLLM(model="tinyllama")

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
