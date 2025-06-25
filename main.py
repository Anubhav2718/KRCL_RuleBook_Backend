# app.py (FastAPI)
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
groq_api_key=os.environ['GROQ_API_KEY']

# Load embeddings & vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db1 = FAISS.load_local("faiss_index_krcl", embeddings, allow_dangerous_deserialization=True)
db2 = FAISS.load_local("faiss_manual_krcl", embeddings, allow_dangerous_deserialization=True)
retriever1 = db1.as_retriever(search_kwargs={"k": 4})
retriever2 = db2.as_retriever(search_kwargs={"k": 4})

# Define retriever functions
def gsr_retriever_tool(query: str) -> str:
    results = retriever1.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in results]) if results else "No relevant rules found in G&SR."

def manual_retriever_tool(query: str) -> str:
    results = retriever2.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in results]) if results else "No relevant content found."

def combined_retriever_tool(query: str) -> str:
    results1 = retriever1.get_relevant_documents(query)
    results2 = retriever2.get_relevant_documents(query)
    combined = results1[:4] + results2[:4]
    return "\n\n".join([doc.page_content for doc in combined]) if combined else "No relevant content found."

# LLM
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-8b-8192"
)
# Agent setup
tools = [
    Tool(name="General and Subsidiary Rules", func=gsr_retriever_tool, description="G&SR tool"),
    Tool(name="Accidental Manual", func=manual_retriever_tool, description="Manual tool"),
    Tool(name="Combined", func=combined_retriever_tool, description="Combined tool"),
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# API Setup
app = FastAPI()

class Query(BaseModel):
    input: str

@app.get("/")
def read_root():
    return {"message": "KRCL RuleBot backend is live!"}

@app.post("/ask")
async def ask_query(query: Query):
    result = agent_executor.invoke({"input": query.input, "chat_history": []})
    return {
        "final_answer": result["output"],
        "action": result["intermediate_steps"][0][0].tool if result["intermediate_steps"] else "",
        "observation": result["intermediate_steps"][0][1] if result["intermediate_steps"] else ""
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)