import streamlit as st
from langchain_community.llms import Ollama
from langchain.agents import Tool
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, AgentType

# --- UI Setup ---
st.set_page_config(page_title="KRCL RuleBot", page_icon="Konkan_Railway_logo.svg.png")
st.image("Konkan_Railway_logo.svg.png", width=50)
st.title("KRCL G&SR 2020 Assistant")

# --- Load Embeddings & Vector DBs ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db1 = FAISS.load_local("faiss_index_krcl", embeddings, allow_dangerous_deserialization=True)
retriever1 = db1.as_retriever(search_kwargs={"k": 4})

db2 = FAISS.load_local("faiss_manual_krcl", embeddings, allow_dangerous_deserialization=True)
retriever2 = db2.as_retriever(search_kwargs={"k": 4})

# --- Custom Retrieval Functions ---
def gsr_retriever_tool(query: str) -> str:
    results = retriever1.get_relevant_documents(query)
    if not results:
        return "No relevant rules found in G&SR."
    return "\n\n".join([doc.page_content for doc in results])

def manual_retriever_tool(query: str) -> str:
    results = retriever2.get_relevant_documents(query)
    if not results:
        return "No relevant content found in the KRCL Manual."
    return "\n\n".join([doc.page_content for doc in results])

def combined_retriever_tool(query: str) -> str:
    results1 = retriever1.get_relevant_documents(query)
    results2 = retriever2.get_relevant_documents(query)

    combined = results1[:4] + results2[:8]  # 0.5 weight each (top 2 from each)
    
    if not combined:
        return "No relevant content found in either source."

    return "\n\n".join([doc.page_content for doc in combined])

# --- LLM Setup ---
llm = Ollama(
    model="mistral",
    temperature=0,
    system="You are a safety-critical railway rule expert. Never guess."
)

# --- Tools ---
tools = [
    Tool(
        name="General and Subsidiary Rules",
        func=gsr_retriever_tool,
        description="Retrieves accurate rule-based answers from the KRCL G&SR 2020 manual."
    ),
    Tool(
        name="Accidental Manual",
        func=manual_retriever_tool,
        description="Fetches relevant content from the KRCL departmental/manual documentation."
    ),
    Tool(
        name="Combined",
        func=combined_retriever_tool,
        description="Blended retrieval from both KRCL rules and manual (50-50 weight)."
    )
]

# --- Agent Setup ---
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# --- Input UI ---
query = st.text_input("Enter your query about KRCL Rules or Manual")

if query:
    with st.spinner("Thinking..."):
        try:
            response = agent_executor.invoke({
                "input": query,
                "chat_history": []
            })

            final_output = response.get("output", "")
            steps = response.get("intermediate_steps", [])

            # --- Display Final Answer ---
            st.markdown(f"**Final Answer:** {final_output.strip()}")

            # --- Display First Action and Observation ---
            if steps:
                action, observation = steps[0]
                st.markdown(f"**Action:** {action.tool}")
                st.markdown(f"**Observation:** {observation}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
