import streamlit as st
from langchain_community.llms import Ollama
from langchain.agents import Tool
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
import re

# --- UI Setup ---
st.set_page_config(page_title="KRCL RuleBot", page_icon="Konkan_Railway_logo.svg.png")
st.image("Konkan_Railway_logo.svg.png", width=50)
st.title("KRCL G&SR 2020 Assistant")

# --- Load Vector DB ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index_krcl", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# --- Wiki Tool ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- Retrieval Tool ---
def krcl_retriever_tool(query: str) -> str:
    results = retriever.get_relevant_documents(query)
    if not results:
        return "The information is not available in the provided rules."
    return "\n\n".join([doc.page_content for doc in results])

# --- LLM and Tools ---
llm = Ollama(
    model="mistral",
    temperature=0,
    system="You are a safety-critical railway rule expert. Never guess."
)

tools = [
    Tool(
        name="General and Subsidary Rules",
        func=krcl_retriever_tool,
        description="Retrieves accurate rule-based answers from the KRCL G&SR 2020 manual."
    )
]

# --- Agent Setup ---
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True  # Needed to access tool trace
)

# --- Input ---
query = st.text_input("Enter your query about KRCL Rules")

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

            # --- Display Action and Observation (from first tool call only) ---
            if steps:
                action, observation = steps[0]
                st.markdown(f"**Action:** {action.tool}")
                st.markdown(f"**Observation:** {observation}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
