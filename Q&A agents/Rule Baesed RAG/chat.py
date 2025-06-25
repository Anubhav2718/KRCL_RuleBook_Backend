import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.documents import Document

# ---- Page Config ----
st.set_page_config(page_title="KRCL GnSR Chat", page_icon="Konkan_Railway_logo.svg.png")
st.image("Konkan_Railway_logo.svg.png", width=50)
st.title("Chat with KRCL G&SR 2020 + Accident Manual")
st.markdown("Ask any question about railway safety and operations based on the official rules.")

# ---- Load Vector Store ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db1 = FAISS.load_local("faiss_index_krcl", embeddings, allow_dangerous_deserialization=True)
db2 = FAISS.load_local("faiss_manual_krcl", embeddings, allow_dangerous_deserialization=True)

retriever1 = db1.as_retriever(search_kwargs={"k": 4})
retriever2 = db2.as_retriever(search_kwargs={"k": 4})

# ---- Load LLM ----
llm = Ollama(model="mistral")

# ---- Prompt Template ----
prompt = ChatPromptTemplate.from_template("""
You are an expert assistant answering questions about railway operations and safety based on two sources:
1. *KRCL General and Subsidiary Rules 2020* (GnSR)
2. *KRCL Accident Manual*

Only answer from the given context. Do not guess or fabricate rules. If the answer cannot be found, say:
**"The information is not available in the provided rules."**

**Answer Format Instructions:**
- Include the **exact rule number or section heading** (if available).
- Mention the **source** clearly: *(GnSR)* or *(Accident Manual)*.
- Provide a **summary or quote** accurately.
- If multiple sources are relevant, list them separately.

<context>
{context}
</context>

Question: {input}

Answer:
""")

# ---- Combine Documents Chain ----
document_chain = create_stuff_documents_chain(llm, prompt)

# ---- Combined Retriever ----
def combined_retriever(query: str) -> list[Document]:
    docs1 = retriever1.get_relevant_documents(query)
    for d in docs1:
        d.metadata["source_name"] = "GnSR"
    docs2 = retriever2.get_relevant_documents(query)
    for d in docs2:
        d.metadata["source_name"] = "Accident Manual"
    return docs1 + docs2

retrieval_chain = RunnableMap({
    "context": RunnableLambda(lambda x: combined_retriever(x["input"])),  # <- FIXED
    "input": lambda x: x["input"]
}) | document_chain

# ---- UI ----
user_question = st.text_input(" Ask your question:",placeholder="e.g., What is the duty of a station master during an emergency?")
if user_question:
    with st.spinner("Fetching rule-based answer..."):
        response = retrieval_chain.invoke({"input": user_question})
        
        # Show the main answer
        st.markdown("###  Answer:")
        st.write(response)
        
        # Optional: Show retrieved context documents (sources)
        with st.expander("Show Retrieved Rules (Context)"):
            docs = combined_retriever(user_question)
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**{i}. Source:** {doc.metadata.get('source_name', 'Unknown')}")
                st.markdown(f"> {doc.page_content.strip()}")
