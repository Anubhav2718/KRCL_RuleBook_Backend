# chat_gnsr.py
import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---- Page Config ----
st.set_page_config(page_title="KRCL GnSR Chat", page_icon="Konkan_Railway_logo.svg.png")
st.image("Konkan_Railway_logo.svg.png", width=50)
st.title(" Chat with KRCL G&SR 2020")
st.markdown("Ask any question about railway safety and operations based on the official rules.")

# ---- Load Vector Store ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index_krcl", embeddings, allow_dangerous_deserialization=True)


retriever = db.as_retriever()

# ---- Load LLM ----
llm = Ollama(model="mistral")

# ---- Prompt Template ----
prompt = ChatPromptTemplate.from_template("""
You are an expert assistant answering questions about railway operations and safety rules from the *KRCL General and Subsidiary Rules 2020* manual.

Your task is to provide accurate, section-based answers **only** from the provided context. Do not guess. Do not invent rules. If the answer cannot be found in the context, clearly state: "The information is not available in the provided rules."

Warning: Your response may be used by railway staff in real operational scenarios. Wrong or misleading answers may lead to **accidents or disciplinary action**. Respond carefully and precisely.

Instructions:
- Identify the correct rule number and section from the context.
- Provide the **exact wording** or an accurate summary.
- Format the rule reference like this: *Rule 2.03 — Knowledge of Rules*.
- If multiple rules apply, cite each.
- If the answer is not in the context, **say so clearly**.

<context>
{context}
</context>

Question: {input}

Answer (with rule reference):
""")

# ---- Chain ----
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ---- UI ----
user_question = st.text_input(" Ask your question:", placeholder="e.g., What is the duty of a station master during an emergency?")
if user_question:
    with st.spinner("Fetching rule-based answer..."):
        response = retrieval_chain.invoke({"input": user_question})
        st.markdown("###  Answer:")
        st.write(response["answer"])
