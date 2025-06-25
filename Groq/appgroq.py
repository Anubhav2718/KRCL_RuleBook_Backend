import streamlit as st
import os
from langchain_groq import ChatGroq  # This is the correct import
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']
FAISS_INDEX_PATH = "faiss_index_krcl"

if "vectors" not in st.session_state:
    # Use the SAME embedding model used to create the FAISS index
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(FAISS_INDEX_PATH):
        st.session_state.vectors = FAISS.load_local(
            FAISS_INDEX_PATH,
            st.session_state.embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        st.error("FAISS index not found. Please generate it first using this embedding model.")
        st.stop()


st.title("ChatGroq Demo")

llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
"""
You are a Railway Rule Assistant. Answer only using exact rule numbers or S.R. references from the provided context.
Do not assume or create rules. Be precise and rule-based.

<context>
{context}
</context>
Question: {input}
"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")