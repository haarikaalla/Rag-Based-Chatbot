# pdf_chatbot_strict_streamlit.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

st.title("Healthcare PDF Chatbot")
st.write("Ask questions strictly based on the PDF content. Answers come only from the PDF.")

# ---------------------------
# Load PDF, split, create FAISS and LLM
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_pdf():
    loader = PyPDFLoader("pdfs/healthcare.pdf")
    documents = loader.load()
    st.write(f"Loaded docs: {len(documents)}")
    
    # Split text into chunks (with overlap to not miss information)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    st.write(f"Split docs: {len(docs)}")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(docs, embeddings)
    
    # Load LLM
    llm = OllamaLLM(model="phi")
    
    return db, llm

db, llm = load_pdf()

# ---------------------------
# System prompt - STRICT PDF only
# ---------------------------
system_prompt = """
You are a strict assistant.
Answer ONLY using the information provided in the Context below.
Include **everything from the Context exactly as it appears**.
Do NOT summarize, interpret, or add any extra information.
Do NOT use outside knowledge.
If the Context does not contain the answer, reply exactly: 'Not found in document'.
"""

# ---------------------------
# Query input
# ---------------------------
query = st.text_input("Type your question here:")

if query:
    # Retrieve top 3 relevant chunks to ensure all details are captured
    results = db.similarity_search(query, k=3)
    
    if not results:
        st.write("Bot: Not found in document")
    else:
        # Combine all relevant chunks into one context
        context = "\n".join([doc.page_content for doc in results])
        
        # Build the prompt for LLM
        prompt = f"""
{system_prompt}

Context:
{context}

Question: {query}
"""
        # Call LLM
        response = llm.invoke(prompt).strip()
        st.write("Bot:", response)