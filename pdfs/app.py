import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

st.title("Healthcare PDF Chatbot")
st.write("Ask questions strictly based on the PDF content. Answers come only from the PDF.")

# ✅ Use cache_resource for unserializable objects
@st.cache_resource(show_spinner=True)
def load_pdf():
    loader = PyPDFLoader("pdfs/healthcare.pdf")
    documents = loader.load()
    st.write(f"Loaded docs: {len(documents)}")
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    st.write(f"Split docs: {len(docs)}")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(docs, embeddings)
    
    llm = OllamaLLM(model="phi")
    
    return db, llm

db, llm = load_pdf()

system_prompt = """
You are a strict assistant.
Answer ONLY using the information provided in the Context below.
Do NOT provide explanations, reasoning, or any extra information.
Do NOT use any outside knowledge.
If the answer is not in the Context, reply exactly: 'Not found in document'.
"""

query = st.text_input("Type your question here:")

if query:
    results = db.similarity_search(query, k=1)
    if not results:
        st.write("Bot: Not found in document")
    else:
        context = results[0].page_content
        prompt = f"""
{system_prompt}

Context:
{context}

Question: {query}
"""
        response = llm.invoke(prompt).strip()
        st.write("Bot:", response)