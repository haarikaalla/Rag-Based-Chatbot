import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------ UI ------------------
st.set_page_config(page_title="Strict PDF Chatbot", layout="wide")
st.header("🛡️ Strict Healthcare PDF Chatbot")

with st.sidebar:
    st.title("Control Panel")
    file = st.file_uploader("Upload Healthcare PDF", type=["pdf"])
    # 1.4 is a good balance for finding synonyms without guessing
    threshold = st.slider("Strictness (Lower = More Strict)", 0.5, 2.0, 1.4)
    if st.button("Reset Bot"):
        st.cache_resource.clear()
        st.rerun()

# ------------------ Process PDF ------------------
@st.cache_resource
def process_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    if not text.strip():
        return None

    # VERY small chunks (150) to separate "Fever" from "Cold"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, 
        chunk_overlap=0,
        separators=["\n", ". "]
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# ------------------ Main Bot Logic ------------------
if file is not None:
    file_bytes = file.read()
    vector_store = process_pdf(file_bytes)
    
    if vector_store:
        st.sidebar.success("✅ PDF Loaded")
        query = st.text_input("Ask a question (e.g., 'What is Fever?'):")

        if query:
            # Search for the best match
            results = vector_store.similarity_search_with_score(query, k=1)
            
            if results:
                best_doc, score = results[0]

                if score < threshold:
                    # EXTRA FILTER: Only show sentences that match the query keyword
                    content = best_doc.page_content.replace('\n', ' ')
                    sentences = content.split('. ')
                    
                    # Filter sentences to only show the ones related to the query
                    filtered_sentences = [s for s in sentences if query.lower() in s.lower() or any(word in s.lower() for word in query.lower().split())]
                    
                    st.subheader("Answer from PDF:")
                    if filtered_sentences:
                        # Join matching sentences back together
                        final_answer = ". ".join(filtered_sentences).strip()
                        if not final_answer.endswith('.'): final_answer += "."
                        st.success(final_answer)
                    else:
                        # Fallback if keyword match is tricky but similarity is high
                        st.success(content.strip())
                        
                    with st.expander("Confidence Metrics"):
                        st.write(f"Match Distance: {round(score, 4)}")
                else:
                    st.warning("Bot: Information not found in document.")
            else:
                st.error("No matches found.")
    else:
        st.error("Could not read PDF text.")
else:
    st.info("Please upload your healthcare.pdf to start.")