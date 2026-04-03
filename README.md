# Rag-Based-Chatbot
Strict Healthcare PDF Chatbot

A high-precision Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **FAISS**. This bot is designed for strict information retrieval, ensuring it only answers questions based on the provided healthcare documentation.

## Features
* **Zero-Hallucination Policy:** Uses a similarity threshold to ensure the bot says "I don't know" if the answer isn't in the PDF.
* **Precise Text Extraction:** Powered by `PyMuPDF` (fitz) for handling complex medical layouts.
* **Local Embeddings:** Uses `sentence-transformers` for efficient, private text vectorization.
* **Adjustable Strictness:** Integrated UI slider to control how closely the answer must match the source text.

## How to Run Locally

1. Clone the repo:

```bash
git clone https://github.com/haarikaalla/Rag-Based-Chatbot.git
cd Rag-Based-Chatbot
pip install -r requirements.txt

link: https://rag-based-chatbot-8goosyktc6cwyfqwd2jse2.streamlit.app/
