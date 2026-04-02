# Rag-Based-Chatbot


This is a Streamlit-based chatbot that answers questions strictly from a PDF document (`healthcare.pdf`) using LangChain, FAISS, and Ollama LLM.

## How to Run Locally

1. Clone the repo:

```bash
git clone https://github.com/haarikaalla/Rag-Based-Chatbot.git
cd Rag-Based-Chatbot
pip install -r requirements.txt
python -m streamlit run pdfs/app.py
