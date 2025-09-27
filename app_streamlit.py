import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Function to process uploaded PDFs
def load_docs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

st.title("📄 Chat with your PDF (Streamlit + LangChain)")

# Upload PDFs
uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        # Extract text
        raw_text = load_docs(uploaded_files)

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)

        # Embeddings + Vector DB
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings)

        # QA Chain
        retriever = vectorstore.as_retriever()
        llm = HuggingFaceHub(repo_id="google/flan-t5-small")  # Example model
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ PDFs processed successfully!")

        # Chat interface
        query = st.text_input("Ask a question about your documents:")
        if query:
            response = qa.run(query)
            st.write("**Answer:**", response)

