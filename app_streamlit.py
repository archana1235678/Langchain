import os
import streamlit as st
import tempfile
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# Temporary directory for uploaded files + Chroma DB
temp_dir = tempfile.mkdtemp()
chroma_db = os.path.join(temp_dir, "chromadb")

@st.cache_resource
def load_qa(uploaded_files):
    # 1. Save uploaded PDFs locally in temp_dir
    docs = []
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    if not docs:
        raise ValueError("❌ No PDF documents uploaded.")

    # 2. Split documents
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Build Chroma DB
    db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_db)
    db.persist()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 5. Load model
    qa_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=qa_model)

    # 6. Prompt
    prompt_template = """Use the following context to answer the question concisely.
If the answer is not in the context, just say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 7. Build RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa


# 🎨 Streamlit UI
st.set_page_config(page_title="📘 PDF Chatbot", layout="wide")
st.title("📘 LangChain PDF Chatbot (Streamlit + FLAN-T5-SMALL)")
st.write("Upload PDFs and ask questions based on their content.")

# Upload PDFs
uploaded_files = st.file_uploader("📂 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    qa = load_qa(uploaded_files)

    query = st.text_input("💬 Your Question:", "")

    if st.button("Ask"):
        if query.strip():
            response = qa.invoke({"query": query})
            st.success(response["result"])
        else:
            st.warning("⚠️ Please type a question.")

