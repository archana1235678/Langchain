import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# 📂 Paths
pdf_folder = r"D:\transformer_project\data\pdfs"        # your PDFs
chroma_db = r"D:\transformer_project\data\chromadb"     # vector DB storage

@st.cache_resource
def load_qa():
    # 1. Load PDFs
    docs = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            docs.extend(loader.load())
    if not docs:
        raise ValueError("❌ No PDF documents found in the folder.")

    # 2. Split documents
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Build/load Chroma DB
    if not os.path.exists(chroma_db):
        db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_db)
        db.persist()
    else:
        db = Chroma(persist_directory=chroma_db, embedding_function=embeddings)

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

# 🚀 Load QA system
qa = load_qa()

# 🎨 Streamlit UI
st.set_page_config(page_title="📘 PDF Chatbot", layout="wide")
st.title("📘 LangChain PDF Chatbot (Streamlit + FLAN-T5-SMALL)")
st.write("Ask questions based on your uploaded PDFs.")

# Input box
query = st.text_input("💬 Your Question:", "")

if st.button("Ask"):
    if query.strip():
        response = qa.invoke({"query": query})
        st.success(response["result"])
    else:
        st.warning("Please type a question.")
