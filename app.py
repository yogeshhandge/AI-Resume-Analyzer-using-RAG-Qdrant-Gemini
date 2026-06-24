# ============================
# Import Required Libraries
# ============================

import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI

# ============================
# Load Environment Variables
# ============================

load_dotenv()

# ============================
# Streamlit UI
# ============================

st.title("📄 AI Resume Analyzer (RAG + Qdrant)")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

# ============================
# Analyze Button
# ============================

if st.button("Analyze Resume"):

    if uploaded_file is None:

        st.error("Please upload a resume.")
        st.stop()

    if not job_description.strip():

        st.error("Please paste a Job Description.")
        st.stop()

    # ============================
    # Save Uploaded Resume
    # ============================

    os.makedirs("resumes", exist_ok=True)

    file_path = f"resumes/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ============================
    # Extract Resume Text
    # ============================

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    resume_text = ""

    for doc in documents:
        resume_text += doc.page_content + "\n"

    # ============================
    # Chunk Resume
    # ============================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.create_documents(
        [resume_text]
    )

    st.success(
        f"Resume Split Into {len(chunks)} Chunks"
    )

    # ============================
    # Create Embeddings
    # ============================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ============================
    # Create Local Qdrant Database
    # ============================

    client = QdrantClient(
        ":memory:"
    )

    # ============================
    # Store Chunks in Qdrant
    # ============================

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path="./qdrant_db",
        collection_name="resume_collection"
    )

    # ============================
    # Retrieve Relevant Resume Sections
    # ============================

    results = vector_store.similarity_search(
        job_description,
        k=5
    )

    # ============================
    # Build Context
    # ============================

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    # ============================
    # Gemini Model
    # ============================

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite"
    )

    # ============================
    # ATS Prompt
    # ============================

    prompt = f"""
    You are an expert ATS system.

    Resume Context:
    {context}

    Job Description:
    {job_description}

    Provide:

    1. ATS Match Score (%)
    2. Matching Skills
    3. Missing Skills
    4. Strengths
    5. Weaknesses
    6. Improvement Suggestions

    Return the output in a professional format.
    """

    # ============================
    # Generate Response
    # ============================

    response = llm.invoke(prompt)

    # ============================
    # Display Analysis
    # ============================

    st.subheader("📊 ATS Analysis")

    st.write(response.content)

    # ============================
    # Show Retrieved Resume Sections
    # ============================

    st.subheader("📄 Retrieved Resume Sections")

    for i, doc in enumerate(results, start=1):

        with st.expander(
            f"Resume Section {i}"
        ):
            st.write(doc.page_content)