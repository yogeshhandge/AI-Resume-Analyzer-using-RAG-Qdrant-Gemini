# AI Resume Analyzer (RAG + Qdrant)

An AI-powered Resume Analyzer built using:

- Streamlit
- LangChain
- Google Gemini
- Qdrant
- HuggingFace Embeddings

## Features

✅ Upload Resume PDF

✅ Compare Resume with Job Description

✅ ATS Match Score

✅ Missing Skills Detection

✅ Improvement Suggestions

✅ Retrieval Augmented Generation (RAG)

## Tech Stack

- Python
- Streamlit
- LangChain
- Gemini 2.5 Flash Lite
- Qdrant
- Sentence Transformers

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Architecture

Resume PDF
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
Qdrant
↓
Retriever
↓
Gemini
↓
ATS Analysis