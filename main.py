import streamlit as st
import os
import tempfile
os.environ["USER_AGENT"] = "genai-rag-app/1.0"

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, YoutubeLoader # type: ignore
from langchain_ollama import OllamaEmbeddings, ChatOllama# type: ignore
from langchain_community.vectorstores import FAISS# type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter# type: ignore
from langchain_core.prompts import ChatPromptTemplate# type: ignore


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    return splitter.split_documents(docs)


def embed_and_vector_store(chunks):
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store


rag_prompt = ChatPromptTemplate.from_template(
    
    """You are a helpful assistant. Answer the question using the context below.
Use only the information from the context. If the context is insufficient,
give the best possible answer based on it.

Context:
{context}

Question:
{question}

Answer:"""
)


llm = ChatOllama(
    
    model="llama3.2:1b",
    temperature=0.2
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_answer(vector_store, question, k=3):
    
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    context = format_docs(docs)
    
    messages = rag_prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)
    return response.content, docs


# Streamlit UI PART

st.set_page_config(page_title="RAG Q&A App", layout="wide")

st.title("📄 RAG-based Document Q&A")
st.caption("PDF or URL → FAISS → Local LLM (Ollama)")

source_type = st.radio(
    "Choose data source:",
    ["PDF", "URL", "YouTube"]
)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ------------------------
# PDF FLOW
# ------------------------
if source_type == "PDF":
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

    if uploaded_file and st.button("Build Knowledge Base"):
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            chunks = split_documents(docs)
            st.session_state.vector_store = embed_and_vector_store(chunks)

            st.success("PDF processed and indexed successfully!")


# ------------------------
# URL FLOW (SINGLE URL ONLY)
# ------------------------
if source_type == "URL":
    url = st.text_input("Enter a single URL (static pages recommended)")

    if url and st.button("Build Knowledge Base"):
        with st.spinner("Processing URL..."):
            loader = WebBaseLoader(web_paths=(url,))
            docs = loader.load()

            chunks = split_documents(docs)
            st.session_state.vector_store = embed_and_vector_store(chunks)

            st.success("URL content processed and indexed successfully!")


# ------------------------
# YOUTUBE FLOW
# ------------------------
if source_type == "YouTube":
    youtube_url = st.text_input("Enter a YouTube video URL")

    if youtube_url and st.button("Build Knowledge Base"):
        with st.spinner("Processing YouTube video..."):
            try:
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url,
                    add_video_info=False,
                    language=["en", "hi"]
                )
                docs = loader.load()

                chunks = split_documents(docs)
                st.session_state.vector_store = embed_and_vector_store(chunks)

                st.success(
                    "YouTube transcript processed and indexed successfully!"
                )

            except Exception as e:
                st.error(
                    "Failed to load YouTube transcript. "
                    "This video may not have captions, "
                    "or YouTube blocked access."
                )
                st.exception(e)



# ------------------------
# QUESTION ANSWERING
# ------------------------
st.divider()

question = st.text_input("Ask a question based on the uploaded content")

if question and st.session_state.vector_store:
    with st.spinner("Generating answer..."):
        answer, sources = rag_answer(
            st.session_state.vector_store,
            question
        )

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        for i, doc in enumerate(sources):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content[:500])
