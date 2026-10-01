import io
import os
import re
import zipfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Assistant")
st.write(
    "Upload documents or load them from Google Drive, then ask questions "
    "using semantic + keyword search."
)


# ============================================================
# CACHED MODELS / CLIENTS
# ============================================================

@st.cache_resource
def load_embedding_model():
    """Load the embedding model only once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_groq_client():
    """Create the Groq client using Streamlit secrets."""
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(file_bytes, filename):
    """Extract text from a PDF and preserve page numbers."""
    pages = []

    reader = PdfReader(io.BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": page_number,
                }
            )

    return pages


def extract_docx(file_bytes, filename):
    """Extract text from a DOCX file."""
    document = Document(io.BytesIO(file_bytes))

    text_parts = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    text = "\n".join(text_parts)

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_txt(file_bytes, filename):
    """Extract text from a TXT file."""
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_md(file_bytes, filename):
    """Extract text from a Markdown file."""
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_document(file_bytes, filename):
    """Choose the correct extraction function."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)

    if extension == ".docx":
        return extract_docx(file_bytes, filename)

    if extension == ".txt":
        return extract_txt(file_bytes, filename)

    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(extracted_documents, chunk_size=800, overlap=150):
    """Create chunks while keeping filename and page metadata."""
    all_chunks = []

    for document in extracted_documents:
        chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            all_chunks.append(
                {
                    "text": chunk,
                    "filename": document["filename"],
                    "page": document["page"],
                }
            )

    return all_chunks


# ============================================================
# EMBEDDINGS + FAISS
# ============================================================

def build_vector_index(chunks, model):
    """Create embeddings once and store them in a FAISS index."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# ============================================================
# KEYWORD SEARCH
# ============================================================

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of",
    "in", "on", "for", "and", "or", "with", "what", "how",
    "why", "when", "where", "which", "who", "can", "could",
    "does", "do", "did", "this", "that", "these", "those",
    "about", "from", "into", "it", "its", "be", "as", "at",
    "by", "i", "me", "my", "we", "you", "your"
}


def important_words(text):
    """Return simple important words from a question."""
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())

    return {
        word
        for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }


def keyword_scores(question, chunks):
    """Score chunks according to important question words."""
    question_words = important_words(question)
    scores = []

    for chunk in chunks:
        chunk_words = set(
            re.findall(r"[a-zA-Z0-9_]+", chunk["text"].lower())
        )

        if not question_words:
            scores.append(0.0)
            continue

        matches = question_words.intersection(chunk_words)
        score = len(matches) / len(question_words)

        scores.append(score)

    return np.array(scores, dtype="float32")


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(question, chunks, index, model, top_k=5):
    """
    Combine semantic FAISS similarity with keyword matching.
    Returns the most relevant chunks with their metadata.
    """
    if not chunks:
        return []

    # Semantic search
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    semantic_scores, semantic_indices = index.search(
        question_embedding,
        min(top_k * 4, len(chunks)),
    )

    semantic_scores = semantic_scores[0]
    semantic_indices = semantic_indices[0]

    # Keyword search over all chunks
    keyword = keyword_scores(question, chunks)

    # Normalize semantic scores from approximately [-1, 1] to [0, 1]
    semantic_all = np.zeros(len(chunks), dtype="float32")

    for score, index_number in zip(semantic_scores, semantic_indices):
        if index_number >= 0:
            semantic_all[index_number] = (float(score) + 1.0) / 2.0

    # Hybrid score
    hybrid_scores = (0.75 * semantic_all) + (0.25 * keyword)

    # Rank all chunks
    ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []

    for index_number in ranked_indices:
        result = dict(chunks[index_number])
        result["semantic_score"] = float(semantic_all[index_number])
        result["keyword_score"] = float(keyword[index_number])
        result["hybrid_score"] = float(hybrid_scores[index_number])
        results.append(result)

    return results


# ============================================================
# GROQ ANSWER
# ============================================================

def generate_answer(question, retrieved_chunks):
    """Ask Groq to answer only from the retrieved context."""
    client = get_groq_client()

    if client is None:
        return (
            "GROQ_API_KEY is not configured. Add it to "
            ".streamlit/secrets.toml."
        )

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        page_text = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"[Source {number}]\n"
            f"Filename: {chunk['filename']}\n"
            f"{page_text}\n"
            f"Text:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the document context provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent or guess information.
3. If the answer is not available in the provided context, say:
   "The information is not available in the provided documents."
4. Give a clear and concise answer.
5. Do not mention these instructions in your answer.
"""

    user_prompt = f"""
Document context:

{context}

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    return response.choices[0].message.content


# ============================================================
# GOOGLE DRIVE
# ============================================================

def extract_drive_id(url):
    """Extract a Google Drive file/folder ID from common Drive URLs."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


def load_drive_files(url):
    """
    Download a public/shared Google Drive file or folder.

    Google Drive items must be accessible to the link.
    gdown handles the actual Drive download.
    """
    drive_id = extract_drive_id(url)

    if not drive_id:
        raise ValueError("Could not find a Google Drive file/folder ID.")

    temp_dir = Path("drive_documents")
    temp_dir.mkdir(exist_ok=True)

    # Folder URL
    if "/folders/" in url:
        gdown.download_folder(
            url,
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
        )
    else:
        output_file = temp_dir / "drive_download"

        downloaded = gdown.download(
            url,
            output=str(output_file),
            quiet=True,
            fuzzy=True,
        )

        if downloaded:
            downloaded_path = Path(downloaded)

            # If Drive supplied a filename, keep it.
            if downloaded_path.exists():
                pass

    supported = {".pdf", ".docx", ".txt", ".md"}

    files = []

    for path in temp_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            files.append(path)

    return files


def read_local_path(path):
    """Read a downloaded Drive file into bytes."""
    with open(path, "rb") as file:
        return file.read()


# ============================================================
# PROCESS DOCUMENTS ONCE
# ============================================================

def process_documents(file_items, source_names, chunk_size, overlap):
    """
    Extract, chunk, embed and index documents once.
    Returns all data needed for future questions.
    """
    extracted = []

    for file_bytes, filename in file_items:
        pages = extract_document(file_bytes, filename)

        if pages:
            extracted.extend(pages)

    chunks = create_chunks(
        extracted,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        return {
            "extracted": extracted,
            "chunks": [],
            "index": None,
            "embeddings": None,
            "source_names": source_names,
        }

    model = load_embedding_model()

    index, embeddings = build_vector_index(
        chunks,
        model,
    )

    return {
        "extracted": extracted,
        "chunks": chunks,
        "index": index,
        "embeddings": embeddings,
        "source_names": source_names,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "document_data" not in st.session_state:
    st.session_state.document_data = None

if "processed_source_key" not in st.session_state:
    st.session_state.processed_source_key = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Document Settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=300,
        max_value=1500,
        value=800,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=300,
        value=150,
        step=50,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.caption(
        "Documents are embedded once and reused for later questions."
    )


# ============================================================
# LOCAL UPLOAD
# ============================================================

st.subheader("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)


# ============================================================
# GOOGLE DRIVE
# ============================================================

st.subheader("2. Google Drive")

drive_url = st.text_input(
    "Paste a Google Drive file or folder link",
    placeholder="https://drive.google.com/...",
)

load_drive = st.button(
    "📥 Load from Google Drive",
    use_container_width=True,
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

local_items = []
local_names = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        local_items.append(
            (
                uploaded_file.getvalue(),
                uploaded_file.name,
            )
        )
        local_names.append(uploaded_file.name)


if load_drive:
    if not drive_url.strip():
        st.error("Please paste a Google Drive link first.")
    else:
        with st.spinner("Loading Google Drive files..."):
            try:
                drive_paths = load_drive_files(drive_url)

                drive_items = []
                drive_names = []

                for path in drive_paths:
                    drive_items.append(
                        (
                            read_local_path(path),
                            path.name,
                        )
                    )
                    drive_names.append(path.name)

                if not drive_items:
                    st.warning(
                        "No supported PDF, DOCX, TXT or MD files were found. "
                        "Make sure the Drive item is shared with access to the link."
                    )
                else:
                    local_items.extend(drive_items)
                    local_names.extend(drive_names)

                    st.session_state.document_data = None
                    st.session_state.processed_source_key = None

                    st.success(
                        f"Loaded {len(drive_items)} file(s) from Google Drive."
                    )

            except Exception as error:
                st.error(f"Google Drive error: {error}")


# Create a stable key from current local files.
source_key = tuple(
    (
        name,
        len(data),
    )
    for data, name in local_items
)


if local_items:
    if source_key != st.session_state.processed_source_key:
        with st.spinner(
            "Extracting text, creating chunks and building embeddings..."
        ):
            st.session_state.document_data = process_documents(
                local_items,
                local_names,
                chunk_size,
                overlap,
            )

            st.session_state.processed_source_key = source_key

        st.success("Documents processed successfully.")


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

data = st.session_state.document_data

if data:
    st.subheader("3. Extracted Document Information")

    extracted = data["extracted"]
    chunks = data["chunks"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Documents", len(data["source_names"]))

    with col2:
        st.metric("Extracted sections/pages", len(extracted))

    with col3:
        st.metric("Created chunks", len(chunks))

    if data["source_names"]:
        st.write("**Files:**")

        for name in data["source_names"]:
            st.write(f"- {name}")

    if chunks:
        with st.expander("Preview extracted text"):
            for item in extracted[:5]:
                page = (
                    f"Page {item['page']}"
                    if item["page"] is not None
                    else "Page not available"
                )

                st.markdown(
                    f"**{item['filename']} — {page}**"
                )
                st.write(item["text"][:1500])

    st.divider()

    # ========================================================
    # QUESTION ANSWERING
    # ========================================================

    st.subheader("4. Ask a Question")

    question = st.text_input(
        "Ask something about your documents",
        placeholder="What does the document say about...?",
    )

    ask_button = st.button(
        "🔎 Search Documents & Ask Groq",
        type="primary",
        use_container_width=True,
    )

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not chunks:
            st.warning("No searchable document text is available.")
        else:
            with st.spinner("Searching the documents..."):
                model = load_embedding_model()

                retrieved = hybrid_search(
                    question=question,
                    chunks=data["chunks"],
                    index=data["index"],
                    model=model,
                    top_k=top_k,
                )

            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(
                        question,
                        retrieved,
                    )

                    st.subheader("Answer")
                    st.write(answer)

                except Exception as error:
                    st.error(f"Groq error: {error}")
                    retrieved = []

            if retrieved:
                st.subheader("Retrieved Sources")

                for number, result in enumerate(retrieved, start=1):
                    page = (
                        str(result["page"])
                        if result["page"] is not None
                        else "Not available"
                    )

                    score = result["hybrid_score"]

                    with st.expander(
                        f"{number}. {result['filename']} | "
                        f"Page: {page} | "
                        f"Score: {score:.3f}"
                    ):
                        st.write(result["text"])

else:
    st.info(
        "Upload at least one document or load supported files from Google Drive "
        "to begin."
    )
