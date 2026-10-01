````markdown
# 📄 AI Document Assistant

A simple Streamlit AI Document Assistant that lets users upload documents, search their contents, and ask questions using Groq.

The application supports:

- PDF
- DOCX
- TXT
- Markdown
- Google Drive files and folders

The application uses document extraction, text chunking, Sentence Transformers embeddings, FAISS semantic search, keyword search, hybrid ranking, and Groq.

---

## Features

### 1. Local Document Upload

Upload multiple documents at the same time.

Supported formats:

- PDF
- DOCX
- TXT
- MD

The application extracts the document text.

For PDF files, page numbers are preserved.

For DOCX, TXT, and MD files, page numbers are shown as unavailable because these formats do not normally contain reliable page boundaries.

---

## 2. Document Extraction

Separate extraction functions are used for each format:

```text
extract_pdf()
extract_docx()
extract_txt()
extract_md()
````

Each extracted document keeps:

```text
filename
page
text
```

Example:

```text
Filename: Artificial_Intelligence.pdf
Page: 1
Text: Artificial intelligence is...
```

---

## 3. Text Chunking

Large documents are divided into smaller overlapping chunks.

The default settings are:

```text
Chunk size: 800 words
Overlap: 150 words
```

The settings can be changed from the sidebar.

Every chunk keeps:

```text
text
filename
page
```

The application displays the total number of created chunks.

---

## 4. Sentence Transformers

The application uses:

```text
all-MiniLM-L6-v2
```

Each document chunk is converted into an embedding.

The embedding model is cached using:

```python
@st.cache_resource
```

This prevents the model from being loaded repeatedly.

---

## 5. FAISS Semantic Search

FAISS is used for vector similarity search.

When a document is processed:

```text
Document
    ↓
Text
    ↓
Chunks
    ↓
Embeddings
    ↓
FAISS Index
```

When the user asks a question:

```text
Question
    ↓
Question Embedding
    ↓
FAISS Search
    ↓
Relevant Chunks
```

---

## 6. Keyword Search

The application also performs a simple keyword search.

Important words are extracted from the user's question.

Common stop words such as:

```text
the
is
a
what
how
and
of
```

are ignored.

The application checks how many important question words occur in each document chunk.

---

## 7. Hybrid Search

The application combines semantic search and keyword search.

The hybrid score is:

```text
75% Semantic Similarity
+
25% Keyword Matching
```

The highest-ranked chunks are returned.

The original metadata is preserved.

Each result contains:

```text
Filename
Page
Text
Semantic score
Keyword score
Hybrid score
```

---

## 8. Groq

The retrieved document chunks are sent to Groq together with the user's question.

The model is instructed to answer only from the supplied document context.

The application uses:

```text
openai/gpt-oss-120b
```

The model is instructed:

```text
Do not use outside knowledge.
Do not invent information.
Do not guess.
```

If the information cannot be found in the retrieved context, the application asks the model to respond:

```text
The information is not available in the provided documents.
```

---

## 9. Retrieved Sources

After every answer, the application displays the retrieved sources.

Each source shows:

```text
Filename
Page number
Hybrid score
Retrieved text chunk
```

For example:

```text
1. Artificial_Intelligence.pdf
Page: 1
Score: 0.842

Artificial intelligence is the field of...
```

This makes it easier to understand where the answer came from.

---

# Google Drive

The application also supports Google Drive.

Paste a Google Drive file or folder link into the Google Drive input.

Example:

```text
https://drive.google.com/...
```

The application can load:

```text
PDF
DOCX
TXT
MD
```

files.

The Google Drive files are sent through the same processing pipeline as local files.

```text
Google Drive
     ↓
Download
     ↓
Extraction
     ↓
Chunking
     ↓
Embeddings
     ↓
FAISS
     ↓
Hybrid Search
     ↓
Groq
```

The Drive files must be accessible through the shared link.

---

# Processing Optimization

The application is designed so that documents are processed once.

The workflow is:

```text
Upload documents
       ↓
Extract text
       ↓
Create chunks
       ↓
Create embeddings
       ↓
Build FAISS index
       ↓
Store in Streamlit session state
```

Questions then reuse the existing:

```text
Chunks
Embeddings
FAISS index
Metadata
```

The application does not recreate all document embeddings for every question.

The Sentence Transformer model is also cached using:

```python
@st.cache_resource
```

---

# Project Structure

```text
ai-document-assistant/
│
├── app.py
├── requirements.txt
└── readme.md
```

---

# Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# Groq API Key

Never put your Groq API key directly inside `app.py`.

For local development, create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

For Streamlit Cloud:

1. Open your Streamlit app.
2. Open the app settings.
3. Open Secrets.
4. Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

The application reads the key using:

```python
st.secrets["GROQ_API_KEY"]
```

The key should never be uploaded to GitHub.

---

# GitHub

Upload these files to your GitHub repository:

```text
app.py
requirements.txt
readme.md
```

Do not upload:

```text
.streamlit/secrets.toml
```

A `.gitignore` file can contain:

```text
.streamlit/secrets.toml
__pycache__/
*.pyc
```

---

# Streamlit Cloud

After uploading the project to GitHub:

1. Open Streamlit Community Cloud.
2. Create a new app.
3. Select your GitHub repository.
4. Select `app.py`.
5. Deploy the application.
6. Add `GROQ_API_KEY` in Streamlit Secrets.

---

# Complete Pipeline

The complete application pipeline is:

```text
PDF / DOCX / TXT / MD
          +
      Google Drive
          ↓
   Document Extraction
          ↓
      Text Chunking
          ↓
Sentence Transformer
          ↓
      Embeddings
          ↓
       FAISS Index
          ↓
     User Question
          ↓
    Question Embedding
          ↓
     Semantic Search
          +
      Keyword Search
          ↓
      Hybrid Ranking
          ↓
    Relevant Chunks
          ↓
          Groq
          ↓
        Answer
          ↓
   Retrieved Sources
```

---

# Dummy Test Documents

You can test the application with documents about:

* Artificial Intelligence
* Web Development
* Cybersecurity
* Data Science
* Cloud Computing
* Database Systems
* Python Programming
* Project Management
* Digital Marketing
* Study Skills

Example questions:

```text
What is artificial intelligence?
```

```text
What is machine learning?
```

```text
What does frontend development use?
```

```text
What are common cybersecurity threats?
```

```text
What is a relational database?
```

```text
What is Python used for?
```

```text
What are the benefits of cloud computing?
```

The retrieved sources should show the relevant PDF filename and page number.

---

# Important Limitations

This is a simple educational implementation.

PDF files with normal selectable text work best.

Scanned/image-only PDFs require OCR, which is not included.

DOCX, TXT, and MD files do not have reliable page boundaries, so their page number is displayed as unavailable.

Google Drive files must be accessible through their shared link.

For a larger production system, you could later add:

* Persistent FAISS indexes
* Persistent metadata storage
* OCR
* Google Drive OAuth
* User authentication
* Document deletion
* Conversation history
* Better keyword ranking
* Multiple embedding models
* Streaming Groq responses

```
```
