
---

#  RAG-Based Intelligent Conversational AI Chatbot using LangChain, FAISS, Groq LLM & Tavily Search

---

##  Overview

This project demonstrates an **end-to-end Retrieval-Augmented Generation (RAG)** pipeline that integrates **local knowledge retrieval** from **MongoDB-related PDFs** with **real-time web intelligence** using **Tavily Search API** and **Groq Llama 3.1 LLM**.

It functions as an **intelligent chatbot** capable of:

* Answering **MongoDB-specific queries** from embedded local documents.
* Handling **unrelated or new topics** using **live web data**.
* Providing summarized, human-like responses through **Generative AI**.

The chatbot was built using:

*  **LangChain** for modular pipeline management.
*  **SentenceTransformer** for embeddings.
*  **FAISS** as a high-performance local vector database.
*  **Groq Llama 3.1 (8B)** for fast, efficient text generation.
*  **Tavily Search API** for online knowledge retrieval.
*  **Streamlit** for an interactive, real-time user interface.

---

##  Step-by-Step Workflow Explanation

### **1️) Uploading the PDF Document**

Placed **MongoDB-related PDFs** inside the `/data` directory.

 **File Supported:** `.pdf`, `.txt`, `.csv`, `.xlsx`, `.docx`, `.json`
 **File Loader:** `data_loader.py` uses **LangChain community document loaders** such as:

* `PyPDFLoader` for PDF files
* `TextLoader` for TXT files
* `UnstructuredExcelLoader` for Excel
* `Docx2txtLoader` for Word documents

 **Function:**
Each file is parsed and converted into LangChain **Document objects** containing:

```python
{
  "page_content": "Extracted text content",
  "metadata": {"source": "file_name.pdf", "page": 1}
}
```

 **Purpose:** Converts unstructured data into a standardized format ready for chunking and embedding.

---

### **2) Text Extraction**

The loaded PDFs are processed page-by-page to extract **raw text**.
This ensures every piece of MongoDB documentation — from replication to sharding — is readable by the model.

 **Libraries Used:** `PyPDFLoader` (LangChain wrapper around `pdfminer.six`)

 **In `data_loader.py`:**

```python
loader = PyPDFLoader("data/mongodb_guide.pdf")
documents = loader.load()
```

---

### **3) Text Chunking**

Once the text is extracted, it’s split into **smaller overlapping chunks** for better retrieval and contextual understanding.

 **Module:** `embeddings.py`
 **Parameters Used:**

* `chunk_size = 1000`
* `chunk_overlap = 200`
* **Splitter:** `RecursiveCharacterTextSplitter`

 **Why Chunking Matters:**
LLMs can only handle limited tokens at a time — chunking ensures:

* No loss of context.
* Efficient embeddings.
* Better retrieval accuracy.
---

### **4) Embeddings Generation**

Each text chunk is converted into **semantic vector embeddings** using:

*  **Model:** `SentenceTransformer("all-MiniLM-L6-v2")`

This model transforms text into 384-dimensional numeric vectors that capture semantic similarity.

 **Code Reference (embeddings.py):**

```python
self.model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = self.model.encode(texts, show_progress_bar=True)
```

 **Output:** Numpy array of shape `(num_chunks, 384)`

 *These embeddings allow the system to “understand” meaning, not just keywords.*

---

### **5️) Vector Store using FAISS**

The generated embeddings are stored in **FAISS (Facebook AI Similarity Search)** — a local vector database that enables **fast semantic search**.

 **Implementation:** `vectorstore.py`

**FAISS Index Files:**

* `faiss.index` → Stores numeric vectors
* `metadata.pkl` → Stores mapping info (chunk → text → source)

 **How it works:**

* When a user queries, the system converts their question into an embedding.
* FAISS compares it with stored vectors.
* It retrieves the most similar chunks based on **cosine similarity / L2 distance**.

 **Benefits:**

* Works offline.
* Extremely fast retrieval.
* Persistent between sessions.

---

### **6) Conversational Retrieval & LLM Summarization**

**Core Component:** `RAGSearch` class in `search.py`

 **Process:**

1. Query is embedded using the same SentenceTransformer.
2. Top 3–5 matching chunks retrieved from FAISS.
3. Chunks merged into context.
4. Context passed into a prompt template for the **Groq Llama 3.1 LLM**.
5. LLM generates a **concise, context-based summary.**

 **Prompt Template Example:**

```python
f"""
Summarize the following context for the query: '{query}'.

Context:
{context}

Summary:
"""
```

 **If context is found →** Use FAISS results.
 **If context is not found →** Invoke Tavily Search API.


---

### **7️) Web Search Integration using Tavily API**

When the chatbot doesn’t find an answer in local PDFs, it uses **Tavily Search API** to fetch fresh, web-based information.

 **Library:** `tavily`
 **Used in:** `search.py → RAGSearch.web_search()`

**Process:**

* Sends query to Tavily.
* Fetches 3–5 top search results.
* Extracts “title” and “content” fields.
* Passes them as additional context to the Groq LLM.

 **Example:**

```python
web_results = self.tavily_client.search(query)
content = " ".join([f"title: {r['title']} | content: {r['content']}" for r in web_results])
```


---

### **8) Streamlit Chat Interface**

Finally, the results are displayed in a **Streamlit-based chatbot UI**.

 **File:** `ui.py`
 **Features:**

* Maintains chat history (`st.session_state.messages`)
* Real-time streaming output effect (`st.write_stream`)
* Responsive and minimalistic interface.

 **User Experience Flow:**

1. User types a question →
2. Query sent to `RAGSearch.search_and_summarize()` →
3. Answer displayed word-by-word →
4. Chat history retained for context continuity.

---
##  User Interface & Output Screenshots

Below are the screenshots demonstrating how the chatbot works across different query contexts.

---

### 1) Initial Interface — Streamlit UI

The chatbot launches with a sleek dark-themed Streamlit interface, awaiting user input.

![alt text](images/Interface.png)
---

###  2) Local Knowledge Query (MongoDB Context)

When the user asks a MongoDB-related question, the chatbot retrieves the answer **from the local FAISS Vector Store** (PDF embeddings).

**Example Query:**  
> What are the features of MongoDB?

**Response:**  
The system fetches relevant chunks from locally stored MongoDB PDFs and summarizes them via **Groq Llama 3.1**.

![alt text](images/local1.png)
![alt text](images/Local2.png)
---

###  3) Web-Based Query (Non-Local Context)

When a query is unrelated to MongoDB, the local search returns no result.  
In this case, the **Tavily Search API** is triggered to fetch real-time information from the web.

**Example Query:**  
> Who are the champions of IPL 2025?

**Response:**  
The chatbot retrieves and summarizes the latest web data, generating an accurate and human-like summary.

![
    
](images/Web.png)


---
##  RAG Flow Summary

| Stage             | Component                        | Technology Used   | Output             |
| ----------------- | -------------------------------- | ----------------- | ------------------ |
| PDF Upload        | `data_loader.py`                 | LangChain Loaders | Documents          |
| Text Extraction   | `PyPDFLoader`                    | pdfminer.six      | Text               |
| Chunking          | `RecursiveCharacterTextSplitter` | LangChain         | Text Chunks        |
| Embeddings        | `SentenceTransformer`            | all-MiniLM-L6-v2  | 384D Vectors       |
| Vector Store      | `FAISS`                          | Local Index       | Top Similar Chunks |
| LLM Summarization | `Groq Llama 3.1`                 | ChatGroq          | Contextual Answers |
| Web Search        | `Tavily API`                     | HTTP Retrieval    | Web Context        |
| UI                | `Streamlit`                      | Frontend          | Chat Interface     |

---
