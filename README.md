# 🤖 LocalAGI: Autonomous RAG Agent

**LocalAGI** is a privacy-first, completely offline RAG (Retrieval-Augmented Generation) system designed to run on consumer hardware (Mac/PC). 

Unlike standard "Chat with PDF" tools, LocalAGI uses a **Hybrid Search Architecture** and **Agentic Query Rewriting** to solve complex positional queries (e.g., *"Who is the first person on the list?"*) that traditional vector databases fail to answer.

---

## 🚀 Key Features

### 🧠 Agentic Intelligence
* **Query Rewriting:** The agent intercepts vague user questions (e.g., *"What's #1?"*) and rewrites them into precise database queries (e.g., *"Identify the name listed at Line 1"*).
* **Self-Correction:** The system validates if documents were found before attempting to answer, reducing hallucinations.

### 🔍 Hybrid Search Engine (The "Magnet")
* **Vector Similarity:** Finds content based on meaning (Semantic Search).
* **Anchor Retrieval:** Automatically force-retrieves the start of documents using a "Magnet" header (`### START OF THE LIST ###`). This guarantees the model always sees critical context like introductions or list headers.

### 📝 Smart Ingestion
* **Line Number Stamping:** Automatically stamps `Line 1:`, `Line 2:` metadata onto every line of text files during ingestion.
* **Fragment Awareness:** The AI is trained to respect these line numbers, allowing it to understand order and position within a document.

### 🔒 100% Local & Private
* **No API Keys:** Runs entirely on `llama.cpp` and `ChromaDB`.
* **Privacy:** Your documents never leave your machine.
* **Model Agnostic:** Works with any GGUF model (Llama 3, Qwen 2, Mistral, etc.).

---

## 🛠️ Installation

### 1. Prerequisites
* Python 3.10+
* Git

### 2. Clone the Repository
```bash
git clone [https://github.com/kazuo-shimada/LocalAGI.git](https://github.com/kazuo-shimada/LocalAGI.git)
cd LocalAGI
