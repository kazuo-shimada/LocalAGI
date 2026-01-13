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

```

### 3. Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
pip install --upgrade gradio  # Requires Gradio 5.0+

```

---

## 📂 Model Setup

This project uses **GGUF** models. You must download them manually (e.g., from [HuggingFace](https://huggingface.co/models)) and place them in the `models/` folder.

**Recommended Setup:**

1. **Chat Model:** `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (or Llama 3)
2. **Embedding Model:** `nomic-embed-text-v1.5.Q4_K_M.gguf`
3. **Vision Adapter (Optional):** `mmproj-model-f16.gguf`

Your folder structure should look like this:

```text
LocalAGI/
├── models/
│   ├── Qwen2-VL-7B.gguf
│   └── nomic-embed.gguf
├── chroma_db/       # Created automatically
├── app.py
└── requirements.txt

```

---

## 🖥️ Usage

1. **Start the Server:**
```bash
python app.py

```


2. **Open in Browser:**
Go to `http://localhost:7860`.
3. **Workflow:**
* **Load Models:** Select your GGUF files and click "Load Agent".
* **Ingest Data:** Upload `.txt` or `.pdf` files. (Text files will be automatically numbered!).
* **Chat:** Ask questions like *"Who is the first name on the list?"* or *"Summarize the document."*



---

## 🧩 Technical Architecture

### The Pipeline

1. **User Input:** "What is #1?"
2. **Rewriter (LLM):** Converts to -> *"Identify the item associated with Line 1."*
3. **Hybrid Retrieval (ChromaDB):**
* *Branch A:* Searches for the rewritten query.
* *Branch B:* Searches for the "Magnet" (`START OF THE LIST`).


4. **Context Construction:** Combines results + System Prompt with "Line Number Trust" rules.
5. **Generation:** The LLM generates a reasoned answer starting with `THOUGHT:`.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)