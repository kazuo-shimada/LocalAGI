# LocalAGI: Private RAG System

A completely local, privacy-focused AI Chatbot capable of Retrieval Augmented Generation (RAG). It runs locally on your machine without Docker or internet APIs.

## 📋 Prerequisites
* Python 3.10 or higher
* Git

## 🛠️ Installation

1.  **Clone this repository**
    ```bash
    git clone <YOUR_REPO_URL_HERE>
    cd LocalAGI
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Model Setup
Because the models are large, they are not included in this repository. 
**Create a folder named `models`** inside the root directory and download these 3 files into it:

1.  **The Chat Model (Qwen2-VL):**
    * [Download Link](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GGUF/resolve/main/qwen2-vl-7b-instruct-q4_k_m.gguf?download=true)
    * Save as: `Qwen2-VL-7B-Instruct-Q4_K_M.gguf`

2.  **The Vision Adapter (mmproj):**
    * [Download Link](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GGUF/resolve/main/mmproj-qwen2-vl-7b-instruct-f16.gguf?download=true)
    * Save as: `mmproj-Qwen2-VL-7B-Instruct-f16.gguf`

3.  **The Embedding Model (Nomic):**
    * [Download Link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf?download=true)
    * Save as: `nomic-embed-text-v1.5.Q4_K_M.gguf`

## 🚀 Running the App

1.  Ensure your virtual environment is active (`source venv/bin/activate`).
2.  Run the application:
    ```bash
    python3 app.py
    ```
3.  Open your browser to: **http://localhost:7860**

## 🤖 Usage
1.  **Load Models:** Click "Load Models" to initialize the AI.
2.  **Ingest Data:** Go to the "Knowledge Base" tab, upload PDFs/TXT files, and click "Ingest".
3.  **Chat:** Go to the "Chat" tab and ask questions about your documents.