import gradio as gr
from llama_cpp import Llama
import os
import glob
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings

# --- GLOBAL VARIABLES ---
CHAT_MODEL = None
EMBED_MODEL = None 
VECTOR_STORE = None 

# --- HELPER: Custom Embedding Class ---
class LocalLlamaEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = Llama(model_path=model_path, embedding=True, verbose=False)
    def embed_documents(self, texts):
        return [self.model.create_embedding(t)['data'][0]['embedding'] for t in texts]
    def embed_query(self, text):
        return self.model.create_embedding(text)['data'][0]['embedding']

def get_gguf_files():
    return [os.path.basename(f) for f in glob.glob("models/*.gguf")]

def load_models(chat, vis, emb):
    global CHAT_MODEL, EMBED_MODEL, VECTOR_STORE
    
    chat_h = None
    if vis and vis != "None":
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_h = Llava15ChatHandler(clip_model_path=f"models/{vis}")
    
    CHAT_MODEL = Llama(model_path=f"models/{chat}", n_gpu_layers=33, n_ctx=2048, chat_handler=chat_h, verbose=True)
    
    if emb and emb != "None":
        EMBED_MODEL = LocalLlamaEmbeddings(f"models/{emb}")
        VECTOR_STORE = Chroma(collection_name="docs", embedding_function=EMBED_MODEL, persist_directory="chroma_db")
        return "✅ Models Loaded (Chat + RAG)", gr.update(interactive=True), gr.update(interactive=True)
    
    return "✅ Chat Model Loaded (No RAG)", gr.update(interactive=True), gr.update(interactive=True)

def ingest(files):
    if not VECTOR_STORE: return "⚠️ Load Embedding Model first!"
    if not files: return "No files provided"
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for path in files:
        if path.endswith(".pdf"): loader = PyPDFLoader(path)
        else: loader = TextLoader(path)
        VECTOR_STORE.add_documents(splitter.split_documents(loader.load()))
    return "✅ Ingested successfully!"

def chat(msg, history, img, sys):
    if not CHAT_MODEL: yield history, ""; return
    
    prompt = msg
    sources = "*No sources*"
    
    # RAG Logic
    if VECTOR_STORE and not img:
        res = VECTOR_STORE.similarity_search(msg, k=3)
        if res:
            ctx = "\n".join([d.page_content for d in res])
            sources = "\n".join([f"- {d.metadata.get('source')}" for d in res])
            prompt = f"Use this context to answer:\n{ctx}\n\nQuestion: {msg}"

    # --- KEY FIX: Use DICTIONARIES for History (Gradio 5/New 4 Style) ---
    history = history or []
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": ""}) 
    
    msgs = [{"role": "system", "content": sys}] if sys else []
    
    if img:
        import base64
        with open(img, "rb") as f: b64 = base64.b64encode(f.read()).decode("utf-8")
        content = [{"type":"text","text":prompt}, {"type":"image_url","image_url": {"url":f"data:image/jpeg;base64,{b64}"}}]
        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user", "content": prompt})

    stream = CHAT_MODEL.create_chat_completion(messages=msgs, stream=True)
    partial = ""
    for chunk in stream:
        if "content" in chunk["choices"][0]["delta"]:
            partial += chunk["choices"][0]["delta"]["content"]
            history[-1]["content"] = partial # Update last dict in list
            yield history, sources

with gr.Blocks(title="LocalAGI") as demo:
    gr.Markdown("## LocalAGI (New Gradio Compatible)")
    
    with gr.Row():
        load_btn = gr.Button("Load Models", variant="primary")
        m_chat = gr.Dropdown(get_gguf_files(), value="Qwen2-VL-7B-Instruct-Q4_K_M.gguf", label="Chat Model")
        m_vis = gr.Dropdown(["None"] + get_gguf_files(), value="mmproj-Qwen2-VL-7B-Instruct-f16.gguf", label="Vision Adapter")
        m_emb = gr.Dropdown(["None"] + get_gguf_files(), value="nomic-embed-text-v1.5.Q4_K_M.gguf", label="Embedding Model")

    # KEY FIX: No 'type' argument, but data will be dicts
    chatbot = gr.Chatbot(height=400) 
    msg = gr.Textbox(label="Message")
    src = gr.Textbox(label="Sources Used")
    
    with gr.Row():
        img = gr.Image(type="filepath", height=100)
        send_btn = gr.Button("Send")

    with gr.Accordion("Knowledge Base", open=False):
        up = gr.File(file_count="multiple")
        ingest_btn = gr.Button("Ingest Documents")
        ingest_status = gr.Textbox(label="Status")
    
    load_btn.click(load_models, [m_chat, m_vis, m_emb], [ingest_status, send_btn, ingest_btn])
    ingest_btn.click(ingest, up, ingest_status)
    
    # KEY FIX: Gradio 5 handles 'type="messages"' implicitly if you pass dicts
    send_btn.click(chat, [msg, chatbot, img, gr.State("")], [chatbot, src])

if __name__ == "__main__":
    demo.launch(server_port=7860)
