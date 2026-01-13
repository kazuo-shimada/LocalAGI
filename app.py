import gradio as gr
from llama_cpp import Llama
import os
import glob
import re
import base64
from PIL import Image
import io
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document 

# --- GLOBAL VARIABLES ---
CHAT_MODEL = None
EMBED_MODEL = None 
VECTOR_STORE = None 

# --- CONFIGURATION ---
RETRIEVAL_K = 5  # Reduced from 10 to save context window space

# --- HELPER: Custom Embedding Class ---
class LocalLlamaEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = Llama(model_path=model_path, embedding=True, verbose=False)
    def embed_documents(self, texts):
        return [self.model.create_embedding(t)['data'][0]['embedding'] for t in texts]
    def embed_query(self, text):
        return self.model.create_embedding(text)['data'][0]['embedding']

# --- HELPER: Smart Image Encoder (Prevents Crash) ---
def encode_image(image_path):
    if not image_path: return None
    
    # Open and Resize Image to save tokens
    with Image.open(image_path) as img:
        # Convert to RGB (fixes issues with PNG transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too big (Max 1024px dimension)
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
        
        # Save to buffer
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- HELPER: History Converter ---
def convert_history_to_ui(history_state):
    ui_messages = []
    for msg in history_state:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        ui_content = ""
        if isinstance(content, list):
            text_part = next((item['text'] for item in content if item['type'] == 'text'), "")
            ui_content = f"🖼️ [Image Analyzed] {text_part}"
        else:
            ui_content = str(content)
            
        ui_messages.append({"role": role, "content": ui_content})
    return ui_messages

def get_gguf_files():
    files_lower = glob.glob("models/*.gguf")
    files_upper = glob.glob("Models/*.gguf")
    all_files = files_lower + files_upper
    if not all_files: return ["No models found"]
    return sorted(list(set([os.path.basename(f) for f in all_files])))

def find_model_path(filename):
    path_lower = os.path.join("models", filename)
    if os.path.exists(path_lower): return os.path.abspath(path_lower)
    path_upper = os.path.join("Models", filename)
    if os.path.exists(path_upper): return os.path.abspath(path_upper)
    if os.path.exists(filename): return os.path.abspath(filename)
    return filename

def load_models(chat_model, vis_model, emb_model):
    global CHAT_MODEL, EMBED_MODEL, VECTOR_STORE
    
    chat_path = find_model_path(chat_model)
    vis_path = find_model_path(vis_model) if vis_model != "None" else None
    emb_path = find_model_path(emb_model) if emb_model != "None" else None

    chat_h = None
    if vis_path:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_h = Llava15ChatHandler(clip_model_path=vis_path)
    
    try:
        CHAT_MODEL = Llama(
            model_path=chat_path, 
            n_gpu_layers=33, 
            n_ctx=8192,  # Maximize context window
            chat_handler=chat_h, 
            verbose=False 
        )
    except Exception as e:
        return f"❌ Chat Load Error: {e}", gr.update(), gr.update()
    
    if emb_path:
        try:
            EMBED_MODEL = LocalLlamaEmbeddings(emb_path)
            VECTOR_STORE = Chroma(collection_name="docs", embedding_function=EMBED_MODEL, persist_directory="chroma_db")
            return f"✅ Loaded: {chat_model} + Vision", gr.update(interactive=True), gr.update(interactive=True)
        except Exception as e:
            return f"❌ Embed Load Error: {e}", gr.update(), gr.update()
    
    return f"✅ Loaded: {chat_model} (No RAG)", gr.update(interactive=True), gr.update(interactive=True)

# --- INGESTION ---
def ingest(files):
    if not VECTOR_STORE: return "⚠️ Load Embedding Model first!"
    if not files: return "No files provided"
    
    # Reduced chunk size slightly to leave room for images
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    docs = []
    for path in files:
        try:
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                content_block = "### START OF DOCUMENT / TOP OF PHYSICAL LIST ###\n"
                for i, line in enumerate(lines):
                    row_idx = i + 1
                    match = re.search(r'^(\d+)', line)
                    if match:
                        logical_num = match.group(1)
                        stamped_line = f"[Row {row_idx}] (Item #{logical_num}) {line}"
                    else:
                        stamped_line = f"[Row {row_idx}] {line}"
                    content_block += stamped_line + "\n"
                docs.append(Document(page_content=content_block, metadata={"source": path}))
            elif path.endswith(".pdf"): 
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
        except Exception as e:
            return f"Error loading {path}: {e}"
    if docs:
        VECTOR_STORE.add_documents(splitter.split_documents(docs))
        return f"✅ Ingested {len(docs)} chunks. (Dual Indexing Active)"
    return "No documents found."

# --- AGENT PIPELINE ---
def run_agent_pipeline(user_query, history_state, system_prompt, img_path=None):
    if not CHAT_MODEL:
        yield history_state, "⚠️ Model not loaded."
        return

    history_state = history_state or []
    
    # 1. ALWAYS Try Retrieval First
    final_context = ""
    search_query = user_query
    
    if VECTOR_STORE:
        yield history_state, "📚 Checking Documents..."
        # We skip LLM rewrite if image is present to avoid confusing the text model
        if not img_path:
            try:
                rewrite_prompt = f"Rewrite for search: '{user_query}'"
                rewrite_response = CHAT_MODEL.create_chat_completion(
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    max_tokens=64, temperature=0.1 
                )
                search_query = rewrite_response['choices'][0]['message']['content'].strip().replace('"', '')
            except:
                pass 

        results = VECTOR_STORE.similarity_search(search_query, k=RETRIEVAL_K)
        # We also grab the anchor, but only 1 to save space
        anchor_docs = VECTOR_STORE.similarity_search("START OF DOCUMENT TOP OF PHYSICAL LIST", k=1)
        combined_docs = anchor_docs + results
        
        seen = set()
        unique_docs = []
        for d in combined_docs:
            if d.page_content not in seen:
                unique_docs.append(d)
                seen.add(d.page_content)

        if unique_docs:
            final_context = "\n---\n".join([d.page_content for d in unique_docs])
        else:
            final_context = "NO_DATA_FOUND"

    # 2. Construct Prompt
    full_text_prompt = ""
    if final_context and final_context != "NO_DATA_FOUND":
        full_text_prompt = f"""### DOCUMENT CONTEXT:
{final_context}

### USER QUESTION:
{user_query}
"""
        sys_instruction = system_prompt + "\nNOTE: Use the provided Document Context AND/OR the Image to answer."
    else:
        full_text_prompt = user_query
        sys_instruction = system_prompt

    # 3. Handle Image Injection
    if img_path:
        yield history_state, "👁️ Analyzing Image + Text..."
        base64_img = encode_image(img_path)
        user_message_content = [
            {"type": "text", "text": full_text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ]
        history_state.append({"role": "user", "content": user_message_content})
    else:
        yield history_state, "🤔 Analyzing Text..."
        history_state.append({"role": "user", "content": full_text_prompt})

    history_state.append({"role": "assistant", "content": ""})
    
    messages = [{"role": "system", "content": sys_instruction}]
    
    # Strict context management to prevent -3 error
    # If image is present, we only keep the VERY LAST exchange + System prompt
    if img_path:
        # Image takes huge memory, so we drop older history to make room
        pass 
    elif len(history_state) > 4:
        valid_history = []
        for m in history_state[-4:-2]:
            if isinstance(m['content'], str):
                valid_history.append(m)
        messages.extend(valid_history)
    
    # Add final message
    if img_path:
        messages.append({"role": "user", "content": user_message_content})
    else:
        messages.append({"role": "user", "content": full_text_prompt})

    try:
        stream = CHAT_MODEL.create_chat_completion(messages=messages, stream=True, max_tokens=1024, temperature=0.2)
        partial_response = ""
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                partial_response += chunk["choices"][0]["delta"]["content"]
                history_state[-1]["content"] = partial_response
                yield history_state, f"✅ Processing..."
                
    except Exception as e:
        history_state[-1]["content"] = f"❌ Generation Error: {e}"
        yield history_state, "Error"

# --- WRAPPER ---
def chat_wrapper(msg, history_state, img, sys_box):
    pipeline = run_agent_pipeline(msg, history_state, sys_box, img_path=img)
    for updated_history, status_msg in pipeline:
        ui_messages = convert_history_to_ui(updated_history)
        yield ui_messages, status_msg, updated_history

# --- UI LAYOUT ---
with gr.Blocks(title="LocalAGI Agent") as demo:
    gr.Markdown("## 🤖 LocalAGI: Autonomous Agent")
    history_state = gr.State([]) 

    with gr.Row():
        with gr.Column(scale=1):
            load_btn = gr.Button("🔄 Load Agent", variant="primary")
            m_chat = gr.Dropdown(get_gguf_files(), value="Qwen2-VL-7B-Instruct-Q4_K_M.gguf", label="LLM")
            m_vis = gr.Dropdown(["None"] + get_gguf_files(), value="mmproj-Qwen2-VL-7B-Instruct-f16.gguf", label="Vision Adapter")
            m_emb = gr.Dropdown(["None"] + get_gguf_files(), value="nomic-embed-text-v1.5.Q4_K_M.gguf", label="Embedding")
            
            up = gr.File(file_count="multiple", label="Upload Data")
            ingest_btn = gr.Button("📂 Ingest")
            ingest_status = gr.Textbox(label="Log")
            
            with gr.Accordion("⚙️ System Prompt", open=False):
                default_sys = "You are a helpful Data Analyst."
                sys_box = gr.Textbox(value=default_sys, lines=4)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, label="Live Interaction")
            with gr.Row():
                msg = gr.Textbox(label="Mission Objective", placeholder="Ask a question...", scale=4)
                send_btn = gr.Button("Execute", variant="primary", scale=1)
            with gr.Row():
                img = gr.Image(type="filepath", height=100, label="Vision Input")
                src = gr.Textbox(label="Agent Status", interactive=False)

    load_btn.click(load_models, [m_chat, m_vis, m_emb], [src, send_btn, ingest_btn])
    ingest_btn.click(ingest, up, ingest_status)
    msg.submit(chat_wrapper, [msg, history_state, img, sys_box], [chatbot, src, history_state])
    send_btn.click(chat_wrapper, [msg, history_state, img, sys_box], [chatbot, src, history_state])

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    if not os.path.exists("chroma_db"): os.makedirs("chroma_db")
    demo.launch(server_port=7868)