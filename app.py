import gradio as gr
from llama_cpp import Llama
import os
import glob
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document 
import base64

# --- GLOBAL VARIABLES ---
CHAT_MODEL = None
EMBED_MODEL = None 
VECTOR_STORE = None 

# --- CONFIGURATION ---
RETRIEVAL_K = 10 

# --- HELPER: Custom Embedding Class ---
class LocalLlamaEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = Llama(model_path=model_path, embedding=True, verbose=False)
    def embed_documents(self, texts):
        return [self.model.create_embedding(t)['data'][0]['embedding'] for t in texts]
    def embed_query(self, text):
        return self.model.create_embedding(text)['data'][0]['embedding']

def get_gguf_files():
    files = glob.glob("models/*.gguf")
    if not files: return ["No models found in /models"]
    return [os.path.basename(f) for f in files]

def load_models(chat_model, vis_model, emb_model):
    global CHAT_MODEL, EMBED_MODEL, VECTOR_STORE
    
    chat_h = None
    if vis_model and vis_model != "None":
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_h = Llava15ChatHandler(clip_model_path=f"models/{vis_model}")
    
    try:
        CHAT_MODEL = Llama(
            model_path=f"models/{chat_model}", 
            n_gpu_layers=33, 
            n_ctx=8192,  
            chat_handler=chat_h, 
            verbose=False 
        )
    except Exception as e:
        return f"❌ Chat Load Error: {e}", gr.update(), gr.update()
    
    if emb_model and emb_model != "None":
        try:
            EMBED_MODEL = LocalLlamaEmbeddings(f"models/{emb_model}")
            VECTOR_STORE = Chroma(collection_name="docs", embedding_function=EMBED_MODEL, persist_directory="chroma_db")
            return "✅ Models Loaded (Ready)", gr.update(interactive=True), gr.update(interactive=True)
        except Exception as e:
            return f"❌ Embed Load Error: {e}", gr.update(), gr.update()
    
    return "✅ Chat Only (No RAG)", gr.update(interactive=True), gr.update(interactive=True)

# --- INGESTION (The "Magnet" stays!) ---
def ingest(files):
    if not VECTOR_STORE: return "⚠️ Load Embedding Model first!"
    if not files: return "No files provided"
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    
    docs = []
    for path in files:
        try:
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # The "Magnet" Header
                numbered_content = "### START OF THE LIST / BEGINNING OF DOCUMENT ###\n"
                for i, line in enumerate(lines):
                    numbered_content += f"Line {i+1}: {line}"
                
                docs.append(Document(page_content=numbered_content, metadata={"source": path}))
            
            elif path.endswith(".pdf"): 
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
                
        except Exception as e:
            return f"Error loading {path}: {e}"
            
    if docs:
        VECTOR_STORE.add_documents(splitter.split_documents(docs))
        return f"✅ Ingested {len(docs)} chunks. (Magnet Active)"
    return "No documents found."

# --- AGENT PIPELINE (Loop Fix) ---
def run_agent_pipeline(user_query, history_state, system_prompt):
    if not CHAT_MODEL:
        yield history_state, "⚠️ Model not loaded."
        return

    history_state = history_state or []
    history_state.append({"role": "user", "content": user_query})
    
    yield history_state, "🤔 Analyzing..." 
    
    final_context = ""
    search_query = user_query

    if VECTOR_STORE:
        # 1. Rewrite (Keep this, it's working)
        rewrite_prompt = f"Rewrite to be specific: '{user_query}'"
        try:
            rewrite_response = CHAT_MODEL.create_chat_completion(
                messages=[{"role": "user", "content": rewrite_prompt}],
                max_tokens=64, temperature=0.1 
            )
            search_query = rewrite_response['choices'][0]['message']['content'].strip().replace('"', '')
        except:
            pass 

        # 2. Retrieve (Magnet + Query)
        results = VECTOR_STORE.similarity_search(search_query, k=RETRIEVAL_K)
        anchor_docs = VECTOR_STORE.similarity_search("START OF THE LIST BEGINNING OF DOCUMENT", k=1)
        
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

    # 3. GENERATE (The Fix is Here)
    if final_context == "NO_DATA_FOUND":
        sys_instruction = "You are a helpful assistant. Admit you found no documents."
        final_prompt = user_query
    elif final_context:
        # FIX: Move instructions to System Prompt ONLY. 
        # Do not repeat them in the User Prompt.
        sys_instruction = system_prompt + "\n\nCRITICAL RULE: The text uses explicit line numbers (Line 1: ...). You MUST trust these numbers to determine the order."
        
        # Clean Final Prompt: Just Context + Question
        final_prompt = f"""### CONTEXT (From Database):
{final_context}

### USER QUESTION:
{user_query}
"""
    else:
        sys_instruction = system_prompt
        final_prompt = user_query

    yield history_state, f"🔍 Searched: '{search_query}'"
    
    history_state.append({"role": "assistant", "content": ""})
    
    messages = [{"role": "system", "content": sys_instruction}]
    
    if len(history_state) > 4:
        valid_history = [m for m in history_state if isinstance(m, dict)]
        messages.extend(valid_history[-4:-2])
    
    messages.append({"role": "user", "content": final_prompt})

    stream = CHAT_MODEL.create_chat_completion(messages=messages, stream=True, max_tokens=1024)
    
    partial_response = ""
    for chunk in stream:
        if "content" in chunk["choices"][0]["delta"]:
            partial_response += chunk["choices"][0]["delta"]["content"]
            history_state[-1]["content"] = partial_response
            yield history_state, f"✅ Used: {search_query}"

# --- WRAPPER ---
def chat_wrapper(msg, history_state, img, sys_box):
    pipeline = run_agent_pipeline(msg, history_state, sys_box)
    for updated_history, status_msg in pipeline:
        ui_messages = [m for m in updated_history if m.get('role') != 'system']
        yield ui_messages, status_msg, updated_history

# --- UI LAYOUT ---
with gr.Blocks(title="LocalAGI Agent") as demo:
    gr.Markdown("## 🤖 LocalAGI: Autonomous Agent")
    
    history_state = gr.State([]) 

    with gr.Row():
        with gr.Column(scale=1):
            load_btn = gr.Button("🔄 Load Agent", variant="primary")
            m_chat = gr.Dropdown(get_gguf_files(), value="Qwen2-VL-7B-Instruct-Q4_K_M.gguf", label="LLM")
            m_vis = gr.Dropdown(["None"] + get_gguf_files(), value="mmproj-Qwen2-VL-7B-Instruct-f16.gguf", label="Vision")
            m_emb = gr.Dropdown(["None"] + get_gguf_files(), value="nomic-embed-text-v1.5.Q4_K_M.gguf", label="Embedding")
            
            up = gr.File(file_count="multiple", label="Upload Data")
            ingest_btn = gr.Button("📂 Ingest")
            ingest_status = gr.Textbox(label="Log")
            
            with gr.Accordion("⚙️ System Prompt", open=False):
                # Simplified System Prompt to reduce confusion
                default_sys = """You are a helpful Data Analyst.
1. Answer based ONLY on the provided Context.
2. Start your answer with "THOUGHT:" to briefly explain which line number you found the answer on.
3. Then provide the final "ANSWER:"."""
                sys_box = gr.Textbox(value=default_sys, lines=4)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, label="Live Interaction")
            
            with gr.Row():
                msg = gr.Textbox(label="Mission Objective", placeholder="Ask a question...", scale=4)
                send_btn = gr.Button("Execute", variant="primary", scale=1)
            
            with gr.Row():
                img = gr.Image(type="filepath", height=80, container=False)
                src = gr.Textbox(label="Agent Status", interactive=False)

    load_btn.click(load_models, [m_chat, m_vis, m_emb], [ingest_status, send_btn, ingest_btn])
    ingest_btn.click(ingest, up, ingest_status)
    
    msg.submit(chat_wrapper, [msg, history_state, img, sys_box], [chatbot, src, history_state])
    send_btn.click(chat_wrapper, [msg, history_state, img, sys_box], [chatbot, src, history_state])

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    if not os.path.exists("chroma_db"): os.makedirs("chroma_db")
    demo.launch(server_port=7860)