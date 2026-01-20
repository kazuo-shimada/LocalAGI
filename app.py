import gradio as gr
from llama_cpp import Llama
import os
import glob
import re
import base64
from PIL import Image
import io
import shutil
from langchain_chroma import Chroma
# We don't need the smart splitter anymore; we are doing it manually
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document 

# --- CONFIGURATION ---
RETRIEVAL_K = 5  # Keep this low to avoid fetching noise

# --- MEMORY WIPE STARTUP ---
if os.path.exists("chroma_db"):
    try:
        shutil.rmtree("chroma_db")
        print("🧹 Startup: Old Memory Wiped.")
    except:
        pass
os.makedirs("chroma_db", exist_ok=True)

# --- HELPER CLASSES ---
class LocalLlamaEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = Llama(model_path=model_path, embedding=True, verbose=False)
    def embed_documents(self, texts):
        return [self.model.create_embedding(t)['data'][0]['embedding'] for t in texts]
    def embed_query(self, text):
        return self.model.create_embedding(text)['data'][0]['embedding']

def encode_image(image_path):
    if not image_path: return None
    with Image.open(image_path) as img:
        if img.mode != 'RGB': img = img.convert('RGB')
        max_size = 1024
        if max(img.size) > max_size: img.thumbnail((max_size, max_size))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

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
    files = []
    for ext in ["/*.gguf", "/*.GGUF"]:
        files.extend(glob.glob("models" + ext))
        files.extend(glob.glob("Models" + ext))
    if not files: return []
    return sorted(list(set([os.path.basename(f) for f in files])))

def find_model_path(filename):
    path_lower = os.path.join("models", filename)
    if os.path.exists(path_lower): return os.path.abspath(path_lower)
    path_upper = os.path.join("Models", filename)
    if os.path.exists(path_upper): return os.path.abspath(path_upper)
    if os.path.exists(filename): return os.path.abspath(filename)
    return filename

def clean_vision_output(raw_text):
    hallucination_triggers = ["Question", "Answer", "label scanner", "list ONLY", "Examples", "do not describe"]
    items = re.split(r'[,\n]', raw_text)
    clean_items = []
    for item in items:
        item = item.strip()
        if not item: continue
        if any(trigger in item for trigger in hallucination_triggers): continue
        clean_items.append(item)
    return ", ".join(list(set(clean_items)))

def clean_final_response(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = []
    forbidden_starts = [
        "You are a", "Use ONLY", "If the detected", "If no match", 
        "Note:", "Instruction:", "Source Material", "User Question",
        "Do NOT", "Match the", "Identify the", "Step 1", "Step 2"
    ]
    for s in sentences:
        s_clean = s.strip()
        if not s_clean: continue
        is_bad = False
        for bad in forbidden_starts:
            if s_clean.startswith(bad):
                is_bad = True
                break
        if not is_bad:
            clean_sentences.append(s_clean)
    return " ".join(clean_sentences)

# --- STRICT PYTHON FILTER ---
def smart_filter_context(raw_docs, detected_inventory):
    """
    Strictly removes irrelevant chunks.
    """
    if not detected_inventory or detected_inventory == "Unknown Bottle":
        return raw_docs
        
    inv_lower = detected_inventory.lower()
    
    # Identify the User's Spirit
    target_spirit = None
    if "tequila" in inv_lower or "julio" in inv_lower or "patron" in inv_lower or "fortaleza" in inv_lower: target_spirit = "tequila"
    elif "gin" in inv_lower or "bombay" in inv_lower or "tanqueray" in inv_lower: target_spirit = "gin"
    elif "vodka" in inv_lower or "absolut" in inv_lower or "grey goose" in inv_lower: target_spirit = "vodka"
    elif "rum" in inv_lower or "bacardi" in inv_lower: target_spirit = "rum"
    elif "whiskey" in inv_lower or "bourbon" in inv_lower: target_spirit = "whiskey"

    if not target_spirit:
        return raw_docs # No strict match, return all

    filtered_docs = []
    print(f"\n--- FILTERING FOR: {target_spirit.upper()} ---")
    
    for doc in raw_docs:
        text = doc.page_content.lower()
        
        # KEY LOGIC: The chunk MUST contain the target spirit.
        if target_spirit in text:
            print(f"✅ KEEP: {doc.page_content[:30]}...")
            filtered_docs.append(doc)
        else:
            print(f"❌ DROP: {doc.page_content[:30]}... (Missing '{target_spirit}')")

    if not filtered_docs:
        print("⚠️ Warning: No docs matched. Using fallback.")
        return raw_docs
        
    return filtered_docs

# --- GLOBAL VARIABLES ---
CHAT_MODEL = None
EMBED_MODEL = None 
VECTOR_STORE = None 

def load_models(chat_model, vis_model, emb_model):
    global CHAT_MODEL, EMBED_MODEL, VECTOR_STORE
    
    chat_path = find_model_path(chat_model)
    vis_path = find_model_path(vis_model) if vis_model != "None" else None
    emb_path = find_model_path(emb_model) if emb_model != "None" else None

    chat_h = None
    if vis_path:
        print(f"⚙️ Loading Vision Handler for {vis_model}...")
        try:
            from llama_cpp.llama_chat_format import Llava16ChatHandler
            chat_h = Llava16ChatHandler(clip_model_path=vis_path)
            print("✅ Loaded Llava16ChatHandler")
        except:
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_h = Llava15ChatHandler(clip_model_path=vis_path)
                print("⚠️ Loaded Llava15ChatHandler (Fallback)")
            except Exception as e:
                print(f"❌ Vision Handler Error: {e}")

    try:
        CHAT_MODEL = Llama(
            model_path=chat_path, 
            n_gpu_layers=33, 
            n_ctx=8192, 
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

# --- INGESTION (MANUAL HARD SLICING) ---
def ingest(files):
    global VECTOR_STORE
    if not EMBED_MODEL: return "⚠️ Load Embedding Model first!"
    if not files: return "No files provided"
    
    try:
        VECTOR_STORE.delete_collection()
        VECTOR_STORE = Chroma(collection_name="docs", embedding_function=EMBED_MODEL, persist_directory="chroma_db")
        print("🧹 In-Memory Collection Wiped.")
    except:
        pass

    docs = []
    for path in files:
        try:
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                
                # --- THE HARD SLICER ---
                # We do NOT use LangChain's splitter. We split manually by "---".
                # This guarantees that chunks never bleed into each other.
                raw_chunks = full_text.split("---")
                
                for chunk in raw_chunks:
                    clean_chunk = chunk.strip()
                    if clean_chunk: # Only add if not empty
                        docs.append(Document(page_content=clean_chunk, metadata={"source": path}))
                        
            elif path.endswith(".pdf"): 
                # For PDFs, we still use the loader, but we can try to split by page or regex if needed.
                # For now, assuming TXT is the primary concern.
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
        except Exception as e:
            return f"Error loading {path}: {e}"
    
    if docs:
        VECTOR_STORE.add_documents(docs)
        return f"✅ Ingested {len(docs)} Distinct Recipe Cards."
    return "No documents found."

# --- AGENT PIPELINE ---
def run_agent_pipeline(user_query, history_state, system_prompt, img_path, temp_slider, strict_mode):
    if not CHAT_MODEL:
        yield history_state, "⚠️ Model not loaded."
        return

    history_state = history_state or []
    final_context = ""
    search_query = user_query
    
    # 1. VISUAL LOOK-AHEAD
    if img_path:
        yield history_state, "👁️ 🔍 Reading text on bottle..."
        base64_img = encode_image(img_path)
        vision_prompt = "Transcribe the largest text on the label. Read the brand name and the type of spirit (e.g. Tequila, Gin, Vodka)."
        vision_msg = [
            {"role": "user", "content": [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}
        ]
        try:
            vision_response = CHAT_MODEL.create_chat_completion(messages=vision_msg, max_tokens=64, temperature=0.1, repeat_penalty=1.2)
            raw_vision = vision_response['choices'][0]['message']['content']
            cleaned_items = clean_vision_output(raw_vision)
            search_query = cleaned_items if cleaned_items else "Unknown Bottle"
            yield history_state, f"🔍 Detected: {search_query}"
        except:
            pass

    # 2. RAG Retrieval & AGGRESSIVE FILTERING
    if VECTOR_STORE:
        yield history_state, "📚 Searching & Filtering..."
        
        if img_path and search_query != "Unknown Bottle":
            rag_query = f"Recipe with {search_query}"
        else:
            rag_query = user_query
            
        raw_results = VECTOR_STORE.similarity_search(rag_query, k=RETRIEVAL_K)
        
        # --- APPLY FILTER ---
        if img_path:
            filtered_results = smart_filter_context(raw_results, search_query)
        else:
            filtered_results = raw_results
        # -------------------------------

        # Deduplicate
        seen = set()
        unique_docs = []
        for d in filtered_results:
            if d.page_content not in seen:
                unique_docs.append(d)
                seen.add(d.page_content)
        
        if unique_docs:
            # We add double newlines to ensure visual separation in the prompt
            final_context = "\n\n".join([d.page_content for d in unique_docs])
        else:
            final_context = "NO_DATA_FOUND"

    # 3. Prompt Construction
    full_text_prompt = ""
    current_sys_instruction = system_prompt
    
    if final_context and final_context != "NO_DATA_FOUND":
        full_text_prompt = f"""
DETECTED INVENTORY:
{search_query if img_path else 'Unknown'}

SOURCE RECIPES FOUND:
{final_context}

INSTRUCTION:
1. State the name of the ONE recipe from the 'SOURCE RECIPES FOUND' that matches the inventory.
2. List the ingredients the user is missing.

User Question: {user_query}
"""
        if strict_mode:
            current_sys_instruction += " You are a Strict Bartender. Use ONLY the Source Recipes Found."
        else:
            current_sys_instruction += " You are a Helpful Mixologist."
            
    else:
        full_text_prompt = user_query
        current_sys_instruction = "You are a helpful assistant. No recipes were found."

    # 4. Payload & Generation
    if img_path:
        yield history_state, "🍸 Calculating..."
        history_state.append({"role": "user", "content": full_text_prompt})
    else:
        yield history_state, "🤔 Analyzing..."
        history_state.append({"role": "user", "content": full_text_prompt})

    history_state.append({"role": "assistant", "content": ""})
    messages = [{"role": "system", "content": current_sys_instruction}]
    
    if len(history_state) > 4:
        valid_history = []
        for m in history_state[-4:-2]:
            if isinstance(m['content'], str): valid_history.append(m)
        messages.extend(valid_history)
    
    messages.append({"role": "user", "content": full_text_prompt})

    try:
        stream = CHAT_MODEL.create_chat_completion(
            messages=messages, 
            stream=True, 
            max_tokens=512, 
            temperature=temp_slider,
            stop=["###", "User Request:", "USER INVENTORY", "User Question:", "Source Material"] 
        )
        
        full_raw_response = ""
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                content_chunk = chunk["choices"][0]["delta"]["content"]
                full_raw_response += content_chunk
                clean_display = clean_final_response(full_raw_response)
                history_state[-1]["content"] = clean_display
                yield history_state, f"✅ Active (Strict: {strict_mode})"
                
    except Exception as e:
        history_state[-1]["content"] = f"❌ Error: {e}"
        yield history_state, "Error"

# --- WRAPPER ---
def chat_wrapper(msg, history_state, img, sys_box, temp, strict):
    pipeline = run_agent_pipeline(msg, history_state, sys_box, img, temp, strict)
    for updated_history, status_msg in pipeline:
        ui_messages = convert_history_to_ui(updated_history)
        yield ui_messages, status_msg, updated_history

# --- UI LAYOUT ---
with gr.Blocks(title="LocalAGI Bartender") as demo:
    gr.Markdown("## 🍸 LocalAGI: Mixologist (Hard Slice)")
    history_state = gr.State([]) 

    found_files = get_gguf_files()
    target_chat = "MiniCPM-V-2_6-Q6_K.gguf"
    target_vis = "mmproj-MiniCPM-V-2_6-f16.gguf"
    target_emb = "nomic-embed-text-v1.5.Q4_K_M.gguf"

    def_chat = target_chat if target_chat in found_files else (found_files[0] if found_files else "None")
    def_vis = target_vis if target_vis in found_files else "None"
    def_emb = target_emb if target_emb in found_files else "None"

    with gr.Row():
        with gr.Column(scale=1):
            load_btn = gr.Button("🔄 Load Agent", variant="primary")
            m_chat = gr.Dropdown(found_files, value=def_chat, label="LLM")
            m_vis = gr.Dropdown(["None"] + found_files, value=def_vis, label="Vision Adapter")
            m_emb = gr.Dropdown(["None"] + found_files, value=def_emb, label="Embedding")
            
            up = gr.File(file_count="multiple", label="Upload Recipes")
            ingest_btn = gr.Button("📂 Ingest (Wipes Old Data)")
            ingest_status = gr.Textbox(label="Log")
            
            gr.Markdown("### 🎛️ Settings")
            temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, label="Creativity")
            strict_mode = gr.Checkbox(value=True, label="Strict RAG Mode")
            
            with gr.Accordion("⚙️ System Prompt", open=False):
                default_sys = "You are an expert Mixologist."
                sys_box = gr.Textbox(value=default_sys, lines=4)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, label="Live Interaction")
            with gr.Row():
                msg = gr.Textbox(label="Request", placeholder="What can I make?", scale=4)
                send_btn = gr.Button("Mix!", variant="primary", scale=1)
            with gr.Row():
                img = gr.Image(type="filepath", height=200, label="Upload Inventory")
                src = gr.Textbox(label="Status", interactive=False)

    load_btn.click(load_models, [m_chat, m_vis, m_emb], [src, send_btn, ingest_btn])
    ingest_btn.click(ingest, up, ingest_status)
    msg.submit(chat_wrapper, [msg, history_state, img, sys_box, temp_slider, strict_mode], [chatbot, src, history_state])
    send_btn.click(chat_wrapper, [msg, history_state, img, sys_box, temp_slider, strict_mode], [chatbot, src, history_state])

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    demo.launch(server_port=7860)