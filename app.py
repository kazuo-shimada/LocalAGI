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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document 

# --- CONFIGURATION (UPDATED FOR MENU MODE) ---
RETRIEVAL_K = 40  # Increased to catch ALL Tequila recipes in a small DB

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

# --- CLEANING FUNCTIONS ---
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
    cut_triggers = ["INSTRUCTION:", "CONSTRAINT:", "SOURCE RECIPES FOUND:", "DETECTED INVENTORY:"]
    for trigger in cut_triggers:
        if trigger in text:
            text = text.split(trigger)[0]

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
            
    return " ".join(clean_sentences).strip()

def clean_user_message_for_ui(full_text):
    if "User Question:" in full_text:
        return full_text.split("User Question:")[-1].strip()
    return full_text

def convert_history_to_ui(history_state):
    ui_messages = []
    for msg in history_state:
        role = msg.get('role', '')
        content = msg.get('content', '')
        ui_content = ""
        if role == 'user':
            if isinstance(content, list):
                text_part = next((item['text'] for item in content if item['type'] == 'text'), "")
                clean_text = clean_user_message_for_ui(text_part)
                ui_content = f"🖼️ [Image Uploaded] {clean_text}"
            else:
                ui_content = clean_user_message_for_ui(str(content))
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

# --- LOGIC FILTER ---
def smart_filter_context(raw_docs, detected_inventory):
    if not detected_inventory or detected_inventory == "Unknown Bottle":
        return raw_docs
        
    inv_lower = detected_inventory.lower()
    
    target_spirit = None
    if "tequila" in inv_lower or "julio" in inv_lower or "patron" in inv_lower or "fortaleza" in inv_lower: target_spirit = "tequila"
    elif "gin" in inv_lower or "bombay" in inv_lower or "tanqueray" in inv_lower: target_spirit = "gin"
    elif "vodka" in inv_lower or "absolut" in inv_lower or "grey goose" in inv_lower: target_spirit = "vodka"
    elif "rum" in inv_lower or "bacardi" in inv_lower: target_spirit = "rum"
    elif "whiskey" in inv_lower or "bourbon" in inv_lower: target_spirit = "whiskey"

    if not target_spirit: return raw_docs 

    filtered_docs = []
    for doc in raw_docs:
        text = doc.page_content.lower()
        if target_spirit in text:
            filtered_docs.append(doc)

    if not filtered_docs: return raw_docs
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

# --- INGESTION ---
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

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "---", "Recipe:"], chunk_size=200, chunk_overlap=0)
    
    docs = []
    for path in files:
        try:
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                docs.append(Document(page_content=full_text, metadata={"source": path}))
            elif path.endswith(".pdf"): 
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
        except Exception as e:
            return f"Error loading {path}: {e}"
    
    if docs:
        final_chunks = splitter.split_documents(docs)
        VECTOR_STORE.add_documents(final_chunks)
        return f"✅ Ingested {len(final_chunks)} Recipes."
    return "No documents found."

# --- AGENT PIPELINE ---
def run_agent_pipeline(user_query, history_state, system_prompt, img_path, temp_slider, strict_mode):
    if not CHAT_MODEL:
        yield history_state, "⚠️ Model not loaded.", ""
        return

    history_state = history_state or []
    final_context = ""
    search_query = user_query
    
    # 1. VISUAL LOOK-AHEAD
    if img_path:
        yield history_state, "👁️ 🔍 Reading text on bottle...", ""
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
            yield history_state, f"🔍 Detected: {search_query}", f"🔍 DETECTED: {search_query}"
        except:
            pass

    # 2. RAG Retrieval
    reasoning_log = f"🔎 INVENTORY DETECTED: {search_query}\n\n"
    
    if VECTOR_STORE:
        yield history_state, "📚 Searching & Filtering...", reasoning_log
        
        if img_path and search_query != "Unknown Bottle":
            rag_query = f"Recipe with {search_query}"
        else:
            rag_query = user_query
            
        raw_results = VECTOR_STORE.similarity_search(rag_query, k=RETRIEVAL_K)
        filtered_results = smart_filter_context(raw_results, search_query) if img_path else raw_results

        seen = set()
        unique_docs = []
        for d in filtered_results:
            if d.page_content not in seen:
                unique_docs.append(d)
                seen.add(d.page_content)
        
        if unique_docs:
            final_context = "\n---\n".join([d.page_content for d in unique_docs])
            reasoning_log += f"📜 RECIPES FOUND & FILTERED ({len(unique_docs)}):\n{final_context[:500]}... [Truncated for Log]"
        else:
            final_context = "NO_DATA_FOUND"
            reasoning_log += "❌ NO MATCHING RECIPES FOUND."

    # 3. Prompt Construction (UPDATED FOR MULTIPLE SUGGESTIONS)
    full_text_prompt = ""
    current_sys_instruction = system_prompt
    
    if final_context and final_context != "NO_DATA_FOUND":
        full_text_prompt = f"""
DETECTED INVENTORY:
{search_query if img_path else 'Unknown'}

SOURCE RECIPES FOUND:
{final_context}

INSTRUCTION:
1. Match the Inventory to the Recipes.
2. If the user asks generally "What can I make?", list up to 5 distinct options from the Source Recipes.
3. For each option, mention its Taste Profile.
4. Briefly list missing ingredients for each.

User Question: {user_query}
"""
        if strict_mode:
            current_sys_instruction += " You are a Sommelier. Use the 'Taste Profile' data. Use ONLY the Source Recipes Found."
        else:
            current_sys_instruction += " You are a Helpful Mixologist."
            
    else:
        full_text_prompt = user_query
        current_sys_instruction = "You are a helpful assistant. No recipes were found."

    # 4. Payload & Generation
    if img_path:
        yield history_state, "🍸 Calculating...", reasoning_log
        history_state.append({"role": "user", "content": full_text_prompt})
    else:
        yield history_state, "🤔 Analyzing...", reasoning_log
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
        # INCREASED MAX TOKENS TO 1024 TO ALLOW LISTING 5+ DRINKS
        stream = CHAT_MODEL.create_chat_completion(
            messages=messages, 
            stream=True, 
            max_tokens=1024, 
            temperature=temp_slider,
            stop=["###", "User Request:", "USER INVENTORY", "User Question:", "Source Material", "INSTRUCTION:", "CONSTRAINT:"] 
        )
        
        full_raw_response = ""
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                content_chunk = chunk["choices"][0]["delta"]["content"]
                full_raw_response += content_chunk
                clean_display = clean_final_response(full_raw_response)
                history_state[-1]["content"] = clean_display
                yield history_state, f"✅ Active (Strict: {strict_mode})", reasoning_log
                
    except Exception as e:
        history_state[-1]["content"] = f"❌ Error: {e}"
        yield history_state, "Error", reasoning_log

# --- WRAPPER ---
def chat_wrapper(msg, history_state, img, sys_box, temp, strict):
    pipeline = run_agent_pipeline(msg, history_state, sys_box, img, temp, strict)
    for updated_history, status_msg, log_data in pipeline:
        ui_messages = convert_history_to_ui(updated_history)
        yield ui_messages, status_msg, updated_history, log_data

# --- UI LAYOUT ---
with gr.Blocks(title="LocalAGI Bartender") as demo:
    gr.Markdown("## 🍸 LocalAGI: Mixologist (Menu Mode)")
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
                default_sys = "You are an expert Sommelier. Guide the user based on taste."
                sys_box = gr.Textbox(value=default_sys, lines=4)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Live Interaction")
            
            with gr.Accordion("🧠 Agent Reasoning (Debug Log)", open=False):
                reasoning_box = gr.TextArea(label="Internal Thoughts", interactive=False, lines=10)
            
            with gr.Row():
                msg = gr.Textbox(label="Request", placeholder="What can I make? (Or ask: What is sweet?)", scale=4)
                send_btn = gr.Button("Mix!", variant="primary", scale=1)
            with gr.Row():
                img = gr.Image(type="filepath", height=200, label="Upload Inventory")
                src = gr.Textbox(label="Status", interactive=False)

    load_btn.click(load_models, [m_chat, m_vis, m_emb], [src, send_btn, ingest_btn])
    ingest_btn.click(ingest, up, ingest_status)
    msg.submit(chat_wrapper, [msg, history_state, img, sys_box, temp_slider, strict_mode], [chatbot, src, history_state, reasoning_box])
    send_btn.click(chat_wrapper, [msg, history_state, img, sys_box, temp_slider, strict_mode], [chatbot, src, history_state, reasoning_box])

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    demo.launch(server_port=7860)