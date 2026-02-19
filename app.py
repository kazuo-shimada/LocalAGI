import gradio as gr
from llama_cpp import Llama
import os
import glob
import re
import base64
import random
from PIL import Image
import io
import shutil
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document 
from ultralytics import YOLO

# --- CONFIGURATION ---
RETRIEVAL_K = 15  # Fetch 15 recipes PER SPIRIT to ensure variety

# --- THEME CSS (Monochrome) ---
monochrome_css = """
body, .gradio-container { background-color: #080808; color: #ffffff; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
#title-header { text-align: center; color: #ffffff; border-bottom: 1px solid #222; padding-bottom: 15px; margin-bottom: 25px; text-transform: uppercase; letter-spacing: 4px; font-weight: 300; }
button.primary { background-color: #ffffff !important; color: #000000 !important; border: 1px solid #ffffff; font-weight: 600; letter-spacing: 0.5px; transition: all 0.2s ease; }
button.primary:hover { background-color: #cccccc !important; border-color: #cccccc; }
button.secondary { background-color: #1a1a1a !important; color: #e0e0e0 !important; border: 1px solid #333333; }
button.secondary:hover { background-color: #333333 !important; border-color: #555555; }
.chatbot { background-color: #111111; border: 1px solid #222222; border-radius: 8px; }
textarea, input { background-color: #111111 !important; color: #ffffff !important; border: 1px solid #333333; }
.label { color: #888888 !important; text-transform: uppercase; font-size: 0.75em; letter-spacing: 1px; font-weight: 600; }
"""

# --- MEMORY WIPE ---
if os.path.exists("chroma_db"):
    try:
        shutil.rmtree("chroma_db")
    except:
        pass
os.makedirs("chroma_db", exist_ok=True)

# --- GLOBAL VARIABLES ---
CHAT_MODEL = None
EMBED_MODEL = None 
VECTOR_STORE = None 
YOLO_MODEL = None

# --- HELPER CLASSES ---
class LocalLlamaEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = Llama(model_path=model_path, embedding=True, verbose=False)
    def embed_documents(self, texts):
        return [self.model.create_embedding(t)['data'][0]['embedding'] for t in texts]
    def embed_query(self, text):
        return self.model.create_embedding(text)['data'][0]['embedding']

def encode_image(image_obj):
    if not image_obj: return None
    if image_obj.mode != 'RGB': image_obj = image_obj.convert('RGB')
    max_size = 1024
    if max(image_obj.size) > max_size: image_obj.thumbnail((max_size, max_size))
    buffered = io.BytesIO()
    image_obj.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_bottle_crops(image_path):
    global YOLO_MODEL
    if YOLO_MODEL is None:
        try:
            print("⚡ Loading YOLOv8 Nano...")
            YOLO_MODEL = YOLO("yolov8n.pt")
        except:
            return None
    
    results = YOLO_MODEL(image_path, verbose=False)
    found_crops = []
    original_img = Image.open(image_path)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 39 and box.conf > 0.4:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = original_img.size
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(w, x2 + 10)
                y2 = min(h, y2 + 10)
                crop = original_img.crop((x1, y1, x2, y2))
                found_crops.append(crop)
    return found_crops

# --- CLEANING FUNCTIONS ---
def clean_vision_output(raw_text):
    chatter_triggers = [
        "brand label is", "label on the front", "prominent label", 
        "first label", "consumer sees", "In this case", "type of alcohol",
        "label shows", "bottle with", "picture of", "produced by", 
        "distilled multiple times", "Question", "Answer"
    ]
    items = re.split(r'[,\n\.]', raw_text)
    clean_items = []
    for item in items:
        item = item.strip()
        if len(item) < 3: continue 
        if any(trigger in item for trigger in chatter_triggers): continue
        item = item.replace("The brand label is ", "").replace("The spirit type is ", "")
        clean_items.append(item)
    return ", ".join(list(set(clean_items)))

def clean_final_response(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    display_text = text
    
    if "---" in display_text:
        parts = display_text.split("---")
        if len(parts) > 1 and len(parts[0]) > 20: 
            display_text = parts[0]

    cut_triggers = ["INSTRUCTION:", "CONSTRAINT:", "SOURCE RECIPES FOUND:", "DETECTED INVENTORY:", "SHOPPING_LIST:", "User Question:"]
    for trigger in cut_triggers:
        if trigger in display_text:
            display_text = display_text.split(trigger)[0]
            
    return display_text.strip()

def clean_user_message_for_ui(full_text):
    if "User Question:" in full_text:
        return full_text.split("User Question:")[-1].strip()
    return full_text

def convert_history_to_ui(history_state):
    ui_messages = []
    for msg in history_state:
        role = msg.get('role', '')
        content = msg.get('content', '')
        display_text = ""
        if isinstance(content, list):
            for item in content:
                if item.get('type') == 'text':
                    clean = clean_user_message_for_ui(item['text'])
                    display_text = f"🖼️ [Image Uploaded] {clean}"
        else:
            display_text = clean_user_message_for_ui(str(content))
        ui_messages.append({"role": role, "content": display_text})
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

# --- SEARCH LOGIC ---
def smart_filter_and_rank(raw_docs, detected_inventory, user_query):
    # This ranking is secondary now, as we do primary logic in retrieval
    if user_query:
        query_words = user_query.lower().split()
        query_words = [w for w in query_words if w not in ["what", "can", "i", "make", "with", "a", "the", "is"]]
        if query_words:
            raw_docs.sort(key=lambda x: sum(word in x.page_content.lower() for word in query_words), reverse=True)
    return raw_docs

# --- LOAD MODELS ---
def load_models(chat_model, vis_model, emb_model):
    global CHAT_MODEL, EMBED_MODEL, VECTOR_STORE, YOLO_MODEL
    
    if YOLO_MODEL is None:
        try:
            print("⚡ Loading YOLOv8 Nano...")
            YOLO_MODEL = YOLO("yolov8n.pt") 
        except Exception as e:
            print(f"⚠️ YOLO Load Error: {e}")

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
            model_path=chat_path, n_gpu_layers=33, n_ctx=8192, chat_handler=chat_h, verbose=False 
        )
    except Exception as e:
        return f"❌ Chat Load Error: {e}", gr.update(), gr.update()
    
    if emb_path:
        try:
            EMBED_MODEL = LocalLlamaEmbeddings(emb_path)
            VECTOR_STORE = Chroma(collection_name="docs", embedding_function=EMBED_MODEL, persist_directory="chroma_db")
            return f"✅ Loaded: {chat_model} + Vision + YOLO", gr.update(interactive=True), gr.update(interactive=True)
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

    final_documents = []
    
    for path in files:
        try:
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                
                # Hard split by "Recipe:" to guarantee 1-to-1 Recipe-to-Document ratio
                raw_recipes = full_text.split("Recipe:")
                
                for r in raw_recipes:
                    if len(r.strip()) < 10: continue 
                    clean_content = f"Recipe:{r}"
                    final_documents.append(Document(page_content=clean_content, metadata={"source": path}))
                    
            elif path.endswith(".pdf"): 
                loader = PyPDFLoader(path)
                final_documents.extend(loader.load())
        except Exception as e:
            return f"Error loading {path}: {e}"
    
    if final_documents:
        VECTOR_STORE.add_documents(final_documents)
        return f"✅ Ingested {len(final_documents)} Individual Recipes."
    return "No documents found."

# --- AGENT PIPELINE (ROUND-ROBIN SEARCH) ---
def run_agent_pipeline(user_query, history_state, system_prompt, img_path, temp_slider, strict_mode, inventory_state):
    if not CHAT_MODEL:
        yield history_state, "⚠️ Model not loaded.", "", inventory_state
        return

    history_state = history_state or []
    
    # 1. INTERCEPT COMMANDS
    is_sweet = "sweet" in user_query.lower()
    is_sour = "sour" in user_query.lower()
    is_strong = "strong" in user_query.lower()
    is_button_click = is_sweet or is_sour or is_strong

    search_query = inventory_state if (inventory_state and inventory_state != "") else "Unknown Bottle"
    
    # 2. MULTI-BOTTLE VISION
    if img_path:
        yield history_state, "👁️ Scanning Shelf (YOLOv8)...", "", inventory_state
        crops = get_bottle_crops(img_path)
        if not crops:
            crops = [Image.open(img_path)]
            
        detected_items = []
        vision_prompt = "Identify the brand name and spirit type (e.g. 'Jack Daniels Whiskey'). Output NOTHING else. No sentences."
        
        for i, crop_img in enumerate(crops):
            yield history_state, f"👁️ Analyzing Bottle {i+1}/{len(crops)}...", "", inventory_state
            base64_img = encode_image(crop_img)
            vision_msg = [
                {"role": "user", "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ]
            try:
                v_response = CHAT_MODEL.create_chat_completion(messages=vision_msg, max_tokens=32, temperature=0.1)
                raw_label = v_response['choices'][0]['message']['content']
                clean_label = clean_vision_output(raw_label)
                if clean_label and "Unknown" not in clean_label:
                    detected_items.append(clean_label)
            except:
                pass
        
        if detected_items:
            search_query = ", ".join(list(set(detected_items)))
            inventory_state = search_query
            yield history_state, f"🔍 Shelf Inventory: {search_query}", f"🔍 SHELF DETECTED:\n{search_query}", inventory_state
        else:
            yield history_state, "⚠️ No readable labels found.", "", inventory_state

    # 3. SAFETY CHECK
    if search_query == "Unknown Bottle" and not is_button_click:
        history_state.append({"role": "user", "content": user_query})
        history_state.append({"role": "assistant", "content": "📸 Please upload a photo of your bottle(s) so I know what we are drinking!"})
        yield history_state, "⚠️ Waiting for Photo", "❌ No Inventory Detected", inventory_state
        return

    # 4. RAG Retrieval (ROUND-ROBIN STRATEGY)
    reasoning_log = f"🔎 INVENTORY: {search_query}\n\n"
    final_context = "NO_DATA_FOUND"
    
    if VECTOR_STORE:
        yield history_state, "📚 Matching Recipes...", reasoning_log, inventory_state
        
        # A. Detect Spirits in Inventory
        spirits_found = []
        inv_lower = search_query.lower()
        if "vodka" in inv_lower: spirits_found.append("Vodka")
        if "gin" in inv_lower: spirits_found.append("Gin")
        if "rum" in inv_lower: spirits_found.append("Rum")
        if "tequila" in inv_lower: spirits_found.append("Tequila")
        if "whiskey" in inv_lower or "bourbon" in inv_lower or "scotch" in inv_lower: spirits_found.append("Whiskey")
        if "brandy" in inv_lower or "cognac" in inv_lower: spirits_found.append("Brandy")

        unique_docs = []
        
        # B. Round-Robin Search
        if spirits_found:
            for spirit in spirits_found:
                # Build specific query for THIS spirit
                if is_sweet: sub_query = f"Sweet {spirit} cocktail"
                elif is_sour: sub_query = f"Sour {spirit} cocktail"
                elif is_strong: sub_query = f"Strong {spirit} cocktail"
                else: sub_query = f"Recipe using {spirit}"
                
                # Fetch distinct batch
                sub_results = VECTOR_STORE.similarity_search(sub_query, k=RETRIEVAL_K)
                unique_docs.extend(sub_results)
        else:
            # Fallback if no specific spirits detected (e.g. "Baileys")
            if is_sweet: rag_query = f"Sweet cocktail with {search_query}"
            elif is_sour: rag_query = f"Sour cocktail with {search_query}"
            elif is_strong: rag_query = f"Strong cocktail with {search_query}"
            else: rag_query = f"Recipe using {search_query}"
            unique_docs = VECTOR_STORE.similarity_search(rag_query, k=RETRIEVAL_K)

        # C. Deduplicate
        seen_content = set()
        deduped_docs = []
        for d in unique_docs:
            if d.page_content not in seen_content:
                deduped_docs.append(d)
                seen_content.add(d.page_content)
        
        # D. Smart Rank (Re-sort based on user query keywords if any)
        filtered_results = smart_filter_and_rank(deduped_docs, search_query, user_query)
        
        if filtered_results:
            # Take top 15 results after ranking
            final_docs = filtered_results[:15]
            final_context = "\n---\n".join([d.page_content for d in final_docs])
            reasoning_log += f"📜 RECIPES FOUND ({len(final_docs)}):\n{final_context}"
        else:
            reasoning_log += "❌ NO MATCHING RECIPES FOUND."

    # 5. Prompt Construction
    full_text_prompt = ""
    current_sys_instruction = system_prompt
    
    if final_context and final_context != "NO_DATA_FOUND":
        instruction_text = (
            "1. Match the Inventory to the Recipes.\n"
            "2. Select ONLY ONE best matching recipe.\n"
            "3. FORMATTING RULE: List ingredients on separate lines using bullet points (*).\n"
            "4. COPY the Ingredients and Instructions EXACTLY from the Source Text. Do not invent details.\n"
            "5. Mention missing ingredients."
        )
        
        if is_sweet: actual_user_q = f"Recommend a SWEET cocktail using {search_query}."
        elif is_sour: actual_user_q = f"Recommend a SOUR cocktail using {search_query}."
        elif is_strong: actual_user_q = f"Recommend a STRONG cocktail using {search_query}."
        else: actual_user_q = user_query
        
        full_text_prompt = f"DETECTED INVENTORY:\n{search_query}\n\nSOURCE RECIPES FOUND:\n{final_context}\n\nINSTRUCTION:\n{instruction_text}\n\nUser Question: {actual_user_q}"
        
        if strict_mode: current_sys_instruction += " You are a Sommelier. Use ONLY the Source Recipes Found."
        else: current_sys_instruction += " You are a Helpful Mixologist."
    else:
        full_text_prompt = user_query
        current_sys_instruction = "You are a helpful assistant. No recipes were found."

    # 6. Payload
    if img_path:
        yield history_state, "🍸 Mixing...", reasoning_log, inventory_state
        history_state.append({"role": "user", "content": full_text_prompt})
    else:
        yield history_state, "🤔 Analyzing...", reasoning_log, inventory_state
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
        temp_val = 0.8 if is_button_click else temp_slider
        stream = CHAT_MODEL.create_chat_completion(
            messages=messages, 
            stream=True, 
            max_tokens=1024, 
            temperature=temp_val,
            repeat_penalty=1.15,
            stop=["###", "User Request:", "USER INVENTORY", "User Question:", "Source Material", "INSTRUCTION:", "CONSTRAINT:"] 
        )
        
        full_raw_response = ""
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                content_chunk = chunk["choices"][0]["delta"]["content"]
                full_raw_response += content_chunk
                
                clean_display = clean_final_response(full_raw_response)
                
                history_state[-1]["content"] = clean_display
                yield history_state, f"✅ Active (Strict: {strict_mode})", reasoning_log, inventory_state
                
    except Exception as e:
        history_state[-1]["content"] = f"❌ Error: {e}"
        yield history_state, "Error", reasoning_log, inventory_state

# --- WRAPPER ---
def chat_wrapper(msg, history_state, img, sys_box, temp, strict, inv_state):
    pipeline = run_agent_pipeline(msg, history_state, sys_box, img, temp, strict, inv_state)
    for updated_history, status_msg, log_data, new_inv_state in pipeline:
        ui_messages = convert_history_to_ui(updated_history)
        yield ui_messages, status_msg, updated_history, log_data, new_inv_state

# --- CHIP EVENT ---
def chip_click(btn_val, history_state, img, sys_box, temp, strict, inv_state):
    for x in chat_wrapper(btn_val, history_state, img, sys_box, temp, strict, inv_state):
        yield x

# --- UI LAYOUT ---
with gr.Blocks(title="LocalAGI Bartender") as demo:
    gr.HTML("<div id='title-header'><h1>🍸 LocalAGI: The AI Sommelier</h1></div>")
    history_state = gr.State([]) 
    inventory_state = gr.State("") 

    found_files = get_gguf_files()
    target_chat = "MiniCPM-V-2_6-Q6_K.gguf"
    target_vis = "mmproj-MiniCPM-V-2_6-f16.gguf"
    target_emb = "nomic-embed-text-v1.5.Q4_K_M.gguf"

    def_chat = target_chat if target_chat in found_files else (found_files[0] if found_files else "None")
    def_vis = target_vis if target_vis in found_files else "None"
    def_emb = target_emb if target_emb in found_files else "None"

    with gr.Row():
        with gr.Column(scale=1):
            load_btn = gr.Button("🔄 Connect to Bar (Load AI)", variant="primary")
            
            with gr.Accordion("🔧 Model Config", open=False):
                m_chat = gr.Dropdown(found_files, value=def_chat, label="Brain (LLM)")
                m_vis = gr.Dropdown(["None"] + found_files, value=def_vis, label="Eyes (Vision)")
                m_emb = gr.Dropdown(["None"] + found_files, value=def_emb, label="Memory (Embed)")
            
            up = gr.File(file_count="multiple", label="Upload Recipe Book")
            ingest_btn = gr.Button("📂 Ingest & Wipe Memory")
            ingest_status = gr.Textbox(label="System Log", interactive=False)
            
            with gr.Accordion("⚙️ System Prompt", open=False):
                default_sys = "You are an expert Sommelier. Guide the user based on taste."
                sys_box = gr.Textbox(value=default_sys, lines=4)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=550, label="Bartender")
            
            with gr.Row():
                chip_sweet = gr.Button("🍬 Sweet", size="sm", variant="secondary")
                chip_sour = gr.Button("🍋 Sour", size="sm", variant="secondary")
                chip_strong = gr.Button("🥃 Strong", size="sm", variant="secondary")

            with gr.Row():
                msg = gr.Textbox(label="Request", placeholder="Type here...", scale=4)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                img = gr.Image(type="filepath", height=150, label="Upload Bottle")
                src = gr.Textbox(label="Vision Status", interactive=False)
                temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, label="Creativity")
                strict_mode = gr.Checkbox(value=True, label="Strict Mode")

            with gr.Accordion("🧠 Agent Reasoning (Debug Log)", open=False):
                reasoning_box = gr.TextArea(label="Internal Thoughts", interactive=False, lines=10)

    load_btn.click(load_models, [m_chat, m_vis, m_emb], [src, send_btn, ingest_btn])
    ingest_btn.click(ingest, up, ingest_status)
    
    msg.submit(chat_wrapper, [msg, history_state, img, sys_box, temp_slider, strict_mode, inventory_state], [chatbot, src, history_state, reasoning_box, inventory_state])
    send_btn.click(chat_wrapper, [msg, history_state, img, sys_box, temp_slider, strict_mode, inventory_state], [chatbot, src, history_state, reasoning_box, inventory_state])
    
    chip_sweet.click(chip_click, [chip_sweet, history_state, img, sys_box, temp_slider, strict_mode, inventory_state], [chatbot, src, history_state, reasoning_box, inventory_state])
    chip_sour.click(chip_click, [chip_sour, history_state, img, sys_box, temp_slider, strict_mode, inventory_state], [chatbot, src, history_state, reasoning_box, inventory_state])
    chip_strong.click(chip_click, [chip_strong, history_state, img, sys_box, temp_slider, strict_mode, inventory_state], [chatbot, src, history_state, reasoning_box, inventory_state])

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    demo.launch(server_port=7860, css=monochrome_css)
