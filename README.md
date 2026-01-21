# 🍸 LocalAGI: The AI Sommelier & Mixologist

**LocalAGI** is a private, offline AI agent that doesn't just read recipes—it understands flavor. It uses Multimodal RAG to identify your physical liquor bottles and cross-references them against your personal database to act as an intelligent bartender.

Unlike basic search tools, LocalAGI features a **Python Logic Layer** that filters out hallucinations and a **Sommelier Engine** that can steer you toward "Sweet," "Smoky," or "Strong" drinks based on metadata tags.

## 🚀 Key Features

* **👁️ Visual Look-Ahead:** Uses `MiniCPM-V 2.6` to scan bottle labels via OCR (Optical Character Recognition).
* **🍷 Sommelier Intelligence:** Understads flavor profiles (e.g., "Sour," "Bozy," "Floral"). You can ask, *"I want something sweet with this Tequila,"* and it will filter the results accordingly.
* **🧠 Logic Filter (The "Smart Bartender"):** A Python algorithm that aggressively removes irrelevant recipes *before* the AI sees them. If you scan Tequila, it physically deletes Gin recipes from the context window to prevent confusion.
* **🔪 Hard-Slice Ingestion:** Custom ingestion engine that splits files by explicit delimiters (`---`), allowing for precision retrieval of specific recipes.
* **🧹 Nuclear Memory Wipe:** Automatically clears the vector database on startup and ingestion to ensure no "ghost data" remains from previous sessions.
* **📊 Menu Mode:** Can suggest up to 5 distinct cocktails at once, creating a curated menu based on your inventory.

## 🛠️ Installation & Setup

### 1. Clone & Install
git clone [https://github.com/YOUR_USERNAME/LocalAGI.git](https://github.com/YOUR_USERNAME/LocalAGI.git)
cd LocalAGI

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

2. Download Models
Place these specific GGUF models in the /models folder:
Role	Filename	Description
LLM (The Brain)	MiniCPM-V-2_6-Q6_K.gguf	Logic & Chat model (8B).
Vision Adapter	mmproj-MiniCPM-V-2_6-f16.gguf	Image encoder.
Embeddings	nomic-embed-text-v1.5.Q4_K_M.gguf	Recipe indexing.

📖 How to Use
1. Start the Server:
2. Bash  python app.py   
3. Open Browser: Go to http://localhost:7860.
4. Ingest Recipes: Upload a .txt file (formatted as below) and click "📂 Ingest".
5. Upload Inventory: Take a clear photo of your bottle(s).
7. Ask: * "What can I make?" (Generates a menu)
    * "I want something sweet." (Filters by Taste Profile)
📝 Recipe Formatting Guide (Critical)
For the AI to function as a Sommelier, your text file must follow this structure exactly.
* Separator: ---
* Metadata: Include Base Spirit: and Taste Profile: lines.
Example recipes.txt:
Plaintext

Recipe: Margarita
Base Spirit: Tequila
Taste Profile: Sour, Citrusy, Salty, Refreshing
Ingredients: 2oz Blanco Tequila, 1oz Lime juice, 0.5oz Cointreau
Instructions: Shake with ice and strain.
---
Recipe: Oaxaca Old Fashioned
Base Spirit: Tequila
Taste Profile: Smoky, Spirit-Forward, Sipper
Ingredients: 1.5oz Reposado, 0.5oz Mezcal, 1tsp Agave
Instructions: Stir with ice.
---

🏗️ Architecture
* Frontend: Gradio 5 (Custom CSS & Layout)
* Backend: Llama-cpp-python (Metal/M-Series Optimized)
* Vector DB: ChromaDB (Ephemeral/Reset on Launch)
	•	Orchestration: Custom Python Logic + LangChain
