# 🧘 OSHO ZEN RAG: Timeless Philosophical Dialogues

A privacy-first, fully local Retrieval-Augmented Generation (RAG) chatbot engine designed to search, retrieve, and synthesize wisdom across Osho's extensive philosophical literature. Built for low-resource efficiency, it runs smoothly on standard laptop CPUs using a hierarchical FAISS vector index and lightweight local inference via Ollama.

---

## ✨ Features

- **100% Local & Offline Privacy**: No API keys required, zero data sent to external cloud providers.
- **Hierarchical Parent-Child Retrieval**: Employs child chunking for fast search precision while feeding rich parent context to the LLM for deep, accurate answers.
- **Low-Resource CPU Optimization**: Fine-tuned to run comfortably on entry-level hardware (Intel Core i3, 8GB RAM).
- **Multi-Turn Conversational Memory**: Preserves context across turns for natural philosophical discussions.
- **Source Transparency**: Direct book and page citations provided with every answer.

---

## 💻 Hardware & Software Prerequisites

### Hardware Requirements
- **CPU**: Intel Core i3 (10th Gen or newer) / AMD Ryzen 3 or equivalent.
- **RAM**: Minimum 8 GB (16 GB recommended).
- **Disk Space**: ~5 GB (for Ollama model, dependencies, and vector database).

### Software Requirements
- **OS**: Windows 10/11, macOS, or Linux.
- **Python**: Version 3.10 or higher.
- **Ollama**: Installed and running locally ([Download Ollama](https://ollama.com)).

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vikash-dev/OSHO-ZEN-RAG-Timeless-Philosophical-Dialogues.git
cd OSHO-ZEN-RAG-Timeless-Philosophical-Dialogues
```

### 2. Set Up Virtual Environment

* **Windows (PowerShell):**
  ```powershell
  python -m venv myvenv
  .\myvenv\Scripts\Activate.ps1
  ```

* **macOS / Linux:**
  ```bash
  python3 -m venv myvenv
  source myvenv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🦙 Ollama Local LLM Configuration

1. Install Ollama from [ollama.com](https://ollama.com) and ensure the daemon is active (`ollama serve`).
2. Pull the recommended lightweight 3B model (optimized for CPU inference):
   ```bash
   ollama run qwen2.5:3b
   ```
   *(Alternatively, if running on a machine with higher specs/GPU, you can use `qwen2.5:7b`).*

---

## 📂 Master Index Download & Placement

Due to file size limits on GitHub, the 1.8 GB pre-merged master vector database (`OSHO_Philosophy_Master_Index`) is hosted externally.

1. Download the merged FAISS index folder from your release/download link.
2. Extract the unzipped `OSHO_Philosophy_Master_Index` directory directly into your project root folder.

### Expected Directory Structure:
```text
OSHO-ZEN-RAG-Timeless-Philosophical-Dialogues/
├── OSHO_Philosophy_Master_Index/
│   ├── index.faiss
│   └── index.pkl
├── chat.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Running the Chatbot

Start the interactive terminal interface by running:

```bash
python chat.py
```

### Example Usage:
```text
==================================================
🤖 OSHO Philosophy RAG Chatbot is Ready!
Type 'exit' or 'quit' to end the conversation.
==================================================

👤 You: What is the difference between mind and consciousness according to Osho?
🔍 Searching knowledge base...
🧠 Qwen is thinking...

🤖 Qwen:
Mind is the accumulation of the past—a storehouse of memories, desires, and conditioning. 
Consciousness, on the other hand, is pure presence in the moment...

📚 Sources: The Ultimate Alchemy Vol 1 (Page 42), Meditation: The First and Last Freedom (Page 108)
--------------------------------------------------
```

---

## 🔧 Architecture & Stack Details

- **Embedding Model**: `BAAI/bge-large-en-v1.5` (via `langchain-huggingface`)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain Community Integration
- **Inference Engine**: Ollama HTTP API (`http://localhost:11434/api/generate`)

---

## 📜 License & Citation

Distributed under the MIT License. Feel free to fork, adapt, and build upon this architecture for local RAG implementations.