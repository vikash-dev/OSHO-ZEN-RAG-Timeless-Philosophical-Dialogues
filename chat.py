import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import json

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MASTER_INDEX_DIR = "./OSHO_Master_Index"  # Or your merged folder path
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "cpu"  # Change to "cuda" if running on a GPU machine

# Ollama local API configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


# ==========================================
# 2. LOAD EMBEDDINGS & MASTER FAISS INDEX
# ==========================================
print("🚀 Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

print(f"📂 Loading FAISS Index from {MASTER_INDEX_DIR}...")
# Ensure you extract/unzip your 18 indexes and merge them into this directory first!
db = FAISS.load_local(MASTER_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant chunks
print("✅ Vector database loaded successfully!")


# ==========================================
# 3. QUERY OLLAMA (QWEN) FUNCTION
# ==========================================
def query_qwen(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error from Ollama: {response.text}"
    except Exception as e:
        return f"Connection error: {e}"


# ==========================================
# 4. CHAT LOOP
# ==========================================
def chat_bot():
    print("\n" + "="*50)
    print("🤖 OSHO Philosophy RAG Chatbot is Ready!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("="*50 + "\n")

    while True:
        user_query = input("\n👤 You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not user_query.strip():
            continue

        print("🔍 Searching knowledge base...")
        # Retrieve relevant chunks using the hierarchical/FAISS index
        relevant_docs = retriever.invoke(user_query)
        
        # Build context from parent content or page content
        context_texts = []
        sources = set()
        for doc in relevant_docs:
            # If using our hierarchical structure, grab the rich parent context if available
            text_to_use = doc.metadata.get("parent_content", doc.page_content)
            context_texts.append(text_to_use)
            sources.add(f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})")

        combined_context = "\n\n---\n\n".join(context_texts)

        # Construct Prompt for Qwen
        prompt = f"""You are a thoughtful philosophical assistant specializing in the teachings of Osho. 
Answer the user's question accurately using ONLY the provided context extracted from the books. If the answer cannot be found in the context, state that you don't know based on the texts.

Context:
{combined_context}

User Question: {user_query}

Answer:"""

        print("🧠 Qwen is thinking...")
        answer = query_qwen(prompt)
        
        print(f"\n🤖 Qwen:\n{answer}")
        print(f"\n📚 Sources: {', '.join(sources)}")
        print("-" * 50)


if __name__ == "__main__":
    chat_bot()