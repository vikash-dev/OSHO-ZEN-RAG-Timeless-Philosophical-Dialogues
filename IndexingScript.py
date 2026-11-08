import os
import glob
import pymupdf  # Modern PyMuPDF import
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
PDF_FOLDER_PATH = "/kaggle/input/datasets/vikashdev/osho-bookspdf/"
OUTPUT_FAISS_DIR = "/kaggle/working/OSHO_philosophy_index"

# Embedding Model running on GPU 0
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda:0"

# Chunking Strategy for Philosophical Text
CHUNK_SIZE = 3500
CHUNK_OVERLAP = 700


# ==========================================
# 2. PDF TEXT EXTRACTION
# ==========================================
def extract_text_from_pdfs(folder_path):
    documents = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in {folder_path}. Please check your path!")
        return documents

    print(f"📚 Found {len(pdf_files)} PDF books. Starting text extraction...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"    -> Processing: {filename}")
        
        try:
            doc = pymupdf.open(pdf_path)
            for page_num in range(len(doc)):
                page_text = doc[page_num].get_text("text")
                if page_text.strip():  # Skip empty pages
                    metadata = {
                        "source": filename,
                        "page": page_num + 1
                    }
                    documents.append(Document(page_content=page_text, metadata=metadata))
            doc.close()
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"✅ Extracted {len(documents)} total pages across all books.")
    return documents


# ==========================================
# 3. TEXT CHUNKING
# ==========================================
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} text chunks from the extracted pages.")
    return chunks


# ==========================================
# 4. EMBEDDING GENERATION & FAISS INDEXING
# ==========================================
def build_and_save_vectorstore(chunks):
    print(f"🚀 Loading Embedding Model '{EMBEDDING_MODEL_NAME}' onto {DEVICE}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("🧠 Generating embeddings and building FAISS vector database on GPU...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(OUTPUT_FAISS_DIR, exist_ok=True)
    vectorstore.save_local(OUTPUT_FAISS_DIR)
    print(f"💾 FAISS index successfully saved to: {OUTPUT_FAISS_DIR}")
    return vectorstore, embeddings


# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    raw_docs = extract_text_from_pdfs(PDF_FOLDER_PATH)
    
    if raw_docs:
        chunks = chunk_documents(raw_docs)
        db, embedding_model = build_and_save_vectorstore(chunks)
        
        print("\n🔍 Running test retrieval query...")
        test_query = "What is the core argument regarding free will and determinism?"
        results = db.similarity_search_with_score(test_query, k=2)
        
        print("\n--- TEST RETRIEVAL RESULTS ---")
        for doc, score in results:
            print(f"📍 Source: {doc.metadata['source']} | Page: {doc.metadata['page']} (Score: {score:.4f})")
            print(f"Snippet: {doc.page_content[:250]}...\n" + "-"*40)