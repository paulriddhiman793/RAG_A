import shutil
import os

# 1. Fix ChromaDB corruption
db_path = "chroma_db"
if os.path.exists(db_path):
    print("Deleting corrupted ChromaDB...")
    try:
        shutil.rmtree(db_path)
    except Exception as e:
        print(f"Failed to delete {db_path}: {e}")

# 2. Pre-download models to prevent API timeout
print("Pre-downloading models...")
from indexing.embeddings import get_embedding_model
get_embedding_model("general")

print("Downloading NLI model...")
from generation.hallucination_guard import get_nli_model
get_nli_model()

print("Re-ingesting document...")
from rag_system import RAGSystem
rag = RAGSystem()
pdf_path = r"docs\Reading Assignment 9 The Real Leadership Lessons from Steve Jobs.pdf"
if os.path.exists(pdf_path):
    rag.ingest(pdf_path)
    print("Ingestion complete.")
else:
    print(f"Could not find {pdf_path}")

print("All fixes applied successfully.")
