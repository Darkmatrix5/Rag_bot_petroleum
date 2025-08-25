import os
import fitz 
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def build_index(data_folder="data"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = []
    sources = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            text = extract_text(os.path.join(data_folder, filename))
            file_chunks = chunk_text(text)
            tagged_chunks = [f"From file: {filename}\n\n{chunk}" for chunk in file_chunks]
            chunks.extend(tagged_chunks)
            sources.extend([filename] * len(tagged_chunks))


    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings))

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    faiss.write_index(index, "faiss.index")

if __name__ == "__main__":
    build_index()
