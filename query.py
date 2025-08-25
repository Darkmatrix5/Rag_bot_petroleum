import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


genai.configure(api_key="AIzaSyAUOVQo2iMsUsXZQeZTDG8kDK1w_5dIhog")
llm = genai.GenerativeModel("gemini-1.5-flash")

def load_index():
    index = faiss.read_index("faiss.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("sources.pkl", "rb") as f:
        sources = pickle.load(f)
    return index, chunks, sources

def search_and_answer(query, k=3):
    index, chunks, sources = load_index()
    query_vector = embedding_model.encode([query])
    _, I = index.search(np.array(query_vector), k)

    retrieved = "\n".join([chunks[i] for i in I[0]])

    context = f"""
You are a petroleum engineer AI assistant. Use the following well workover report excerpts to answer the user's question clearly and concisely.

Context:
{retrieved}

Question:
{query}

Answer:"""

    response = llm.generate_content(context)
    return response.text.strip()