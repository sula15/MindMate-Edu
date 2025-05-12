import os
import tkinter as tk
from tkinter import Tk, ttk, messagebox
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, MilvusException
from dotenv import load_dotenv
import google.generativeai as genai
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Gemini API Key Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Please check your .env file.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load Sentence Transformer (AllMini - 384D embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Milvus Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "AllMini_MDPCC"

# Connect to Milvus
try:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, alias="default")
except MilvusException as e:
    messagebox.showerror("Connection Error", f"Could not connect to Milvus: {e}")
    exit()

# Preprocess query
def preprocess_query(query):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    tokens = word_tokenize(query.lower())
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    
    return " ".join(processed_tokens)

# Get query embedding
def get_embedding(text):
    return embedding_model.encode([text])[0].tolist()

# Perform similarity search
def perform_search(query):
    try:
        preprocessed_query = preprocess_query(query)
        user_vector = get_embedding(preprocessed_query)

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 500},  
        }

        collection = Collection(COLLECTION_NAME)

        search_results = collection.search(
            data=[user_vector],
            anns_field="embedding",
            param=search_params,
            limit=5,
            output_fields=[
                "module_code", "module_name", "lecture_number", "lecture_title", 
                "source", "page", "text_chunk"
            ]
        )

        return [
            {
                "module_code": hit.entity.module_code,
                "module_name": hit.entity.module_name,
                "lecture_number": hit.entity.lecture_number,
                "lecture_title": hit.entity.lecture_title,
                "source": hit.entity.source,
                "page": hit.entity.page,
                "text_chunk": hit.entity.text_chunk,
                "distance": hit.distance
            }
            for hit in search_results[0]
        ]

    except MilvusException as e:
        messagebox.showerror("Search Error", f"An error occurred during search: {e}")
        return []

# Generate answer using Gemini
def generate_answer_with_gemini(query, retrieved_chunks):
    try:
        if not retrieved_chunks:
            return "No relevant context found. Please refine your query."

        context = "\n".join([
            f"Module: {chunk['module_name']} ({chunk['module_code']})\n"
            f"Lecture {chunk['lecture_number']}: {chunk['lecture_title']}\n"
            f"Source: {chunk['source']} (Page {chunk['page']})\n"
            f"Extract: {chunk['text_chunk']}\n"
            for chunk in retrieved_chunks
        ])

        prompt = (
            f"The user asked: {query}\n"
            f"The retrieved context is:\n{context}\n"
            f"Please provide a detailed and concise answer based on the above information."
        )

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        messagebox.showerror("LLM Error", f"An error occurred with Gemini: {str(e)}")
        return "An error occurred while generating the answer."

# GUI Logic
def submit_query():
    query = query_input.get().strip()
    if not query:
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    result_tree.delete(*result_tree.get_children())

    results = perform_search(query)

    if not results:
        messagebox.showinfo("No Results", "No relevant results found.")
        return

    for idx, item in enumerate(results):
        result_tree.insert(
            "", "end",
            values=(idx + 1, item["module_name"], item["lecture_title"], item["source"], item["page"], f"{item['distance']:.4f}")
        )

    llm_answer = generate_answer_with_gemini(query, results)
    llm_output_text.delete("1.0", tk.END)
    llm_output_text.insert(tk.END, llm_answer)

# Create GUI window
root = Tk()
root.title("AI-Enhanced RAG System")
root.geometry("1000x700")

# Query Frame
query_frame = ttk.LabelFrame(root, text="Enter Query", padding=(10, 10))
query_frame.pack(fill="x", padx=10, pady=10)

query_input = ttk.Entry(query_frame, width=80)
query_input.pack(side="left", padx=10, pady=10, fill="x", expand=True)

submit_button = ttk.Button(query_frame, text="Search and Generate", command=submit_query)
submit_button.pack(side="right", padx=10, pady=10)

# Results Frame
result_frame = ttk.LabelFrame(root, text="Search Results", padding=(10, 10))
result_frame.pack(fill="both", padx=10, pady=10, expand=True)

result_tree = ttk.Treeview(result_frame, columns=("Rank", "Module", "Lecture", "Source", "Page", "Distance"), show="headings")
result_tree.heading("Rank", text="Rank")
result_tree.heading("Module", text="Module")
result_tree.heading("Lecture", text="Lecture")
result_tree.heading("Source", text="Source")
result_tree.heading("Page", text="Page")
result_tree.heading("Distance", text="Distance")
result_tree.column("Rank", width=50, anchor="center")
result_tree.column("Module", width=200, anchor="w")
result_tree.column("Lecture", width=300, anchor="w")
result_tree.column("Source", width=200, anchor="w")
result_tree.column("Page", width=50, anchor="center")
result_tree.column("Distance", width=100, anchor="center")
result_tree.pack(fill="both", expand=True)

# LLM Output Frame
llm_frame = ttk.LabelFrame(root, text="LLM Generated Answer", padding=(10, 10))
llm_frame.pack(fill="both", padx=10, pady=10, expand=True)

llm_output_text = tk.Text(llm_frame, wrap="word", height=10)
llm_output_text.pack(fill="both", expand=True, padx=10, pady=10)

# Auto disconnect on close
def on_closing():
    connections.disconnect(alias="default")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
