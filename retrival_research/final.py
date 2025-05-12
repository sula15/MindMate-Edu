import os
import tkinter as tk
from tkinter import Tk, ttk, messagebox
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, Collection, MilvusException
from dotenv import load_dotenv
from openai import OpenAI
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key Setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file.")

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to preprocess and optimize the query
def preprocess_query(query):
    """
    Preprocesses the query with:
    - Tokenization
    - Lowercasing
    - Removal of stopwords
    - Stemming/Lemmatization
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Tokenize and process query
    tokens = word_tokenize(query.lower())  # Convert to lowercase and tokenize
    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words
    ]
    return " ".join(processed_tokens)

# Function to generate embeddings using CLIP
def get_clip_embedding(text, target_dim=512):  # Default CLIP embedding size is 512
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
    embeddings = outputs.squeeze().tolist()
    return pad_or_truncate(embeddings, target_dim)

# Function to pad/truncate embeddings to a specific dimension
def pad_or_truncate(vector, target_dim):
    if len(vector) > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))

# Milvus connection setup
connections.connect(host="localhost", port="19530", alias="default")

# Function to perform similarity search using CLIP
def perform_search(query):
    try:
        # Preprocess the query
        preprocessed_query = preprocess_query(query)
        user_vector = get_clip_embedding(preprocessed_query, target_dim=512)

        # Define search parameters for HNSW with COSINE metric
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 500},  # ef for search performance
        }

        # Initialize Milvus collection
        collection_name = "CLIP_HNSW"
        collection = Collection(collection_name)

        # Perform similarity search in Milvus
        search_results = collection.search(
            data=[user_vector],
            anns_field="vector",
            param=search_params,
            limit=5,
            output_fields=["description"]
        )

        return [(hit.entity.description, hit.distance) for hit in search_results[0]]

    except MilvusException as e:
        messagebox.showerror("Search Error", f"An error occurred during search: {e}")
        return []

# Function to generate answer with LLM
def generate_answer_with_llm(query, retrieved_chunks):
    try:
        # Combine user query and retrieved chunks
        context = "\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
        prompt = f"The user asked: {query}\nThe retrieved context is:\n{context}\nPlease provide a comprehensive answer based on the above information."

        # Generate the answer using OpenAI's chat API
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust model as necessary
            messages=[{"role": "user", "content": prompt}],
        )

        # Accessing the response correctly
        answer = chat_completion.choices[0].message.content
        return answer
    except Exception as e:
        messagebox.showerror("LLM Error", f"An error occurred: {e}")
        return "An error occurred while generating the answer."

# Tkinter GUI for CLIP and LLM Integration
def submit_query():
    query = query_input.get()
    if not query.strip():
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    # Perform search and display results
    results = perform_search(query)
    result_tree.delete(*result_tree.get_children())
    retrieved_chunks = [description for description, _ in results]

    for idx, (description, distance) in enumerate(results):
        result_tree.insert("", "end", values=(idx + 1, description, f"{distance:.4f}"))

    # Generate LLM answer
    llm_answer = generate_answer_with_llm(query, retrieved_chunks)
    llm_output_text.delete("1.0", tk.END)
    llm_output_text.insert(tk.END, llm_answer)

root = Tk()
root.title("RAG System with CLIP and LLM")
root.geometry("900x700")

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

result_tree = ttk.Treeview(result_frame, columns=("Rank", "Description", "Distance"), show="headings")
result_tree.heading("Rank", text="Rank")
result_tree.heading("Description", text="Description")
result_tree.heading("Distance", text="Distance")
result_tree.column("Rank", width=50, anchor="center")
result_tree.column("Description", width=550, anchor="w")
result_tree.column("Distance", width=100, anchor="center")
result_tree.pack(fill="both", expand=True)

# LLM Output Frame
llm_frame = ttk.LabelFrame(root, text="LLM Generated Answer", padding=(10, 10))
llm_frame.pack(fill="both", padx=10, pady=10, expand=True)

llm_output_text = tk.Text(llm_frame, wrap="word", height=10)
llm_output_text.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()

# Disconnect Milvus connection
connections.disconnect(alias="default")
