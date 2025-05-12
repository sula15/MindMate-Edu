import os
import glob
import fitz  # PyMuPDF
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
pdf_folder = os.path.join(os.path.dirname(__file__), "../MDPCC")  # Relative path to PDF folder
output_csv = os.path.join(os.path.dirname(__file__), "embeddings.csv")
output_json = os.path.join(os.path.dirname(__file__), "embeddings.json")

# Using a model that naturally produces 384-dimensional embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # This produces 384d embeddings

# === LOAD PYTORCH MODEL ===
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to create embeddings with PyTorch
def get_embeddings(text, max_length=512):
    """Generate 384-dimensional embeddings for the given text."""
    # Tokenize and truncate
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling - take average of all tokens
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # Convert to list for storage
    embedding_list = embeddings[0].numpy().tolist()
    
    # Verify dimension
    if len(embedding_list) != 384:
        print(f"Warning: Unexpected embedding dimension: {len(embedding_list)}")
    
    return embedding_list

# === FUNCTION TO EXTRACT TEXT FROM PDF ===
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file with page information."""
    pages_text = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:  # Only add if there's text on the page
                    pages_text.append((int(page_num + 1), text))  # Ensure page is int type
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return pages_text

# === FUNCTION TO CHUNK TEXT ===
def chunk_text(text, max_chunk_size=1000, overlap=100):
    """Split text into overlapping chunks of roughly max_chunk_size characters."""
    chunks = []
    
    if len(text) <= max_chunk_size:
        return [text]
        
    for i in range(0, len(text), max_chunk_size - overlap):
        chunk = text[i:i + max_chunk_size]
        chunks.append(chunk)
    
    return chunks

# === FUNCTION TO VALIDATE FIELD SIZES ===
def validate_field_sizes(record):
    """Validate and truncate fields to match Milvus schema size limits."""
    # Define max sizes according to schema
    max_sizes = {
        "module_code": 20,
        "module_name": 200,
        "lecture_title": 200,
        "source": 500,
        "file_type": 20,
        "text_chunk": 65535,
        "created_at": 50,
        "last_updated": 50
    }
    
    for field, max_size in max_sizes.items():
        if field in record and isinstance(record[field], str):
            if len(record[field]) > max_size:
                print(f"Warning: Truncating {field} from {len(record[field])} to {max_size} characters")
                record[field] = record[field][:max_size]
    
    return record

# === MAIN PROCESSING ===
data = []
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
print(f"Found {len(pdf_files)} PDFs in {pdf_folder}")

# Process a sample file first to verify
for pdf_idx, pdf in enumerate(pdf_files[:1]):
    pdf_filename = os.path.basename(pdf)
    print(f"Processing sample: {pdf}")

    # Extract module code and name from filename
    module_code = "MDPCC"  # Default
    module_name = "Machine Learning & Data Visualization"  # Default
    
    # Try to extract lecture number and title from filename
    filename_parts = os.path.basename(pdf).replace(".pdf", "").split("_")
    lecture_number = 0.0  # Make sure it's a float to match Double type
    lecture_title = "Unknown"
    
    if len(filename_parts) >= 2:
        try:
            lecture_number = float(filename_parts[1])
            lecture_title = " ".join(filename_parts[2:]) if len(filename_parts) > 2 else "Lecture " + filename_parts[1]
        except ValueError:
            pass
    
    # Extract text content from PDF
    pages_text = extract_text_from_pdf(pdf)

    if not pages_text:
        print(f"Skipping {pdf}, no text found.")
        continue

    # Process each page
    for page_num, page_text in pages_text[:1]:  # Sample just one page
        # Chunk the text
        text_chunks = chunk_text(page_text)
        print(f"  Processing page {page_num} - split into {len(text_chunks)} chunks")

        # Process each chunk
        for chunk_idx, chunk in enumerate(text_chunks[:1]):  # Sample just one chunk
            try:
                # Generate embeddings
                embedding = get_embeddings(chunk)
                
                # Create record, no ID field as it's auto-generated
                record = {
                    "embedding": embedding,
                    "module_code": module_code,
                    "module_name": module_name,
                    "lecture_number": lecture_number,
                    "lecture_title": lecture_title,
                    "source": pdf_filename,
                    "file_type": "pdf",
                    "page": page_num,  # This is int64 in schema
                    "text_chunk": chunk,
                    "created_at": current_time,
                    "last_updated": current_time
                }
                
                # Validate and truncate fields if necessary
                record = validate_field_sizes(record)
                
                data.append(record)
                print(f"    Processed chunk {chunk_idx+1}/{len(text_chunks)}")
                
                # Print the first 10 values of the embedding to verify
                print(f"    Embedding dimension: {len(embedding)}")
                print(f"    First 10 values: {embedding[:10]}")
                
            except Exception as e:
                print(f"    Error processing chunk {chunk_idx}: {e}")

# === SAVE SAMPLE TO CSV & JSON ===
if data:
    df = pd.DataFrame(data)
    
    # Print the column names for verification
    print("Columns in the DataFrame:", df.columns.tolist())
    
    # Generate a sample file first to verify
    sample_csv = output_csv.replace(".csv", "_sample.csv")
    sample_json = output_json.replace(".json", "_sample.json")
    
    # Save as CSV
    df.to_csv(sample_csv, index=False)
    print(f"Sample embeddings saved to {sample_csv}")

    # Save as JSON
    df.to_json(sample_json, orient="records", indent=4)
    print(f"Sample embeddings saved to {sample_json}")
    
    # Print the first record for inspection
    print("\nFirst record format (JSON):")
    print(pd.DataFrame([data[0]]).to_json(orient="records"))
else:
    print("No data processed. Check for errors above.")

# === FULL PROCESSING ===
if data:  # If the sample worked, proceed with full processing
    print("\nDo you want to process all PDF files? (y/n)")
    response = input().lower()
    
    if response == 'y':
        print("Processing all PDF files...")
        data = []  # Reset data
        
        for pdf_idx, pdf in enumerate(pdf_files):
            pdf_filename = os.path.basename(pdf)
            print(f"Processing: {pdf} ({pdf_idx+1}/{len(pdf_files)})")
            
            # Extract module code and name from filename
            module_code = "MDPCC"  # Default
            module_name = "Machine Learning & Data Visualization"  # Default
            
            filename_parts = os.path.basename(pdf).replace(".pdf", "").split("_")
            lecture_number = 0.0
            lecture_title = "Unknown"
            
            if len(filename_parts) >= 2:
                try:
                    lecture_number = float(filename_parts[1])
                    lecture_title = " ".join(filename_parts[2:]) if len(filename_parts) > 2 else "Lecture " + filename_parts[1]
                except ValueError:
                    pass
            
            pages_text = extract_text_from_pdf(pdf)
            
            if not pages_text:
                print(f"Skipping {pdf}, no text found.")
                continue
                
            for page_num, page_text in pages_text:
                text_chunks = chunk_text(page_text)
                print(f"  Processing page {page_num} - split into {len(text_chunks)} chunks")
                
                for chunk_idx, chunk in enumerate(text_chunks):
                    try:
                        embedding = get_embeddings(chunk)
                        
                        record = {
                            "embedding": embedding,
                            "module_code": module_code,
                            "module_name": module_name,
                            "lecture_number": lecture_number,
                            "lecture_title": lecture_title,
                            "source": pdf_filename,
                            "file_type": "pdf",
                            "page": page_num,
                            "text_chunk": chunk,
                            "created_at": current_time,
                            "last_updated": current_time
                        }
                        
                        # Validate and truncate fields if necessary
                        record = validate_field_sizes(record)
                        
                        data.append(record)
                        
                        # Progress update (not for every chunk to avoid console spam)
                        if chunk_idx % 5 == 0 or chunk_idx == len(text_chunks) - 1:
                            print(f"    Processed chunk {chunk_idx+1}/{len(text_chunks)}")
                            
                    except Exception as e:
                        print(f"    Error processing chunk {chunk_idx}: {e}")
        
        # Save full dataset
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_csv, index=False)
            print(f"Full embeddings saved to {output_csv}")
            
            # Ask before saving JSON (can be very large)
            print("Do you want to save the full dataset as JSON as well? (y/n)")
            json_response = input().lower()
            if json_response == 'y':
                df.to_json(output_json, orient="records", indent=4)
                print(f"Full embeddings saved to {output_json}")
            else:
                print("JSON export skipped.")
        else:
            print("No data processed for full dataset.")
    else:
        print("Full processing skipped.")