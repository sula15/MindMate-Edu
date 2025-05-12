import fitz
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional, Literal, Union, Dict, List, Any
import gc
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import os
import uuid
import pymongo
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import base64
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for multi-model embedding generation"""
    # Embedding weights for final combination
    image_weight: float = 0.5
    text_weight: float = 0.5

    # Normalization options
    norm_type: Literal['l1', 'l2', 'none'] = 'l2'
    normalize_embeddings: bool = True
    normalize_combined: bool = True

    # Text extraction
    text_margin: int = 100  # Margin around images for text extraction
    min_text_length: int = 5  # Minimum text length to consider

    # Memory optimization
    batch_size: int = 4  # Batch size for processing
    clear_cache_interval: int = 10  # Clear CUDA cache every N batches

    # Model configuration
    clip_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "all-MiniLM-L6-v2"  # SentenceTransformer model
    use_separate_text_model: bool = True  # Whether to use SentenceTransformer or CLIP for text
    
    # Dimensionality reduction
    output_dim: int = 384
    use_dim_reduction: bool = True
    
    # Filtering options
    similarity_threshold: float = 0.98
    save_filtered: bool = True
    
    # Alignment options
    use_embedding_alignment: bool = False
    alignment_strength: float = 0.5
    
    # Database options
    mongodb_uri: str = "mongodb://localhost:27017/"
    mongodb_db: str = "lecture_images"  # Default database name
    mongodb_collection: str = "pdf_images_7"  # Updated default collection name
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "combined_embeddings_7"  # Updated default collection name
    
    # Debug options
    debug_mode: bool = False
    save_debug_files: bool = False
    debug_directory: str = "./debug_output"

    def validate(self):
        """Validate configuration parameters"""
        assert self.image_weight + self.text_weight == 1.0, "Weights must sum to 1.0"
        assert 0 <= self.image_weight <= 1.0, "Image weight must be between 0 and 1"
        assert self.norm_type in ['l1', 'l2', 'none'], "Invalid normalization type"
        assert 0.0 <= self.similarity_threshold <= 1.0, "Similarity threshold must be between 0 and 1"
        assert 0.0 <= self.alignment_strength <= 1.0, "Alignment strength must be between 0 and 1"


class DimensionalityReducer(nn.Module):
    """Linear projection layer for reducing embedding dimensions"""
    def __init__(self, input_dim: int, output_dim: int = 384):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def normalize_tensor(tensor: torch.Tensor, norm_type: str) -> torch.Tensor:
    """Normalize tensor based on specified type"""
    if norm_type == 'none':
        return tensor
    p = 1 if norm_type == 'l1' else 2
    return F.normalize(tensor, p=p, dim=1)


def simple_align_embeddings(image_embeddings, text_embeddings, alpha=0.5):
    """
    Simple Embedding Alignment Method
    
    Args:
        image_embeddings: Normalized image embeddings (N x d)
        text_embeddings: Normalized text embeddings (N x d)
        alpha: Alignment strength (0-1, default: 0.5)
    
    Returns:
        Aligned image and text embeddings
    """
    # Step 1: Calculate the average gap between paired embeddings
    gap_vectors = text_embeddings - image_embeddings
    avg_gap = torch.mean(gap_vectors, dim=0, keepdim=True)
    
    # Step 2: Move the embeddings toward each other by a fraction of the gap
    image_aligned = image_embeddings + (alpha/2) * avg_gap
    text_aligned = text_embeddings - (alpha/2) * avg_gap
    
    # Step 3: Renormalize to unit length (stay on the hypersphere)
    image_aligned = image_aligned / torch.norm(image_aligned, dim=1, keepdim=True)
    text_aligned = text_aligned / torch.norm(text_aligned, dim=1, keepdim=True)
    
    return image_aligned, text_aligned


def extract_text_around_image(page, image_bbox, margin=80):
    """Extract text within a margin around the image's bounding box"""
    x0, y0, x1, y1 = image_bbox
    expanded_bbox = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    text = page.get_text("text", clip=expanded_bbox)
    return text.strip()


def clear_gpu_cache():
    """Clear GPU cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def setup_mongodb(config):
    """Setup MongoDB connection and collection"""
    client = pymongo.MongoClient(config.mongodb_uri)
    db = client[config.mongodb_db]
    collection = db[config.mongodb_collection]
    
    # Create indexes for efficient querying
    collection.create_index("milvus_id")
    collection.create_index("lecture_code")
    collection.create_index("module_id")  # Add index for module_id
    collection.create_index("page_number")
    
    return collection


# # Current problematic code in setup_milvus function
# def setup_milvus(config, embedding_dim):
#     """Setup Milvus connection and collection with enhanced error handling"""
#     try:
#         # Connect to Milvus with explicit timeout
#         connections.connect("default", 
#                            host=config.milvus_host, 
#                            port=config.milvus_port,
#                            timeout=30)  # Increase timeout
        
#         logger.info(f"Connected to Milvus server at {config.milvus_host}:{config.milvus_port}")
        
#         # Calculate the exact dimension for vectors
#         vector_dim = embedding_dim * 2
#         logger.info(f"Setting up collection with vector dimension: {vector_dim}")
        
#         # Check if collection exists and drop if it does - THIS IS THE PROBLEM
#         if utility.has_collection(config.milvus_collection):
#             logger.info(f"Dropping existing collection: {config.milvus_collection}")
#             utility.drop_collection(config.milvus_collection)
        
#         # Create collection and schema...

# Improved function that reuses existing collections
def setup_milvus(config, embedding_dim):
    """Setup Milvus connection and collection with enhanced error handling"""
    try:
        # Connect to Milvus with explicit timeout
        connections.connect("default", 
                           host=config.milvus_host, 
                           port=config.milvus_port,
                           timeout=30)  # Increase timeout
        
        logger.info(f"Connected to Milvus server at {config.milvus_host}:{config.milvus_port}")
        
        # Calculate the exact dimension for vectors
        vector_dim = embedding_dim * 2
        logger.info(f"Vector dimension: {vector_dim}")
        
        # Check if collection exists and use it if it does
        if utility.has_collection(config.milvus_collection):
            logger.info(f"Using existing collection: {config.milvus_collection}")
            collection = Collection(config.milvus_collection)
            collection.load()
            return collection
        
        # If collection doesn't exist, create it
        logger.info(f"Creating new collection: {config.milvus_collection}")
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="combined_embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="page_number", dtype=DataType.INT32),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="module_id", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="lecture_code", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="lecture_number", dtype=DataType.INT32),
            FieldSchema(name="lecture_title", dtype=DataType.VARCHAR, max_length=200)
        ]
        
        # Create collection schema
        schema = CollectionSchema(fields, "PDF Combined Embeddings")
        
        # Create collection
        collection = Collection(config.milvus_collection, schema)
        
        # Create index with COSINE similarity
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        logger.info("Creating index on combined_embedding field with COSINE similarity")
        collection.create_index("combined_embedding", index_params)
        logger.info("Milvus collection setup complete")
        
        return collection
        
    except Exception as e:
        logger.error(f"Error setting up Milvus: {e}")
        raise

def save_models(image_dim_reducer, text_dim_reducer, config):
    """Save the trained dimensionality reducers to disk"""
    if config.use_dim_reduction and image_dim_reducer is not None and text_dim_reducer is not None:
        os.makedirs("saved_models", exist_ok=True)
        torch.save(image_dim_reducer.state_dict(), "saved_models/image_dim_reducer.pt")
        torch.save(text_dim_reducer.state_dict(), "saved_models/text_dim_reducer.pt")
        logger.info("Saved dimensionality reducers to disk")
        
        # Save configuration parameters needed for search
        config_params = {
            "output_dim": config.output_dim,
            "image_weight": config.image_weight,
            "text_weight": config.text_weight,
            "norm_type": config.norm_type,
            "normalize_combined": config.normalize_combined,
            "text_model_name": config.text_model_name,
        }
        
        with open("saved_models/config_params.json", "w") as f:
            json.dump(config_params, f)
        
        return True
    return False


def process_batch_multimodel(
    batch_images: List[Image.Image], 
    batch_texts: List[str], 
    clip_model: CLIPModel, 
    clip_processor: CLIPProcessor,
    sentence_transformer: Optional[SentenceTransformer],
    config: EmbeddingConfig,
    device: str,
    image_dim_reducer: Optional[DimensionalityReducer] = None,
    text_dim_reducer: Optional[DimensionalityReducer] = None
) -> tuple:
    """Process a batch using separate models for image and text embeddings"""
    with torch.no_grad():
        # Process images with CLIP
        clip_image_inputs = clip_processor(
            images=batch_images,
            return_tensors="pt",
        ).to(device)
        
        image_features = clip_model.get_image_features(**{
            k: v for k, v in clip_image_inputs.items() if k in ['pixel_values']
        })
        
        # Process text based on configuration
        if config.use_separate_text_model and sentence_transformer is not None:
            # Use SentenceTransformer for text embeddings
            text_features = sentence_transformer.encode(
                batch_texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            # Move to the same device as images
            if text_features.device != device:
                text_features = text_features.to(device)
        else:
            # Use CLIP for text embeddings
            clip_text_inputs = clip_processor(
                text=batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=77
            ).to(device)
            
            text_features = clip_model.get_text_features(**{
                k: v for k, v in clip_text_inputs.items() 
                if k in ['input_ids', 'attention_mask']
            })
        
        # Apply dimensionality reduction if configured
        if config.use_dim_reduction:
            if image_dim_reducer is not None:
                image_features = image_dim_reducer(image_features)
            
            if text_dim_reducer is not None:
                text_features = text_dim_reducer(text_features)
        
    return image_features, text_features


def image_to_base64(image):
    """Convert PIL Image to base64 encoded string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_embeddings_and_store(
    pdf_path: str,
    metadata: Dict[str, Any] = None,
    config: Optional[EmbeddingConfig] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Process PDF with multimodal models to extract images, 
    generate embeddings, and store in MongoDB and Milvus
    """
    logger.info(f"Processing on device: {device.upper()}")

    if config is None:
        config = EmbeddingConfig()
    config.validate()

    # Initialize metadata dictionary if not provided
    pdf_metadata = metadata or {}
    
    # Ensure metadata has consistent types
    if "lecture_number" in pdf_metadata and pdf_metadata["lecture_number"] is not None:
        pdf_metadata["lecture_number"] = int(pdf_metadata["lecture_number"])
    else:
        pdf_metadata["lecture_number"] = 0
        
    if "lecture_code" not in pdf_metadata or pdf_metadata["lecture_code"] is None:
        pdf_metadata["lecture_code"] = ""
    
    # Handle the new module_id field
    if "module_id" not in pdf_metadata or pdf_metadata["module_id"] is None:
        pdf_metadata["module_id"] = ""
        
    if "lecture_title" not in pdf_metadata or pdf_metadata["lecture_title"] is None:
        pdf_metadata["lecture_title"] = ""

    # Initialize models
    clip_model = CLIPModel.from_pretrained(config.clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    
    # Load SentenceTransformer if configured
    sentence_transformer = None
    if config.use_separate_text_model:
        logger.info(f"Using SentenceTransformer ({config.text_model_name}) for text embeddings")
        sentence_transformer = SentenceTransformer(config.text_model_name).to(device)
    
    # Get embedding dimensions
    clip_dim = 512  # Default CLIP embedding size
    
    # Determine text embedding dimension
    if config.use_separate_text_model and sentence_transformer is not None:
        text_dim = sentence_transformer.get_sentence_embedding_dimension()
        logger.info(f"SentenceTransformer embedding dimension: {text_dim}")
    else:
        text_dim = clip_dim
        
    # Initialize dimensionality reducers if configured
    image_dim_reducer = None
    text_dim_reducer = None
    
    if config.use_dim_reduction:
        logger.info(f"Using dimensionality alignment to {config.output_dim} dimensions")
        
        image_dim_reducer = DimensionalityReducer(
            input_dim=clip_dim,
            output_dim=config.output_dim
        ).to(device)
        image_dim_reducer.train(False)  # Set to evaluation mode
        
        text_dim_reducer = DimensionalityReducer(
            input_dim=text_dim,
            output_dim=config.output_dim
        ).to(device)
        text_dim_reducer.train(False)  # Set to evaluation mode
        
    # Final embedding dimension after reduction
    final_dim = config.output_dim if config.use_dim_reduction else clip_dim

    # Setup databases
    mongo_collection = setup_mongodb(config)
    milvus_collection = setup_milvus(config, final_dim)

    # Initialize storage for initial image collection
    all_images = []
    all_texts = []
    image_info = []  # Store page and position info

    # Open PDF and get total pages
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
    except Exception as e:
        logger.error(f"Error opening PDF: {e}")
        return {"error": str(e)}

    # First pass: collect all images
    logger.info("Phase 1: Collecting images...")
    pbar = tqdm(range(total_pages), desc="Collecting images")
    for page_num in pbar:
        page = pdf_document.load_page(page_num)
        images_on_page = page.get_images(full=True)
        
        pbar.set_postfix({"Images on page": len(images_on_page)})

        for img_index, img in enumerate(images_on_page):
            # Extract image and text
            xref = img[0]
            try:
                base_image = pdf_document.extract_image(xref)
                img_bytes = base_image["image"]
                image = Image.open(io.BytesIO(img_bytes))

                # Get image location and extract text
                img_bbox = next((page.get_image_bbox(img_info)
                               for img_info in page.get_images(full=True)
                               if img_info[0] == xref), None)

                nearby_text = ""
                if img_bbox:
                    nearby_text = extract_text_around_image(page, img_bbox, config.text_margin)
                
                if len(nearby_text.strip()) < config.min_text_length:
                    nearby_text = f"Image on page {page_num + 1} at position {img_index + 1}"

                # Store image and info
                all_images.append(image)
                all_texts.append(nearby_text)
                image_info.append({
                    'page': page_num + 1,
                    'position': img_index + 1,
                    'bbox': img_bbox,
                    'xref': xref
                })
            except Exception as e:
                logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                continue

    if not all_images:
        logger.error("No images found in the PDF document.")
        return {"error": "No images found"}

    # Second pass: generate initial embeddings for similarity check (using CLIP for consistency)
    logger.info("\nPhase 2: Generating initial embeddings for filtering...")
    batch_size = config.batch_size
    initial_embeddings = []

    for i in tqdm(range(0, len(all_images), batch_size), desc="Generating embeddings"):
        batch_images = all_images[i:i + batch_size]
        
        with torch.no_grad():
            inputs = clip_processor(
                images=batch_images,
                return_tensors="pt"
            ).to(device)
            
            batch_embeddings = clip_model.get_image_features(**{
                k: v for k, v in inputs.items() if k in ['pixel_values']
            })
            
            # Note: We don't reduce dimensions for similarity filtering
            initial_embeddings.extend(batch_embeddings.split(1))

    initial_embeddings = torch.cat(initial_embeddings, dim=0)
    
    # Filter duplicates using the improved clustering approach
    logger.info("\nPhase 3: Filtering duplicates using cluster-based approach...")
    image_embeddings = F.normalize(initial_embeddings, p=2, dim=1)
    
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(image_embeddings, image_embeddings.t())
    
    # Initialize for cluster-based approach
    similarity_np = similarity_matrix.cpu().numpy()
    clusters = []
    processed = set()
    
    for i in range(len(all_images)):
        if i in processed:
            continue
            
        # Find all images similar to the current one
        similar_indices = set(np.where(similarity_np[i] > config.similarity_threshold)[0].tolist())
        
        # Skip if this image is already part of another cluster
        if any(idx in processed for idx in similar_indices if idx != i):
            processed.add(i)
            continue
            
        # Create a new cluster
        cluster = similar_indices
        
        # Expand the cluster by checking transitive similarities
        expanded = True
        while expanded:
            expanded = False
            new_additions = set()
            
            for idx in cluster:
                if idx in processed:
                    continue
                    
                similar_to_idx = set(np.where(similarity_np[idx] > config.similarity_threshold)[0].tolist())
                new_items = similar_to_idx - cluster
                
                if new_items:
                    new_additions.update(new_items)
                    expanded = True
            
            cluster.update(new_additions)
        
        if len(cluster) > 0:
            clusters.append(list(cluster))
            processed.update(cluster)
    
    # For each cluster, identify the representative image
    unique_indices = []
    filtered_indices = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            # If only one image in cluster, keep it
            unique_indices.append(cluster[0])
        else:
            # For clusters of similar images, find the most representative one
            # Calculate average similarity of each image to others in the cluster
            avg_similarities = []
            
            for idx in cluster:
                other_indices = [j for j in cluster if j != idx]
                if other_indices:
                    avg_sim = np.mean([similarity_np[idx][j] for j in other_indices])
                    avg_similarities.append((idx, avg_sim))
                else:
                    avg_similarities.append((idx, 0))
            
            # Sort by average similarity (descending)
            avg_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Keep the image with highest average similarity
            best_idx = avg_similarities[0][0]
            unique_indices.append(best_idx)
            
            # Filter out others
            filtered_indices.extend([idx for idx, _ in avg_similarities[1:]])
    
    # Process unique images with the multi-model approach and store in databases
    logger.info(f"\nPhase 4: Processing {len(unique_indices)} unique images with multi-model embeddings...")
    unique_images = [all_images[i] for i in unique_indices]
    unique_texts = [all_texts[i] for i in unique_indices]
    unique_info = [image_info[i] for i in unique_indices]
    
    # Prepare for database insertion
    mongo_docs = []
    milvus_records = []
    
    # Define metadata from PDF
    lecture_code = str(pdf_metadata.get("lecture_code", ""))
    module_id = str(pdf_metadata.get("module_id", ""))  # Extract the module_id
    lecture_number = int(pdf_metadata.get("lecture_number", 0))
    lecture_title = str(pdf_metadata.get("lecture_title", ""))
    
    for i in tqdm(range(0, len(unique_images), batch_size), desc="Processing and storing data"):
        batch_images = unique_images[i:i + batch_size]
        batch_texts = unique_texts[i:i + batch_size]
        batch_info = unique_info[i:i + batch_size]
        
        # Generate embeddings
        batch_image_embeds, batch_text_embeds = process_batch_multimodel(
            batch_images, batch_texts, 
            clip_model, clip_processor, sentence_transformer,
            config, device, image_dim_reducer, text_dim_reducer
        )
        
        # Normalize embeddings if configured
        if config.normalize_embeddings:
            batch_image_embeds = normalize_tensor(batch_image_embeds, config.norm_type)
            batch_text_embeds = normalize_tensor(batch_text_embeds, config.norm_type)
        
        # Apply embedding alignment if configured
        if config.use_embedding_alignment:
            batch_image_embeds, batch_text_embeds = simple_align_embeddings(
                batch_image_embeds, 
                batch_text_embeds, 
                alpha=config.alignment_strength
            )
        
        # Create combined embeddings
        batch_combined_embeds = torch.cat([
            config.image_weight * batch_image_embeds,
            config.text_weight * batch_text_embeds
        ], dim=1)
        
        if config.normalize_combined:
            batch_combined_embeds = normalize_tensor(batch_combined_embeds, config.norm_type)
        
        # Process each item in the batch
        for j in range(len(batch_images)):
            # Create unique ID for cross-referencing between MongoDB and Milvus
            doc_id = str(uuid.uuid4())
            
            # Extract metadata for this image
            page_num = batch_info[j]['page']
            image_pos = batch_info[j]['position']
            
            # Clean text by removing problematic characters
            clean_text = str(batch_texts[j]).replace("\n", " ").replace("\r", " ")
            # Remove any other control characters
            clean_text = ''.join(ch if ord(ch) >= 32 else ' ' for ch in clean_text)
            # Trim extra spaces
            clean_text = ' '.join(clean_text.split()).strip()
            
            # Prepare MongoDB document
            mongo_doc = {
                "_id": doc_id,
                "milvus_id": doc_id,
                "image_base64": image_to_base64(batch_images[j]),
                "text": clean_text,
                "page_number": page_num,
                "position": image_pos,
                "pdf_path": pdf_path,
                "processed_at": datetime.now()
            }
            
            # Add any additional metadata provided by the user
            if pdf_metadata:
                mongo_doc.update(pdf_metadata)
            mongo_docs.append(mongo_doc)
            
            # Convert embeddings to Python lists ensuring they're float32
            combined_embed = np.array(
                batch_combined_embeds[j].cpu().numpy(), 
                dtype=np.float32
            ).tolist()
            
            # Create a single Milvus record as a dictionary
            milvus_record = {
                "id": doc_id,
                "combined_embedding": combined_embed,
                "page_number": int(page_num),
                "text": clean_text,
                "module_id": module_id,  # Add module_id field
                "lecture_code": lecture_code,
                "lecture_number": lecture_number,
                "lecture_title": lecture_title
            }
            
            # Add to the list of records
            milvus_records.append(milvus_record)
        
        # Periodically clear GPU cache
        if len(mongo_docs) % config.clear_cache_interval == 0:
            clear_gpu_cache()
    
    # Store data in MongoDB
    logger.info(f"Storing {len(mongo_docs)} documents in MongoDB...")
    if mongo_docs:
        try:
            mongo_collection.insert_many(mongo_docs)
            logger.info("Successfully inserted documents into MongoDB")
        except Exception as e:
            logger.error(f"Error inserting into MongoDB: {e}")
    
    # Store embeddings in Milvus
    logger.info(f"Storing {len(milvus_records)} entities in Milvus...")
    successful_inserts = 0
    
    if milvus_records:
        try:
            # Direct insertion
            milvus_collection.insert(milvus_records)
            milvus_collection.flush()
            successful_inserts = len(milvus_records)
            logger.info(f"Successfully inserted {successful_inserts} entities into Milvus")
        except Exception as e:
            logger.error(f"Error inserting into Milvus: {e}")
            
            # Try inserting in smaller batches
            logger.info("Attempting to insert in smaller batches...")
            successful_inserts = 0
            batch_size = 5  # Smaller batch size for reliability
            
            for i in range(0, len(milvus_records), batch_size):
                batch = milvus_records[i:i+batch_size]
                try:
                    milvus_collection.insert(batch)
                    successful_inserts += len(batch)
                    logger.info(f"Inserted batch {i//batch_size + 1}/{ceil(len(milvus_records)/batch_size)}")
                except Exception as batch_error:
                    logger.error(f"Error inserting batch {i//batch_size + 1}: {batch_error}")
            
            if successful_inserts > 0:
                milvus_collection.flush()
                logger.info(f"Successfully inserted {successful_inserts} entities in batches")
    
    # Save models for later use in search
    save_models(image_dim_reducer, text_dim_reducer, config)
    
    # Release model resources
    del clip_model, clip_processor, sentence_transformer, image_dim_reducer, text_dim_reducer
    clear_gpu_cache()
    
    return {
        "num_original_images": len(all_images),
        "num_filtered": len(filtered_indices),
        "num_unique": len(unique_indices),
        "num_inserted_milvus": successful_inserts,
        "mongodb_collection": config.mongodb_collection,
        "milvus_collection": config.milvus_collection,
        "pdf_metadata": pdf_metadata
    }


def search_images_by_text(
    query: str,
    top_k: int = 5,
    milvus_collection: str = "combined_embeddings_7",  # Updated default collection name
    mongodb_collection: str = "pdf_images_7",  # Updated default collection name
    mongodb_db: str = "lecture_images",
    mongodb_uri: str = "mongodb://localhost:27017/",
    milvus_host: str = "localhost",
    milvus_port: str = "19530",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Search for images using text query - retrieving from Milvus and MongoDB
    """
    logger.info(f"Searching for images matching: '{query}' in collection: {milvus_collection}")
    
    # Load saved configuration
    try:
        with open("saved_models/config_params.json", "r") as f:
            saved_config = json.load(f)
            
        # Extract parameters from saved config
        output_dim = saved_config.get("output_dim", 512)
        image_weight = saved_config.get("image_weight", 0.3)
        text_weight = saved_config.get("text_weight", 0.7)
        norm_type = saved_config.get("norm_type", "l2")
        normalize_combined = saved_config.get("normalize_combined", True)
        text_model_name = saved_config.get("text_model_name", "all-MiniLM-L6-v2")
        
        logger.info(f"Loaded configuration: output_dim={output_dim}, weights={image_weight}/{text_weight}")
    except FileNotFoundError:
        logger.warning("Could not find saved configuration, using defaults")
        output_dim = 512
        image_weight = 0.3
        text_weight = 0.7
        norm_type = "l2"
        normalize_combined = True
        text_model_name = "all-MiniLM-L6-v2"
    
    # Load models
    logger.info("Loading models...")
    sentence_transformer = SentenceTransformer(text_model_name).to(device)
    
    # Get text embedding dimension
    text_dim = sentence_transformer.get_sentence_embedding_dimension()
    logger.info(f"SentenceTransformer embedding dimension: {text_dim}")
    
    # Create new dimension reducers with same architecture
    text_dim_reducer = DimensionalityReducer(
        input_dim=text_dim,
        output_dim=output_dim
    ).to(device)
    text_dim_reducer.train(False)
    
    # Load the saved weights for dimension reducers
    try:
        text_dim_reducer.load_state_dict(torch.load("saved_models/text_dim_reducer.pt"))
        logger.info("Loaded saved text dimension reducer weights")
    except FileNotFoundError:
        logger.warning("Could not load saved dimension reducer weights")
    
    # Generate query embedding with SentenceTransformer
    logger.info("Generating query embedding...")
    with torch.no_grad():
        # Use SentenceTransformer for text
        query_text_embedding = sentence_transformer.encode(
            [query], 
            convert_to_tensor=True,
            show_progress_bar=False
        ).to(device)
        
        # Convert to proper tensor shape if needed
        if len(query_text_embedding.shape) == 1:
            query_text_embedding = query_text_embedding.unsqueeze(0)
        
        # Apply dimension reduction to match original pipeline
        query_text_embedding = text_dim_reducer(query_text_embedding)
    
    # Normalize the embedding
    query_text_embedding = normalize_tensor(query_text_embedding, norm_type)
    
    # Use text embedding for image embedding (since we only have text)
    query_image_embedding = query_text_embedding.clone()
    
    # Create combined embedding with weights
    combined_query_embedding = torch.cat([
        image_weight * query_image_embedding,
        text_weight * query_text_embedding
    ], dim=1)
    
    # Normalize combined embedding
    if normalize_combined:
        combined_query_embedding = normalize_tensor(combined_query_embedding, norm_type)
    
    # Convert to numpy and then list for Milvus
    query_vector = combined_query_embedding[0].cpu().numpy().astype(np.float32).tolist()
    
    # Search Milvus for similar vectors
    logger.info(f"Searching vector database: {milvus_collection}...")
    try:
        # Connect to Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        collection = Collection(milvus_collection)
        collection.load()
        
        # Use COSINE similarity for search
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}
        }
        
        # Perform search
        milvus_results = collection.search(
            data=[query_vector],
            anns_field="combined_embedding",
            param=search_params,
            limit=top_k,
            output_fields=["page_number", "text", "module_id", "lecture_code", "lecture_number", "lecture_title"]
        )
        
        milvus_hits = milvus_results[0]
        logger.info(f"Search returned {len(milvus_hits)} results")
        
    finally:
        # Clean up Milvus connection
        connections.disconnect("default")
    
    # Retrieve images from MongoDB using the IDs from Milvus
    logger.info(f"Retrieving images from MongoDB collection: {mongodb_collection}...")
    matches = []
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(mongodb_uri)
        db = client[mongodb_db]
        mongo_collection = db[mongodb_collection]
        
        for hit in milvus_hits:
            # Get the document ID
            doc_id = hit.id
            
            # Retrieve document from MongoDB
            mongo_doc = mongo_collection.find_one({"milvus_id": doc_id})
            
            if mongo_doc:
                # Decode base64 image
                image_data = base64.b64decode(mongo_doc["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                
                # Create match object with all metadata
                match = {
                    "image": image,
                    "text": mongo_doc["text"],
                    "page_number": mongo_doc["page_number"],
                    "position": mongo_doc.get("position", 0),
                    "module_id": mongo_doc.get("module_id", ""),  # Add module_id to results
                    "lecture_code": mongo_doc.get("lecture_code", ""),
                    "lecture_title": mongo_doc.get("lecture_title", ""),
                    "lecture_number": mongo_doc.get("lecture_number", 0),
                    "similarity_score": hit.score,
                    "pdf_path": mongo_doc.get("pdf_path", "")
                }
                
                matches.append(match)
            else:
                logger.warning(f"Document with ID {doc_id} not found in MongoDB collection {mongodb_collection}")
    
    finally:
        # Clean up MongoDB connection
        client.close()
    
    logger.info(f"Found {len(matches)} matching images")
    
    # Clean up model resources
    del sentence_transformer, text_dim_reducer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return matches


def display_search_results(query: str, matches, top_k: int = 6):
    """
    Display search results using matplotlib
    """
    if not matches:
        logger.info("No matches found.")
        return
    
    # Limit to top_k
    matches = matches[:min(top_k, len(matches))]
    
    # Calculate grid layout
    n_images = len(matches)
    n_cols = min(3, n_images)
    n_rows = ceil(n_images / n_cols)
    
    # Create figure
    plt.figure(figsize=(15, 5 * n_rows))
    
    # Display each image with its score and metadata
    for idx, match in enumerate(matches):
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Display the image
        plt.imshow(match["image"])
        plt.axis('off')
        
        # Build the title with metadata
        title = f"Score: {match['similarity_score']:.3f}\n"
        
        # Add module_id if available
        if match.get('module_id'):
            title += f"Module: {match['module_id']}"
            if match.get('lecture_code'):
                title += f" - {match['lecture_code']}\n"
            else:
                title += "\n"
        elif match.get('lecture_code'):
            title += f"{match['lecture_code']}"
            if match.get('lecture_number'):
                title += f" Lecture {match['lecture_number']}\n"
            else:
                title += "\n"
                
        if match.get('lecture_title'):
            title += f"{match['lecture_title']}\n"
            
        title += f"Page {match['page_number']}"
        if 'position' in match:
            title += f", Pos {match['position']}\n"
        else:
            title += "\n"
        
        # Add truncated text
        text = match["text"]
        truncated_text = text[:100] + "..." if len(text) > 100 else text
        title += truncated_text
        
        plt.title(title, fontsize=10, wrap=True)
    
    plt.suptitle(f'Search Results for Query: "{query}"', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return


def search_and_display_images(
    query: str,
    top_k: int = 5,
    milvus_collection: str = "combined_embeddings_7",  # Updated default collection name
    mongodb_collection: str = "pdf_images_7",  # Updated default collection name
    mongodb_db: str = "lecture_images"
):
    """
    Convenience function to search for images and display results
    """
    # Search for images
    matches = search_images_by_text(
        query=query,
        top_k=top_k,
        milvus_collection=milvus_collection,
        mongodb_collection=mongodb_collection,
        mongodb_db=mongodb_db
    )
    
    # Display results
    display_search_results(query, matches, top_k)
    
    return matches


# Example usage
if __name__ == "__main__":
    # Create configuration for database storage
    config = EmbeddingConfig(
        image_weight=0.3,
        text_weight=0.7,
        norm_type='l2',
        text_margin=100,
        batch_size=8,
        clear_cache_interval=5,
        output_dim=512,
        use_dim_reduction=True,
        similarity_threshold=0.98,
        save_filtered=False,
        text_model_name='all-MiniLM-L6-v2',
        use_separate_text_model=True,
        use_embedding_alignment=True,
        alignment_strength=0.5,
        # Database settings
        mongodb_uri="mongodb://localhost:27017/",
        mongodb_db="lecture_images",
        mongodb_collection="pdf_images_7",  # Updated collection name
        milvus_host="localhost",
        milvus_port="19530",
        milvus_collection="combined_embeddings_7"  # Updated collection name
    )

    # Example metadata for a PDF
    lecture_metadata = {
        "module_id": "MDPCC",  # Added module_id
        "lecture_code": "IT3061",
        "lecture_number": 3,
        "lecture_title": "Networking and Content Delivery Networks"
    }

    # Process a PDF and store in databases
    result = create_embeddings_and_store(
        pdf_path="2021-S2-IT3061-Lecture-03-Networking-and-Content-Delivery.pdf",
        metadata=lecture_metadata,
        config=config
    )
    
    # Search for images with a text query
    matches = search_and_display_images(
        query="Partitioning the VPC",
        top_k=5
    )
