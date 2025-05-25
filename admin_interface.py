import streamlit as st
import os
import subprocess
import sys
import tempfile
import torch
import pandas as pd
from PIL import Image
import io
import base64
import hashlib
from datetime import datetime
import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import PDF processing components
from pdf_processor import (
    EmbeddingConfig, 
    create_embeddings_and_store
)

# MongoDB and Milvus imports
import pymongo
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Import LangChain components for text processing
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# PPTX to PDF conversion function (included directly to avoid import issues)
def convert_pptx_to_pdf_in_memory(input_file):
    """
    Convert a PPTX file to PDF and return the PDF content as bytes.
    
    Args:
        input_file (str): Path to the PPTX file
    
    Returns:
        bytes: PDF content as bytes that can be used in memory
    """
    # Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found")
    
    # Create a temporary directory to store the output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate a temporary output path
        temp_output = os.path.join(temp_dir, os.path.splitext(os.path.basename(input_file))[0] + ".pdf")
        
        try:
            # Try using LibreOffice directly
            logger.info("Attempting conversion with LibreOffice...")
            
            # Determine the LibreOffice executable name
            libreoffice_cmd = "soffice"
            if sys.platform == "darwin":  # macOS
                if os.path.exists("/Applications/LibreOffice.app/Contents/MacOS/soffice"):
                    libreoffice_cmd = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            
            # Run the LibreOffice command with output to temp directory
            cmd = [
                libreoffice_cmd,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", temp_dir,
                input_file
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"LibreOffice conversion failed: {process.stderr}")
                raise RuntimeError("LibreOffice conversion failed")
            
            # Read the generated PDF into memory
            with open(temp_output, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
            
            return pdf_content
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"LibreOffice method failed: {str(e)}")
            
            # Try alternative methods
            try:
                # Try using unoconv if available
                logger.info("Attempting conversion with unoconv...")
                subprocess.run(["unoconv", "-f", "pdf", "-o", temp_output, input_file], check=True)
                with open(temp_output, 'rb') as pdf_file:
                    pdf_content = pdf_file.read()
                return pdf_content
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.error("unoconv method failed.")
            
            # If we're on Windows, try using the COM API
            if sys.platform.startswith('win'):
                try:
                    logger.info("Attempting conversion with PowerPoint COM API...")
                    import comtypes.client # type: ignore
                    
                    powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
                    powerpoint.Visible = 1
                    
                    presentation = powerpoint.Presentations.Open(os.path.abspath(input_file))
                    presentation.SaveAs(os.path.abspath(temp_output), 32)  # 32 is the PDF format
                    presentation.Close()
                    powerpoint.Quit()
                    
                    with open(temp_output, 'rb') as pdf_file:
                        pdf_content = pdf_file.read()
                    return pdf_content
                except Exception as e:
                    logger.error(f"PowerPoint COM API method failed: {str(e)}")
            
            # If all methods fail
            raise RuntimeError("Could not convert PPTX to PDF. No suitable conversion method available.")

# App configuration
st.set_page_config(
    page_title="RAG System Admin Interface",
    page_icon="🔐",
    layout="wide"
)

# Function to calculate file hash for duplicate detection
def calculate_file_hash(file_content):
    """Calculate MD5 hash of file content"""
    return hashlib.md5(file_content).hexdigest()

# Function to check if file already exists in the database
def is_duplicate_file(file_hash, metadata, mongodb_uri, mongodb_db, mongodb_collection):
    """Check if a file with the same hash and metadata has already been processed"""
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(mongodb_uri)
        db = client[mongodb_db]
        collection = db[mongodb_collection]
        
        # Check for existing documents with the same hash and metadata
        query = {
            "file_hash": file_hash,
            "module_id": metadata.get("module_id", ""),
            "lecture_code": metadata.get("lecture_code", ""),
            "lecture_number": metadata.get("lecture_number", 0)
        }
        
        existing_doc = collection.find_one(query)
        client.close()
        return existing_doc is not None
    except Exception as e:
        logger.error(f"Error checking for duplicates: {e}")
        return False

# Function to process a document for text extraction and indexing
def process_document_for_text(file_path, metadata, milvus_collection):
    """Process a document for text extraction and indexing"""
    try:
        # Load document with LangChain
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # Add metadata to each document
        for doc in documents:
            # Make sure all our important metadata is included
            doc.metadata.update({
                "module_id": metadata.get("module_id", ""),
                "lecture_code": metadata.get("lecture_code", ""),
                "lecture_number": metadata.get("lecture_number", 0),
                "lecture_title": metadata.get("lecture_title", ""),
                "page_number": doc.metadata.get("page", 0),  # PyMuPDF already includes page numbers
                "source_type": metadata.get("source_type", ""),
                "file_hash": metadata.get("file_hash", "")
            })
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(chunks)} text chunks")
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Let LangChain create the collection with all metadata preserved
        # LangChain automatically creates an index on the vector field
        vectorstore = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=milvus_collection,
            connection_args={"host": "localhost", "port": "19530"},
            drop_old=False
        )
        logger.info(f"Created Milvus collection: {milvus_collection}")
        
        # No need to create another index on the vector field - LangChain already did this
        
        # Optionally check the collection to confirm everything is set up correctly
        connections.connect("default", host="localhost", port="19530")
        collection = Collection(milvus_collection)
        collection.load()
        
        # Log the schema for debugging
        schema = collection.schema
        logger.info(f"Collection schema fields: {[field.name for field in schema.fields]}")
        
        # Find and log the vector field
        for field in schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                logger.info(f"Vector field found: {field.name} with dimension {field.params['dim']}")
        
        # Check if the collection has indexes
        try:
            index_info = collection.index()
            logger.info(f"Collection already has an index: {index_info.params}")
        except Exception as idx_err:
            logger.warning(f"Could not get index info: {str(idx_err)}")
        
        connections.disconnect("default")
        
        return {
            "success": True,
            "chunk_count": len(chunks),
            "message": f"Successfully processed and indexed {len(chunks)} text chunks",
            "milvus_text_collection": milvus_collection
        }
        
    except Exception as e:
        logger.error(f"Error processing document for text: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

# Function to process a document (PDF or PPTX)
def process_document(file, file_type, metadata, config, process_type="Text and Images", milvus_text_collection="combined_text_collection"):
    """Process a document (PDF or PPTX) for the RAG system"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
            file_content = file.getvalue()
            tmp_file.write(file_content)
            file_path = tmp_file.name
        
        # Calculate file hash for duplicate detection
        file_hash = calculate_file_hash(file_content)
        
        # Check for duplicates
        if is_duplicate_file(
            file_hash, 
            metadata, 
            config.mongodb_uri, 
            config.mongodb_db, 
            config.mongodb_collection
        ):
            # Clean up temporary file
            try:
                os.unlink(file_path)
            except:
                pass
            
            return {
                "success": False,
                "message": "This document has already been processed. Duplicates are not allowed."
            }
        
        # If file is PPTX, convert to PDF first
        if file_type == "pptx":
            try:
                # Convert PPTX to PDF
                pdf_content = convert_pptx_to_pdf_in_memory(file_path)
                
                # Create a new temporary file for the PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as pdf_tmp_file:
                    pdf_tmp_file.write(pdf_content)
                    pdf_path = pdf_tmp_file.name
                
                # Clean up the original PPTX temp file
                try:
                    os.unlink(file_path)
                except:
                    pass
                
                # Update file_path to point to the PDF
                file_path = pdf_path
                
            except Exception as e:
                logger.error(f"Error converting PPTX to PDF: {e}")
                return {
                    "success": False,
                    "message": f"Error converting PPTX to PDF: {str(e)}"
                }
        
        # Add file hash to metadata for future duplicate detection
        metadata["file_hash"] = file_hash
        
        results = {}
        
        # Process based on the selected type
        if process_type in ["Text Only", "Text and Images"]:
            # Process for text
            logger.info(f"Processing text embeddings into collection: {milvus_text_collection}")
            text_result = process_document_for_text(file_path, metadata, milvus_text_collection)
            results.update(text_result)
            
        if process_type in ["Images Only", "Text and Images"]:
            # Process for images
            logger.info(f"Processing image embeddings into collection: {config.milvus_collection}")
            image_result = create_embeddings_and_store(
                pdf_path=file_path,
                metadata=metadata,
                config=config
            )
            
            # Merge image results with text results
            if isinstance(image_result, dict):
                for key, value in image_result.items():
                    if key not in results:
                        results[key] = value
        
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

# Function to get available collections
def get_collections(db_type, connection_params):
    """Get available collections in MongoDB or Milvus"""
    if db_type == "mongodb":
        try:
            client = pymongo.MongoClient(connection_params.get("uri", "mongodb://localhost:27017/"))
            db = client[connection_params.get("db", "lecture_images")]
            collections = db.list_collection_names()
            client.close()
            return collections
        except Exception as e:
            logger.error(f"Error getting MongoDB collections: {e}")
            return []
    
    elif db_type == "milvus":
        try:
            host = connection_params.get("host", "localhost")
            port = connection_params.get("port", "19530")
            connections.connect("default", host=host, port=port)
            collections = utility.list_collections()
            connections.disconnect("default")
            return collections
        except Exception as e:
            logger.error(f"Error getting Milvus collections: {e}")
            return []
    
    return []

# Function to show database connection status
def show_database_status():
    """Show the status of MongoDB and Milvus connections"""
    st.sidebar.subheader("Database Connection Status")
    
    # Check MongoDB connection
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()
        st.sidebar.success("✅ MongoDB: Connected")
        client.close()
    except Exception as e:
        st.sidebar.error(f"❌ MongoDB: Not connected ({str(e)})")
    
    # Check Milvus connection
    try:
        connections.connect("default", host="localhost", port="19530")
        collections = utility.list_collections()
        st.sidebar.success("✅ Milvus: Connected")
        connections.disconnect("default")
    except Exception as e:
        st.sidebar.error(f"❌ Milvus: Not connected ({str(e)})")

# Main admin interface
def main():
    st.title("RAG System Admin Interface")
    
    # Initialize session state for collection selections
    if 'mongodb_collection' not in st.session_state:
        st.session_state.mongodb_collection = "Create New Collection..."
    if 'milvus_text_collection' not in st.session_state:
        st.session_state.milvus_text_collection = "Create New Collection..."
    if 'milvus_image_collection' not in st.session_state:
        st.session_state.milvus_image_collection = "Create New Collection..."
    
    # Check database connections
    show_database_status()
    
    # Create tabs for the interface
    tabs = st.tabs(["Document Processing", "Collection Management", "System Status"])
    
    # Document Processing Tab
    with tabs[0]:
        st.header("Document Processing and Embedding")
        
        # Database connection settings
        st.subheader("Database Connection Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            # MongoDB settings
            mongodb_uri = st.text_input("MongoDB URI", value="mongodb://localhost:27017/")
            mongodb_db = st.text_input("MongoDB Database", value="lecture_images")
            
            # Get MongoDB collections
            mongo_collections = get_collections("mongodb", {"uri": mongodb_uri, "db": mongodb_db})
            all_mongo_collections = ["Create New Collection..."] + mongo_collections
            
            # MongoDB collection selection with session state preservation
            mongodb_collection = st.selectbox(
                "MongoDB Collection", 
                options=all_mongo_collections,
                index=all_mongo_collections.index(st.session_state.mongodb_collection) 
                if st.session_state.mongodb_collection in all_mongo_collections 
                else 0,
                key="mongodb_collection_select",
                on_change=lambda: setattr(st.session_state, 'mongodb_collection', 
                                         st.session_state.mongodb_collection_select)
            )
            
            # If "Create New Collection" is selected, allow entering a name
            if mongodb_collection == "Create New Collection...":
                new_collection = st.text_input("New Collection Name", value="pdf_images")
                mongodb_collection = new_collection
        
        with col2:
            # Milvus settings
            milvus_host = st.text_input("Milvus Host", value="localhost")
            milvus_port = st.text_input("Milvus Port", value="19530")
            
            # Get Milvus collections
            milvus_collections = get_collections("milvus", {"host": milvus_host, "port": milvus_port})
            all_milvus_collections = ["Create New Collection..."] + milvus_collections
            
            # Milvus text collection selection with session state preservation
            milvus_text_collection = st.selectbox(
                "Milvus Text Collection", 
                options=all_milvus_collections,
                index=all_milvus_collections.index(st.session_state.milvus_text_collection) 
                if st.session_state.milvus_text_collection in all_milvus_collections 
                else 0,
                key="milvus_text_collection_select",
                on_change=lambda: setattr(st.session_state, 'milvus_text_collection', 
                                         st.session_state.milvus_text_collection_select)
            )
            
            # If "Create New Collection" is selected, allow entering a name
            if milvus_text_collection == "Create New Collection...":
                new_text_collection = st.text_input("New Text Collection Name", value="combined_text_collection")
                milvus_text_collection = new_text_collection
            
            # Milvus image collection selection with session state preservation
            milvus_image_collection = st.selectbox(
                "Milvus Image Collection", 
                options=all_milvus_collections,
                index=all_milvus_collections.index(st.session_state.milvus_image_collection) 
                if st.session_state.milvus_image_collection in all_milvus_collections 
                else 0,
                key="milvus_image_collection_select",
                on_change=lambda: setattr(st.session_state, 'milvus_image_collection', 
                                         st.session_state.milvus_image_collection_select)
            )
            
            # If "Create New Collection" is selected, allow entering a name
            if milvus_image_collection == "Create New Collection...":
                new_image_collection = st.text_input("New Image Collection Name", value="combined_embeddings")
                milvus_image_collection = new_image_collection
        
        # Document metadata section
        st.subheader("Document Metadata")
        col1, col2 = st.columns(2)
        
        with col1:
            module_id = st.text_input("Module ID", value="MDPCC", help="Module identifier (e.g., MDPCC, IOT)")
            lecture_code = st.text_input("Lecture Code", help="Code identifier for the lecture (e.g., IT3061)")
            lecture_number = st.number_input("Lecture Number", min_value=0, help="Lecture number in the course sequence")
        
        with col2:
            lecture_title = st.text_input("Lecture Title", help="Title of the lecture")
            source_type = st.selectbox("Source Type", ["Lecture", "Lab", "Tutorial", "Internet", "Other"])
            process_type = st.radio("Process Type", ["Text and Images", "Text Only", "Images Only"])
        
        # Advanced configuration options
        with st.expander("Advanced Configuration", expanded=False):
            st.subheader("Embedding Configuration")
            
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                image_weight = st.slider(
                    "Image Weight", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.3, 
                    step=0.1,
                    help="Weight given to image embeddings (text weight = 1 - image weight)"
                )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold", 
                    min_value=0.80, 
                    max_value=0.99, 
                    value=0.98, 
                    step=0.01,
                    help="Threshold for determining similar images during filtering"
                )
                
                batch_size = st.slider(
                    "Batch Size", 
                    min_value=1, 
                    max_value=16, 
                    value=8,
                    help="Number of images to process at once (higher values use more memory)"
                )
            
            with adv_col2:
                use_dim_reduction = st.checkbox(
                    "Use Dimensionality Reduction", 
                    value=True,
                    help="Reduce embedding dimensions for efficiency"
                )
                
                output_dim = st.select_slider(
                    "Output Dimension", 
                    options=[128, 256, 384, 512, 768],
                    value=512,
                    help="Dimension of the final embeddings"
                )
                
                use_alignment = st.checkbox(
                    "Use Embedding Alignment", 
                    value=True,
                    help="Align image and text embeddings for better multimodal representation"
                )
        
        # File upload section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "pptx"])
        
        # Process button
        if st.button("Process Document"):
            if uploaded_file is not None:
                # Get file type from extension
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                if file_type not in ["pdf", "pptx"]:
                    st.error("Unsupported file type. Please upload a PDF or PPTX file.")
                else:
                    # Show progress indicator
                    with st.spinner(f"Processing {file_type.upper()} document..."):
                        # Prepare metadata
                        metadata = {
                            "module_id": module_id,
                            "lecture_code": lecture_code,
                            "lecture_number": int(lecture_number) if lecture_number else 0,
                            "lecture_title": lecture_title,
                            "source_type": source_type,
                            "processed_by": "Admin Interface",
                            "processed_at": datetime.now().isoformat()
                        }
                        
                        # Configure processing based on selected options
                        config = EmbeddingConfig(
                            # Database settings
                            mongodb_uri=mongodb_uri,
                            mongodb_db=mongodb_db,
                            mongodb_collection=mongodb_collection,
                            milvus_host=milvus_host,
                            milvus_port=milvus_port,
                            milvus_collection=milvus_image_collection,
                            
                            # Embedding settings
                            image_weight=image_weight,
                            text_weight=1.0 - image_weight,
                            similarity_threshold=similarity_threshold,
                            batch_size=batch_size,
                            
                            # Advanced settings
                            use_dim_reduction=use_dim_reduction,
                            output_dim=output_dim,
                            use_embedding_alignment=use_alignment
                        )
                        
                        # Process the document with the selected process type and text collection
                        result = process_document(
                            uploaded_file, 
                            file_type, 
                            metadata, 
                            config,
                            process_type,
                            milvus_text_collection
                        )
                        
                        # Display results
                        if "success" in result and result["success"] == False:
                            st.error(result["message"])
                        elif "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.success("Document processed successfully!")
                            
                            # Show text processing results if available
                            if "chunk_count" in result:
                                st.info(f"Processed {result['chunk_count']} text chunks into Milvus collection: {result.get('milvus_text_collection', milvus_text_collection)}")
                            
                            # Create a results summary table if available
                            if "num_original_images" in result:
                                result_data = {
                                    "Metric": [
                                        "Original Images", 
                                        "Filtered (Similar) Images", 
                                        "Unique Images", 
                                        "Stored in Database"
                                    ],
                                    "Count": [
                                        result["num_original_images"],
                                        result["num_filtered"],
                                        result["num_unique"],
                                        result["num_inserted_milvus"]
                                    ]
                                }
                                
                                st.dataframe(pd.DataFrame(result_data))
                                
                                # Show database targets
                                st.write("**Storage Information:**")
                                if "milvus_text_collection" in result:
                                    st.write(f"Text Embeddings: {result.get('milvus_text_collection', milvus_text_collection)}")
                                st.write(f"MongoDB Collection: {result.get('mongodb_collection', mongodb_collection)}")
                                st.write(f"Image Embeddings: {result.get('milvus_collection', milvus_image_collection)}")
            else:
                st.warning("Please upload a document first.")
    
    # Collection Management Tab
    with tabs[1]:
        st.header("Collection Management")
        
        # Select database type
        db_type = st.radio("Database Type", ["MongoDB", "Milvus"])
        
        if db_type == "MongoDB":
            # MongoDB collection management
            st.subheader("MongoDB Collections")
            
            # Connection settings
            mongo_uri = st.text_input("MongoDB URI", value="mongodb://localhost:27017/", key="mongo_uri_cm")
            mongo_db = st.text_input("MongoDB Database", value="lecture_images", key="mongo_db_cm")
            
            # List collections
            try:
                client = pymongo.MongoClient(mongo_uri)
                db = client[mongo_db]
                collections = db.list_collection_names()
                
                if collections:
                    # Select a collection to manage
                    selected_collection = st.selectbox("Select Collection", options=collections)
                    
                    # Get collection statistics
                    try:
                        stats = db.command("collstats", selected_collection)
                        count = db[selected_collection].count_documents({})
                        
                        # Display statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Document Count", count)
                        with col2:
                            st.metric("Size", f"{stats['size'] / (1024*1024):.2f} MB")
                        
                        # Sample documents
                        with st.expander("Sample Documents", expanded=False):
                            samples = list(db[selected_collection].find().limit(3))
                            for i, sample in enumerate(samples):
                                # Remove large fields like image_base64 for display
                                if "image_base64" in sample:
                                    sample["image_base64"] = "[BINARY DATA]"
                                st.json(sample, expanded=False)
                        
                        # Manage collection
                        with st.expander("Manage Collection", expanded=False):
                            st.warning(f"Deleting the collection '{selected_collection}' will permanently remove all its data.")
                            
                            # Delete collection button
                            if st.button(f"Delete Collection '{selected_collection}'"):
                                confirm = st.text_input("Type the collection name to confirm deletion:")
                                if confirm == selected_collection:
                                    db[selected_collection].drop()
                                    st.success(f"Collection '{selected_collection}' deleted successfully!")
                                    st.rerun()
                                elif confirm:
                                    st.error("Collection name does not match. Deletion cancelled.")
                    
                    except Exception as e:
                        st.error(f"Error getting collection stats: {str(e)}")
                else:
                    st.info("No collections found in the database.")
                
                client.close()
            
            except Exception as e:
                st.error(f"Error connecting to MongoDB: {str(e)}")
        
        else:  # Milvus
            # Milvus collection management
            st.subheader("Milvus Collections")
            
            # Connection settings
            milvus_host = st.text_input("Milvus Host", value="localhost", key="milvus_host_cm")
            milvus_port = st.text_input("Milvus Port", value="19530", key="milvus_port_cm")
            
            # List collections
            try:
                connections.connect("default", host=milvus_host, port=milvus_port)
                collections = utility.list_collections()
                
                if collections:
                    # Select a collection to manage
                    selected_collection = st.selectbox("Select Collection", options=collections)
                    
                    # Get collection statistics
                    try:
                        collection = Collection(selected_collection)
                        collection.load()
                        stats = collection.get_stats()
                        
                        # Display statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Entity Count", stats.get("row_count", "Unknown"))
                        with col2:
                            try:
                                # Get index info
                                index_info = collection.index().params
                                st.metric("Index Type", index_info.get("index_type", "Unknown"))
                            except Exception as idx_err:
                                st.warning("No index found for this collection")
                        
                        # Collection schema
                        with st.expander("Collection Schema", expanded=False):
                            schema = collection.schema
                            for field in schema.fields:
                                st.write(f"**{field.name}**: {field.dtype}")
                        
                        # Manage collection
                        with st.expander("Manage Collection", expanded=False):
                            st.warning(f"Deleting the collection '{selected_collection}' will permanently remove all its data.")
                            
                            # Delete collection button
                            if st.button(f"Delete Collection '{selected_collection}'"):
                                confirm = st.text_input("Type the collection name to confirm deletion:")
                                if confirm == selected_collection:
                                    utility.drop_collection(selected_collection)
                                    st.success(f"Collection '{selected_collection}' deleted successfully!")
                                    st.rerun()
                                elif confirm:
                                    st.error("Collection name does not match. Deletion cancelled.")
                    
                    except Exception as e:
                        st.error(f"Error getting collection stats: {str(e)}")
                else:
                    st.info("No collections found in Milvus.")
                
                connections.disconnect("default")
            
            except Exception as e:
                st.error(f"Error connecting to Milvus: {str(e)}")
    
    # System Status Tab
    with tabs[2]:
        st.header("System Status")
        
        # Environment information
        st.subheader("Environment Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Python Version:** {sys.version.split()[0]}")
            st.write(f"**PyTorch Version:** {torch.__version__}")
            st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"**CUDA Version:** {torch.version.cuda}")
                st.write(f"**GPU Device:** {torch.cuda.get_device_name(0)}")
        
        with col2:
            # Check for required libraries
            libraries = {
                "PyMuPDF": "fitz",
                "transformers": "transformers",
                "sentence-transformers": "sentence_transformers",
                "pymilvus": "pymilvus",
                "pymongo": "pymongo",
                "streamlit": "streamlit"
            }
            
            st.write("**Required Libraries:**")
            for name, module_name in libraries.items():
                try:
                    __import__(module_name)
                    st.write(f"✅ {name}: Installed")
                except ImportError:
                    st.write(f"❌ {name}: Not found")
        
        # Database status
        st.subheader("Database Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**MongoDB Status:**")
            try:
                client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
                info = client.server_info()
                st.write(f"✅ Connected to MongoDB (Version: {info.get('version', 'Unknown')})")
                
                # Show databases and collection counts
                st.write("**Databases:**")
                databases = client.list_database_names()
                
                for db_name in databases:
                    if db_name not in ["admin", "config", "local"]:
                        db = client[db_name]
                        collections = db.list_collection_names()
                        st.write(f"- {db_name}: {len(collections)} collections")
                
                client.close()
            except Exception as e:
                st.write(f"❌ MongoDB connection error: {str(e)}")
        
        with col2:
            st.write("**Milvus Status:**")
            try:
                connections.connect("default", host="localhost", port="19530")
                collections = utility.list_collections()
                st.write(f"✅ Connected to Milvus")
                
                # List collections and entity counts
                st.write("**Collections:**")
                
                for coll_name in collections:
                    try:
                        collection = Collection(coll_name)
                        collection.load()
                        stats = collection.get_stats()
                        st.write(f"- {coll_name}: {stats.get('row_count', 'Unknown')} entities")
                    except Exception as coll_err:
                        st.write(f"- {coll_name}: Unable to get entity count")
                
                connections.disconnect("default")
            except Exception as e:
                st.write(f"❌ Milvus connection error: {str(e)}")

if __name__ == "__main__":
    main()