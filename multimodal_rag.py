import streamlit as st
import os
import tempfile
import torch
import pandas as pd
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime
import uuid
import json
import sys
import logging
from typing import Dict, List, Any, Optional

# LangChain imports
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

# Import semantic chunker
from semantic_chunker import process_document_with_semantic_chunking

# Import PDF processing components
from pdf_processor import (
    EmbeddingConfig, 
    create_embeddings_and_store, 
    search_images_by_text
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------- CORE FUNCTIONALITY (KEEP THIS SECTION AT MODULE LEVEL) --------

# Enhanced function with lecture number
def format_image_info(img):
    """Format image information for display with lecture number"""
    # Format source information with both lecture code and lecture number
    source_info = f"**Source**: {img.get('lecture_code', 'Unknown')}"
    
    # Explicitly include lecture number as a separate piece of information
    lecture_info = f"Lecture {img.get('lecture_number', 'Unknown')}"
    
    # Include page number
    page_info = f"Page {img.get('page_number', 'Unknown')}"
    
    # Combine all information into a well-formatted string
    return f"{source_info}, {lecture_info}, {page_info}"

# Initialize LLM
def get_llm():
    # Get API key from environment or from session if already set
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
        convert_system_message_to_human=True
    )

# Core functions for search and retrieval
def search_text_chunks(query, top_k=5, collection_name="combined_text_collection"):
    """Search for relevant text chunks using the query"""
    try:
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Connect to Milvus with configurable collection name
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"host": "localhost", "port": "19530"}
        )
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=top_k)
        
        # Format for easier consumption
        results = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "module_code": doc.metadata.get("module_code", "Unknown"),
                "module_name": doc.metadata.get("module_name", "Unknown"),
                "lecture_number": doc.metadata.get("lecture_number", "Unknown"),
                "lecture_title": doc.metadata.get("lecture_title", "Unknown"),
            }
            for doc in docs
        ]
        
        return results
    except Exception as e:
        logger.error(f"Error searching text: {e}")
        return []


def search_images(query, top_k=3, milvus_collection="combined_embeddings_7", 
                mongodb_collection="pdf_images_7"):
    """Search for relevant images using the query"""
    try:
        # Log the start of the image search
        logger.info(f"Starting image search for query: '{query}' with top_k={top_k}")
        
        # Use existing image search function with configurable collection names
        matches = search_images_by_text(
            query=query,
            top_k=top_k,
            milvus_collection=milvus_collection,
            mongodb_collection=mongodb_collection
        )
        
        # Log the number of matches found
        logger.info(f"Image search returned {len(matches)} matches")
        
        # Format the results
        results = []
        for i, match in enumerate(matches):
            try:
                # Convert PIL image to base64 for display in Streamlit
                if not match.get("image"):
                    logger.warning(f"Match {i} has no image data")
                    continue
                    
                buffer = io.BytesIO()
                match["image"].save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Create result dictionary WITH lecture_number explicitly included
                result = {
                    "image_data": img_str,
                    "similarity_score": match.get("similarity_score", 0),
                    "lecture_code": match.get("lecture_code", "Unknown"),
                    "lecture_number": match.get("lecture_number", "Unknown"),  # Explicitly include lecture_number
                    "lecture_title": match.get("lecture_title", "Unknown"),
                    "page_number": match.get("page_number", "Unknown"),
                    "text": match.get("text", "No text available")
                }
                
                # Log lecture info for debugging
                logger.info(f"Image {i+1} lecture info: code={result['lecture_code']}, number={result['lecture_number']}")
                
                results.append(result)
                
            except Exception as img_err:
                logger.error(f"Error processing image {i}: {img_err}")
        
        return results
    except Exception as e:
        logger.error(f"Error searching images: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def get_chat_history(messages):
    """Format chat history for context in prompts"""
    chat_history = ""
    for msg in messages:
        if msg["role"] != "system":
            if msg["role"] == "user":
                chat_history += f"User: {msg['content']}\n"
            else:
                # For assistant messages, we only include text part
                if isinstance(msg["content"], dict) and "answer_text" in msg["content"]:
                    chat_history += f"Assistant: {msg['content']['answer_text']}\n"
                else:
                    chat_history += f"Assistant: {msg['content']}\n"
    return chat_history

# LangChain Tools functions
def text_search_tool(query: str) -> str:
    """Tool to search for relevant text information."""
    results = search_text_chunks(query)
    if not results:
        return "No relevant text information found."
    
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append(
            f"Text Result {i+1}:\n"
            f"Module: {result['module_code']} - {result['module_name']}\n"
            f"Source: {result['source']}, Lecture {result.get('lecture_number', 'Unknown')}, Page: {result['page']}\n"
            f"Content: {result['content']}\n"
        )
    
    return "\n".join(formatted_results)

def image_search_tool(query: str) -> str:
    """Tool to search for relevant images."""
    results = search_images(query)
    if not results:
        return "No relevant images found."
    
    formatted_results = []
    for i, img in enumerate(results):
        # Explicitly include lecture number in the formatted output
        formatted_results.append(
            f"Image {i+1}:\n"
            f"From {img.get('lecture_code', 'Unknown')}, "
            f"Lecture {img.get('lecture_number', 'Unknown')}, "
            f"Page {img.get('page_number', 'Unknown')} - "
            f"Related to: {img.get('text', '')[:100]}...\n"
            f"Similarity score: {img.get('similarity_score', 0):.2f}\n"
        )
    
    return "\n".join(formatted_results)

def generate_combined_response(query):
    """Generate a response that incorporates both text and image information using LangChain"""
    try:
        # Make sure the LLM is working properly
        llm = get_llm()
        
        # Test the LLM with a simple query first
        try:
            logger.info("Testing LLM with simple query...")
            test_response = llm.invoke("Hello, are you working properly?")
            logger.info("LLM test successful")
        except Exception as llm_test_error:
            logger.error(f"LLM test failed: {llm_test_error}")
            raise
        
        # Setup tools
        tools = [
            Tool(
                name="TextSearch",
                func=text_search_tool,
                description="Searches for relevant text from documents. Use this when you need to find specific information from text."
            ),
            Tool(
                name="ImageSearch",
                func=image_search_tool,
                description="Searches for relevant images from documents. Use this when you need to find or show visual information."
            )
        ]
        
        # Get raw text and image results for our enhanced context
        text_results = search_text_chunks(query)
        image_results = search_images(query)
        
        # Format text context with explicit lecture numbers
        text_context = ""
        if text_results:
            text_context = "\n\n".join([
                f"Source: {result['module_code']}, Lecture {result.get('lecture_number', 'Unknown')}, Page {result['page']}\n"
                f"Content: {result['content']}"
                for result in text_results
            ])
        
        # Format image context with explicit lecture numbers
        image_context = ""
        if image_results:
            image_context = "\n".join([
                f"Image {i+1}: From {img.get('lecture_code', 'Unknown')}, Lecture {img.get('lecture_number', 'Unknown')}, "
                f"Page {img.get('page_number', 'Unknown')} - "
                f"Related to: {img.get('text', '')[:100]}...\n"
                for i, img in enumerate(image_results)
            ])
        
        # Create enhanced prompt that explicitly instructs to cite lecture numbers
        prompt = f"""
        User Query: {query}
        
        Text Context:
        {text_context if text_context else "No relevant text information found."}
        
        Image References:
        {image_context if image_context else "No relevant images found."}
        
        INSTRUCTIONS:
        1. Based on the provided context, answer the user's query thoroughly.
        2. ALWAYS cite the specific lecture numbers when providing information (e.g., "As explained in Lecture 3..." or "According to Lecture 5...").
        3. If there are relevant images that would help illustrate your answer, mention them by referring to their number and lecture (e.g., "As shown in Image 1 from Lecture 2...").
        4. Make the lecture number references a natural part of your answer, not just citations at the end.
        5. If the lecture number is unknown for some sources, you can mention this (e.g., "from an unspecified lecture").
        """
        
        try:
            # First try the enhanced direct approach with lecture citation instructions
            logger.info("Generating response with explicit lecture citation instructions...")
            response = llm.invoke(prompt)
            logger.info("Direct LLM response generation successful")
            
            return {
                "answer_text": response.content,
                "image_results": image_results,
                "has_images": len(image_results) > 0,
                "original_query": query
            }
        except Exception as direct_error:
            logger.error(f"Error with direct approach: {direct_error}")
            
            # Fall back to the agent-based approach if direct approach fails
            try:
                logger.info("Falling back to agent-based approach...")
                # Create conversation memory
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                
                # Initialize agent
                agent = initialize_agent(
                    tools,
                    llm,
                    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                    verbose=True,
                    memory=memory,
                    handle_parsing_errors=True,
                    max_iterations=3
                )
                
                # Get agent response
                agent_prompt = f"""Please answer this query, and make sure to reference the specific lecture numbers when providing information: {query}"""
                agent_response = agent.run(input=agent_prompt)
                logger.info("Agent completed successfully")
                
                return {
                    "answer_text": agent_response,
                    "image_results": image_results,
                    "has_images": len(image_results) > 0,
                    "original_query": query
                }
            except Exception as agent_error:
                logger.error(f"Agent error: {agent_error}")
                return {
                    "answer_text": f"I encountered an error while generating an answer. Please try again or rephrase your question.",
                    "image_results": image_results,
                    "has_images": len(image_results) > 0,
                    "original_query": query
                }
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Simplified fallback if everything else fails
        return {
            "answer_text": f"I encountered an error while processing your query. Please try again or rephrase your question.",
            "image_results": [],
            "has_images": False,
            "original_query": query
        }

# Document processing functions
def process_document_for_text(file, metadata, collection_name="combined_text_collection"):
    """Process a document for text extraction and indexing"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            pdf_path = tmp_file.name
        
        # Load document with LangChain
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add metadata to each document
        for doc in documents:
            doc.metadata.update(metadata)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and store in Milvus
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Store in Milvus with configurable collection name
        vectorstore = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            connection_args={"host": "localhost", "port": "19530"}
        )
        
        # Clean up temporary file
        try:
            os.unlink(pdf_path)
        except:
            pass
        
        return {
            "success": True,
            "chunk_count": len(chunks),
            "message": f"Successfully processed and indexed {len(chunks)} text chunks in collection '{collection_name}'"
        }
        
    except Exception as e:
        logger.error(f"Error processing document for text: {e}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

def process_document_for_images(file, metadata, config):
    """Process a document for image extraction and indexing"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            pdf_path = tmp_file.name
        
        # Process the PDF for images
        result = create_embeddings_and_store(
            pdf_path=pdf_path,
            metadata=metadata,
            config=config
        )
        
        # Clean up temporary file
        try:
            os.unlink(pdf_path)
        except:
            pass
        
        if "error" in result:
            return {
                "success": False,
                "message": f"Error: {result['error']}"
            }
        else:
            return {
                "success": True,
                "result": result,
                "message": f"Successfully processed {result['num_unique']} unique images"
            }
            
    except Exception as e:
        logger.error(f"Error processing document for images: {e}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

# -------- STREAMLIT APP FUNCTION (ONLY RUNS WHEN DIRECTLY EXECUTED) --------

def run_app():
    """Main Streamlit application function - only called when this script is run directly"""
    # Check if API key is available
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("Gemini API key not found. Please check your .env file or set it in the app.")
        GEMINI_API_KEY = st.text_input("Enter your Gemini API key:", type="password")
        if not GEMINI_API_KEY:
            st.stop()
    
    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "I am a multimodal assistant that can help you find information from both text and images in your documents."}
        ]

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize session state for options
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = False
    if 'include_images' not in st.session_state:
        st.session_state.include_images = True
    
    # Main app interface
    st.title("Multimodal RAG System")

    # Create tabs for chat and upload
    tab1, tab2 = st.tabs(["Chat Interface", "Upload Documents"])

    # Chat Interface Tab
    with tab1:
        st.header("Chat with your documents")
        
        # Chat sidebar with options
        with st.sidebar:
            st.header("Chat Options")
            
            # Use session state to track checkbox values
            show_sources = st.checkbox("Show text sources", value=st.session_state.show_sources)
            st.session_state.show_sources = show_sources
            
            include_images = st.checkbox("Include images in responses", value=st.session_state.include_images)
            st.session_state.include_images = include_images
            
            # Add a debug section
            with st.expander("Debug Information", expanded=False):
                st.write(f"Show sources: {st.session_state.show_sources}")
                st.write(f"Include images: {st.session_state.include_images}")
                st.write(f"Number of messages: {len(st.session_state.messages)}")
                st.write("Testing MongoDB connection:")
                
                try:
                    import pymongo
                    client = pymongo.MongoClient("mongodb://localhost:27017/")
                    db = client["lecture_images"]
                    collection = db["pdf_images_5"]
                    count = collection.count_documents({})
                    st.write(f"MongoDB connection successful. Found {count} documents.")
                    
                    # Get schema info from Milvus
                    from pymilvus import connections, Collection
                    connections.connect("default", host="localhost", port="19530")
                    collection = Collection("combined_embeddings_5")
                    schema = collection.schema
                    st.write("Milvus schema fields:", [field.name for field in schema.fields])
                    
                except Exception as e:
                    st.write(f"Database connection error: {str(e)}")
            
            st.divider()
            
            if st.button("Clear Chat History"):
                st.session_state.messages = [
                    {"role": "system", "content": "I am a multimodal assistant that can help you find information from both text and images in your documents."}
                ]
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.rerun()
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue  # Skip system messages
                
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    # If the message contains text and images
                    if isinstance(message["content"], dict) and "answer_text" in message["content"]:
                        # Show the text answer
                        st.markdown(message["content"]["answer_text"])
                        
                        # Show images if available and enabled
                        image_results = message["content"].get("image_results", [])
                        if image_results and st.session_state.include_images:
                            st.subheader("Relevant Images:")
                            
                            # Log the number of images found for debugging
                            st.write(f"Found {len(image_results)} relevant images")
                            
                            # Determine number of columns (max 3)
                            num_cols = min(3, len(image_results))
                            cols = st.columns(num_cols)
                            
                            for i, (col, img) in enumerate(zip(cols, image_results)):
                                with col:
                                    try:
                                        # Display image from base64
                                        st.image(
                                            f"data:image/png;base64,{img['image_data']}", 
                                            caption=f"Image {i+1}", 
                                            use_column_width=True
                                        )
                                        st.markdown(f"**Score**: {img['similarity_score']:.2f}")
                                        st.markdown(format_image_info(img))
                                    except Exception as img_err:
                                        st.error(f"Error displaying image {i+1}: {str(img_err)}")
                    
                        # Display text sources if enabled
                        if st.session_state.show_sources and isinstance(message["content"], dict):
                            text_results = text_search_tool(message["content"].get("original_query", ""))
                            with st.expander("View Text Sources", expanded=False):
                                st.markdown(text_results)
                    else:
                        # Just show text
                        st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response with a spinner
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_combined_response(prompt)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Display text response
                    st.markdown(response["answer_text"])
                    
                    # Add original query to the response for source retrieval
                    response["original_query"] = prompt
                    
                    # Display images if available and enabled
                    image_results = response.get("image_results", [])
                    if image_results and st.session_state.include_images:
                        st.subheader("Relevant Images:")
                        
                        # Debug information
                        st.write(f"Found {len(image_results)} relevant images")
                        
                        if not image_results:
                            st.warning("No images were found for this query.")
                        else:
                            # Determine number of columns (max 3)
                            num_cols = min(3, len(image_results))
                            cols = st.columns(num_cols)
                            
                            for i, (col, img) in enumerate(zip(cols, image_results)):
                                with col:
                                    try:
                                        # Display image from base64
                                        st.image(
                                            f"data:image/png;base64,{img['image_data']}", 
                                            caption=f"Image {i+1}", 
                                            use_column_width=True
                                        )
                                        st.markdown(f"**Score**: {img['similarity_score']:.2f}")
                                        st.markdown(format_image_info(img))
                                    except Exception as img_err:
                                        st.error(f"Error displaying image {i+1}: {str(img_err)}")
                    
                    # Display text sources if enabled
                    if st.session_state.show_sources:
                        text_results = text_search_tool(prompt)
                        with st.expander("View Text Sources", expanded=False):
                            st.markdown(text_results)

    # Upload Documents Tab
    with tab2:
        st.header("Upload and Process Documents")
        
        # Create columns for upload tabs
        text_tab, image_tab = st.tabs(["Text Processing", "Image Processing"])
        
        # Text Processing Tab
        with text_tab:
            st.subheader("Process Document for Text RAG")
            
            # Form for metadata and upload
            with st.form("text_upload_form"):
                # Metadata fields
                st.markdown("### Document Metadata")
                col1, col2 = st.columns(2)
                
                with col1:
                    module_code = st.text_input("Module Code", value="MDPCC")
                    module_name = st.text_input("Module Name", value="Machine Learning & Data Visualization")
                    lecture_number = st.number_input("Lecture Number", min_value=0)
                
                with col2:
                    lecture_title = st.text_input("Lecture Title")
                    source_type = st.selectbox("Source Type", ["Lecture", "Internet", "Other"])
                    
                # File upload
                uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
                
                # Process button
                submit_button = st.form_submit_button("Process Document for Text")
            
            # Process text document with semantic chunking
            if submit_button and uploaded_file:
                with st.spinner("Processing document with semantic chunking..."):
                    # Prepare metadata
                    text_metadata = {
                        "module_code": module_code,
                        "module_name": module_name,
                        "lecture_number": lecture_number,
                        "lecture_title": lecture_title,
                        "source_type": source_type,
                        "source": uploaded_file.name,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    # Process document with semantic chunking
                    result = process_document_with_semantic_chunking(
                        uploaded_file,
                        text_metadata,
                        collection_name="combined_text_collection",  # Keep your existing collection name
                        max_chunk_size=3000,  # Adjusted for your ~2500 word documents
                        chunk_overlap=500
                    )
                    
                    if result["success"]:
                        st.success(result["message"])
                        # Show document structure in expandable section
                        with st.expander("Document Structure"):
                            for heading, subheadings in result.get("chunk_structure", {}).items():
                                st.markdown(f"### {heading}")
                                for subheading in subheadings:
                                    st.markdown(f"- {subheading}")
                    else:
                        st.error(result["message"])
        
        # Image Processing Tab
        with image_tab:
            st.subheader("Process Document for Image Extraction")
            
            # Form for metadata and upload
            with st.form("image_upload_form"):
                # Metadata fields
                st.markdown("### Document Metadata")
                col1, col2 = st.columns(2)
                
                with col1:
                    img_module_id = st.text_input("Module ID", value="MDPCC", key="img_module_id")
                    img_lecture_code = st.text_input("Lecture Code", value="IT3061", key="img_lecture_code")
                    img_lecture_number = st.number_input("Lecture Number", min_value=0, key="img_lecture_number")
                
                with col2:
                    img_lecture_title = st.text_input("Lecture Title", key="img_lecture_title")
                    
                # File upload
                img_uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], key="img_upload")
                
                # Process button
                img_submit_button = st.form_submit_button("Process Document for Images")
            
            # Process image document
            if img_submit_button and img_uploaded_file:
                with st.spinner("Processing document for image extraction..."):
                    # Prepare metadata
                    image_metadata = {
                        "module_id": img_module_id,
                        "lecture_code": img_lecture_code,
                        "lecture_number": img_lecture_number,
                        "lecture_title": img_lecture_title,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    # Configure image embedding
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
                        # Database settings
                        mongodb_uri="mongodb://localhost:27017/",
                        mongodb_db="lecture_images",
                        mongodb_collection="pdf_images_7",
                        milvus_host="localhost",
                        milvus_port="19530",
                        milvus_collection="combined_embeddings_7"
                    )
                    
                    # Process document
                    result = process_document_for_images(img_uploaded_file, image_metadata, config)
                    
                    if result["success"]:
                        st.success(result["message"])
                        if "result" in result:
                            with st.expander("Processing Details"):
                                st.json(result["result"])
                    else:
                        st.error(result["message"])

    # Add a footer
    st.markdown("---")
    st.markdown("Multimodal RAG System | Created with Streamlit, LangChain, and Gemini")

# Only run the Streamlit app code when file is executed directly (not imported)
if __name__ == "__main__":
    # Set page config only when running directly
    st.set_page_config(
        page_title="Multimodal RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Run the app
    run_app()