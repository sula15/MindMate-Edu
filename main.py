import streamlit as st
import sys
import os
import logging
import json
from datetime import datetime
import pandas as pd
import tempfile
import torch
import io
import numpy as np
import speech_recognition as sr
import librosa # type: ignore
from transformers import BertTokenizer

# Set page config as the very first Streamlit command
st.set_page_config(
    page_title="Enhanced RAG System with Personalization",
    page_icon="🤖",
    layout="wide"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import anxiety detection module
from anxiety_detection import render_anxiety_detection_ui

# Import anxiety detection model
from model import MultimodalFusion

# Import personalization components only after page config
from student_personalization import StudentPersonalization, PersonalizationConfig
from personalization_ui import (
    add_personalization_to_sidebar,
    format_student_query,
    end_student_interaction,
    render_feedback_ui,
    render_analytics_dashboard,
    render_learning_path_recommendations
)
# Import existing RAG system components
from multimodal_rag import (
    search_images,
    search_text_chunks,
    get_llm,
    process_document_for_text,
    process_document_for_images,
    format_image_info
)

from pdf_processor import(
    EmbeddingConfig, 
    create_embeddings_and_store
)

from semantic_chunker import process_document_with_semantic_chunking


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "I am a multimodal assistant that can help you find information from both text and images in your documents."}
    ]

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = False
    
if 'include_images' not in st.session_state:
    st.session_state.include_images = True
    
if 'show_analytics' not in st.session_state:
    st.session_state.show_analytics = False
    
if 'current_module_id' not in st.session_state:
    st.session_state.current_module_id = None
    
if 'current_interaction_id' not in st.session_state:
    st.session_state.current_interaction_id = None

# Store collection names in session state for consistency
if 'text_collection' not in st.session_state:
    st.session_state.text_collection = "combined_text_collection"
    
if 'mongodb_collection' not in st.session_state:
    st.session_state.mongodb_collection = "pdf_images_7"
    
if 'milvus_collection' not in st.session_state:
    st.session_state.milvus_collection = "combined_embeddings_7"

# Initialize anxiety chatbot history
if 'anxiety_chat_history' not in st.session_state:
    st.session_state.anxiety_chat_history = []
    
if 'anxiety_transcript' not in st.session_state:
    st.session_state.anxiety_transcript = ""

# Function to generate response with personalization
def generate_personalized_response(query):
    """Generate a response that incorporates personalization for the student"""
    try:
        student_id = st.session_state.get("student_id")
        current_module_id = st.session_state.get("current_module_id")
        
        # If student is logged in, format query with learning style
        if student_id:
            formatted_query, interaction_id = format_student_query(
                query=query, 
                student_id=student_id,
                module_id=current_module_id
            )
            
            # Store interaction ID in session state for feedback
            st.session_state.current_interaction_id = interaction_id
        else:
            formatted_query = query
            interaction_id = None
        
        # Search for relevant content - using session state collection names
        text_results = search_text_chunks(
            formatted_query, 
            collection_name=st.session_state.text_collection
        )
        
        image_results = search_images(
            formatted_query,
            milvus_collection=st.session_state.milvus_collection,
            mongodb_collection=st.session_state.mongodb_collection
        )
        
        # Get LLM
        llm = get_llm()
        
        # Create enhanced context
        text_context = "\n\n".join([
            f"Source: {result['module_code']}, Lecture {result.get('lecture_number', 'Unknown')}, Page {result['page']}\n"
            f"Content: {result['content']}"
            for result in text_results
        ]) if text_results else "No relevant text information found."
        
        image_context = "\n".join([
            f"Image {i+1}: From {img.get('lecture_code', 'Unknown')}, Lecture {img.get('lecture_number', 'Unknown')}, "
            f"Page {img.get('page_number', 'Unknown')} - "
            f"Related to: {img.get('text', '')[:100]}...\n"
            for i, img in enumerate(image_results)
        ]) if image_results else "No relevant images found."
        
        # Create prompt with learning style if available
        prompt = f"""
        User Query: {formatted_query}
        
        Text Context:
        {text_context}
        
        Image References:
        {image_context}
        
        INSTRUCTIONS:
        1. Based on the provided context, answer the user's query thoroughly.
        2. ALWAYS cite the specific lecture numbers when providing information (e.g., "As explained in Lecture 3..." or "According to Lecture 5...").
        3. If there are relevant images that would help illustrate your answer, mention them by referring to their number and lecture (e.g., "As shown in Image 1 from Lecture 2...").
        4. Make the lecture number references a natural part of your answer, not just citations at the end.
        5. If the lecture number is unknown for some sources, you can mention this (e.g., "from an unspecified lecture").
        """
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        # Store retrieved sources for analytics
        retrieved_sources = []
        
        # Determine the primary module_id and lecture_code from sources
        primary_module_id = None
        primary_lecture_code = None
        primary_lecture_number = None
        module_lecture_counts = {}
        
        # Process text results
        for result in text_results:
            source_module_id = result.get("module_code")
            source_lecture_number = result.get("lecture_number")
            
            # Track frequency of module_id/lecture_number combinations
            key = f"{source_module_id}_{source_lecture_number}"
            if key in module_lecture_counts:
                module_lecture_counts[key] += 1
            else:
                module_lecture_counts[key] = 1
                
            retrieved_sources.append({
                "type": "text",
                "module_id": source_module_id,
                "lecture_number": source_lecture_number,
                "page": result.get("page")
            })
        
        # Process image results
        for img in image_results:
            source_module_id = img.get("module_id") or img.get("lecture_code")
            source_lecture_code = img.get("lecture_code")
            source_lecture_number = img.get("lecture_number")
            
            # Track frequency of module_id/lecture_number combinations
            key = f"{source_module_id}_{source_lecture_number}"
            if key in module_lecture_counts:
                module_lecture_counts[key] += 1
            else:
                module_lecture_counts[key] = 1
                
            retrieved_sources.append({
                "type": "image",
                "module_id": source_module_id,
                "lecture_code": source_lecture_code,
                "lecture_number": source_lecture_number,
                "page": img.get("page_number")
            })
        
        # Find the most frequent module_id/lecture_number combination
        if module_lecture_counts:
            most_common_key = max(module_lecture_counts.items(), key=lambda x: x[1])[0]
            parts = most_common_key.split('_')
            if len(parts) >= 2:
                primary_module_id = parts[0]
                primary_lecture_number = parts[1]
                
                # Find a matching lecture_code from image sources
                for source in retrieved_sources:
                    if (source.get("module_id") == primary_module_id and 
                        source.get("lecture_number") == primary_lecture_number and
                        source.get("lecture_code")):
                        primary_lecture_code = source.get("lecture_code")
                        break
        
        # Update interaction with retrieved sources
        if interaction_id:
            personalization = StudentPersonalization(PersonalizationConfig())
            
            # Update with sources and extracted metadata
            update_data = {
                "retrieved_sources": retrieved_sources
            }
            
            # Add module_id and lecture_code if we found them
            if primary_module_id:
                update_data["module_id"] = primary_module_id
                # Update current module in session state for recommendations
                st.session_state.current_module_id = primary_module_id
                
            if primary_lecture_code:
                update_data["lecture_code"] = primary_lecture_code
                
            if primary_lecture_number:
                update_data["lecture_number"] = primary_lecture_number
            
            # Apply the updates
            personalization.db[personalization.config.interactions_collection].update_one(
                {"_id": interaction_id},
                {"$set": update_data}
            )
        
        return {
            "answer_text": response.content,
            "image_results": image_results,
            "has_images": len(image_results) > 0,
            "original_query": query,
            "interaction_id": interaction_id,
            "module_id": primary_module_id,
            "lecture_code": primary_lecture_code
        }
    except Exception as e:
        logger.error(f"Error generating personalized response: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "answer_text": f"I encountered an error while processing your query. Please try again or rephrase your question.",
            "image_results": [],
            "has_images": False,
            "original_query": query,
            "interaction_id": None
        }

# Add personalization components to the sidebar
add_personalization_to_sidebar()

# Create tabs for chat, upload, analytics, and wellness
tab1, tab2, tab3, tab4 = st.tabs(["Chat Interface", "Upload Documents", "Learning Analytics", "Wellness Check"])

# Chat Interface Tab
with tab1:
    st.header("Chat with your documents")
    
    # Chat sidebar with options
    with st.sidebar:
        st.header("Chat Options")
        
        # Options
        show_sources = st.checkbox("Show text sources", value=st.session_state.show_sources)
        st.session_state.show_sources = show_sources
        
        include_images = st.checkbox("Include images in responses", value=st.session_state.include_images)
        st.session_state.include_images = include_images
        
        # Collection configuration
        st.subheader("Collection Settings")
        text_collection = st.text_input(
            "Text Collection", 
            value=st.session_state.text_collection,
            help="Name of the Milvus collection for text search"
        )
        st.session_state.text_collection = text_collection
        
        mongodb_collection = st.text_input(
            "MongoDB Images Collection", 
            value=st.session_state.mongodb_collection,
            help="Name of the MongoDB collection for image storage"
        )
        st.session_state.mongodb_collection = mongodb_collection
        
        milvus_collection = st.text_input(
            "Milvus Images Collection", 
            value=st.session_state.milvus_collection,
            help="Name of the Milvus collection for image embeddings"
        )
        st.session_state.milvus_collection = milvus_collection
        
        # Add a debug section
        with st.expander("Debug Information", expanded=False):
            st.write(f"Show sources: {st.session_state.show_sources}")
            st.write(f"Include images: {st.session_state.include_images}")
            st.write(f"Number of messages: {len(st.session_state.messages)}")
            st.write(f"Student ID: {st.session_state.get('student_id')}")
            st.write(f"Learning Style: {st.session_state.get('learning_style')}")
            st.write(f"Current Module: {st.session_state.get('current_module_id')}")
            st.write(f"Text Collection: {st.session_state.text_collection}")
            st.write(f"MongoDB Images Collection: {st.session_state.mongodb_collection}")
            st.write(f"Milvus Images Collection: {st.session_state.milvus_collection}")
            
        st.divider()
        
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "system", "content": "I am a multimodal assistant that can help you find information from both text and images in your documents."}
            ]
            st.rerun()
    
    # Display recommendations if module is set and student is logged in
    if st.session_state.get("student_id") and st.session_state.get("current_module_id"):
        render_learning_path_recommendations(st.session_state.current_module_id)
    
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
                    
                    # Show feedback UI if interaction_id exists
                    interaction_id = message["content"].get("interaction_id")
                    if interaction_id and st.session_state.get("student_id"):
                        render_feedback_ui(interaction_id)
                    
                    # Show images if available and enabled
                    image_results = message["content"].get("image_results", [])
                    if image_results and st.session_state.include_images:
                        st.subheader("Relevant Images:")
                        
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
                        query = message["content"].get("original_query", "")
                        if query:
                            text_results = search_text_chunks(
                                query, 
                                collection_name=st.session_state.text_collection
                            )
                            text_context = "\n\n".join([
                                f"Source: {result['module_code']}, Lecture {result.get('lecture_number', 'Unknown')}, Page {result['page']}\n"
                                f"Content: {result['content']}"
                                for result in text_results
                            ]) if text_results else "No relevant text information found."
                            
                            with st.expander("View Text Sources", expanded=False):
                                st.markdown(text_context)
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
                response = generate_personalized_response(prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display text response
                st.markdown(response["answer_text"])
                
                # Show feedback UI if interaction_id exists
                interaction_id = response.get("interaction_id")
                if interaction_id and st.session_state.get("student_id"):
                    render_feedback_ui(interaction_id)
                
                # Display images if available and enabled
                image_results = response.get("image_results", [])
                if image_results and st.session_state.include_images:
                    st.subheader("Relevant Images:")
                    
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
                    text_results = search_text_chunks(
                        prompt,
                        collection_name=st.session_state.text_collection
                    )
                    text_context = "\n\n".join([
                        f"Source: {result['module_code']}, Lecture {result.get('lecture_number', 'Unknown')}, Page {result['page']}\n"
                        f"Content: {result['content']}"
                        for result in text_results
                    ]) if text_results else "No relevant text information found."
                    
                    with st.expander("View Text Sources", expanded=False):
                        st.markdown(text_context)

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
                # Add text collection field
                text_collection_name = st.text_input(
                    "Text Collection Name", 
                    value=st.session_state.text_collection,
                    help="Name of the Milvus collection to store text embeddings"
                )
                
            # File upload
            uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
            
            # Process button
            submit_button = st.form_submit_button("Process Document for Text")
        
        # Process text document
        if submit_button and uploaded_file:
            with st.spinner("Processing document for text extraction..."):
                # Update session state collection
                st.session_state.text_collection = text_collection_name
                
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
                
                # Process document with custom collection name
                result = process_document_for_text(
                    uploaded_file, 
                    text_metadata,
                    collection_name=text_collection_name
                )
                
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
    
    # Image Processing Tab
    with image_tab:
        st.subheader("Process Document for Image Extraction")
        
        # Set up columns for form inputs
        col1, col2 = st.columns(2)
        
        # Database configuration
        with col1:
            st.markdown("### Database Configuration")
            mongo_collection = st.text_input(
                "MongoDB Collection Name", 
                value=st.session_state.mongodb_collection,
                help="Name of the MongoDB collection to store images and metadata"
            )
            
            milvus_collection = st.text_input(
                "Milvus Collection Name", 
                value=st.session_state.milvus_collection,
                help="Name of the Milvus collection to store embeddings"
            )
        
        # Document metadata
        with col2:
            st.markdown("### Document Metadata")
            lecture_code = st.text_input(
                "Lecture Code", 
                help="Code identifier for the lecture (e.g., IT3061)"
            )
            
            module_id = st.text_input(
                "Module ID", 
                help="Module identifier (e.g., MDPCC, IOT)"
            )
            
            lecture_number_img = st.number_input(
                "Lecture Number", 
                min_value=0, 
                help="Lecture number in the course sequence"
            )
            
            lecture_title_img = st.text_input(
                "Lecture Title", 
                help="Title of the lecture"
            )
        
        # Advanced configuration (collapsible)
        with st.expander("Advanced Configuration", expanded=False):
            st.subheader("Embedding Configuration")
            
            # Create columns for advanced settings
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
        
        # File uploader for PDF
        uploaded_file_img = st.file_uploader("Upload a PDF document for image extraction", type=["pdf"], key="img_upload")
        
        # Process button
        if st.button("Process PDF for Images"):
            if uploaded_file_img is not None:
                # Update session state collections
                st.session_state.mongodb_collection = mongo_collection
                st.session_state.milvus_collection = milvus_collection
                
                # Display progress information
                with st.spinner("Processing PDF for image extraction..."):
                    # Create config with user-specified parameters
                    config = EmbeddingConfig(
                        # Embedding weights
                        image_weight=image_weight,
                        text_weight=1.0 - image_weight,
                        
                        # Similarity threshold for filtering
                        similarity_threshold=similarity_threshold,
                        
                        # Processing parameters
                        batch_size=batch_size,
                        
                        # Dimension reduction settings
                        use_dim_reduction=use_dim_reduction,
                        output_dim=output_dim,
                        
                        # Alignment settings
                        use_embedding_alignment=use_alignment,
                        
                        # Database settings
                        mongodb_collection=mongo_collection,
                        milvus_collection=milvus_collection
                    )
                    
                    # Prepare metadata
                    image_metadata = {
                        "lecture_code": lecture_code,
                        "module_id": module_id,
                        "lecture_number": int(lecture_number_img) if lecture_number_img else 0,
                        "lecture_title": lecture_title_img,
                        "processed_by": "Multimodal RAG App",
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    # Process document
                    result = process_document_for_images(uploaded_file_img, image_metadata, config)
                    
                    if result["success"]:
                        st.success(result["message"])
                        
                        # If detailed results are available
                        if "result" in result:
                            # Create a DataFrame to display the results
                            result_data = {
                                "Metric": [
                                    "Original Images", 
                                    "Filtered (Similar) Images", 
                                    "Unique Images", 
                                    "Stored in Database"
                                ],
                                "Count": [
                                    result["result"]["num_original_images"],
                                    result["result"]["num_filtered"],
                                    result["result"]["num_unique"],
                                    result["result"]["num_inserted_milvus"]
                                ]
                            }
                            
                            # Display results as a DataFrame
                            st.dataframe(pd.DataFrame(result_data))
                    else:
                        st.error(result["message"])
            else:
                st.warning("Please upload a PDF file first.")

# Learning Analytics Tab
with tab3:
    st.header("Learning Analytics")
    
    if st.session_state.get("student_id"):
        render_analytics_dashboard()
    else:
        st.info("Please log in to view your learning analytics")
        st.warning("Use the Student Personalization panel in the sidebar to log in")

# Anxiety Detection / Wellness Tab 
with tab4:
    # Call the anxiety detection UI renderer from our module
    render_anxiety_detection_ui()

# Add a footer
st.markdown("---")
st.markdown("Enhanced RAG System with Personalization | Created for Education")