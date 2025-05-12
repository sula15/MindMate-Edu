"""
Semantic PDF Chunking Module

This module provides functionality to process PDFs using semantic chunking based on the document structure.
It detects headings based on font sizes and styling, then chunks documents according to these semantic boundaries.
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

# LangChain imports
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

# PDF processing
import fitz  # PyMuPDF
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticPDFProcessor:
    """Process PDFs using semantic chunking based on document structure"""
    
    def __init__(
        self,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        min_chunk_size: int = 500,
        max_chunk_size: int = 4000,
        chunk_overlap: int = 300,
        heading_detection_threshold: float = 1.2,  # Font size ratio threshold for heading detection
    ):
        """
        Initialize the semantic PDF processor.
        
        Args:
            headers_to_split_on: List of tuples (markdown_header, name) to split on
            min_chunk_size: Minimum size of chunks in characters
            max_chunk_size: Maximum size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            heading_detection_threshold: Ratio of font size to normal text to be considered a heading
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_detection_threshold = heading_detection_threshold
        
        # Default header hierarchy if none provided
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        # Fallback splitter for large sections
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def detect_headings_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF with detected headings converted to Markdown format
        to enable semantic chunking.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Markdown formatted text with headings
        """
        logger.info(f"Detecting headings from {pdf_path}")
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        markdown_text = ""
        prev_font_size = None
        prev_font_name = None
        prev_is_bold = False
        
        # Track the normal text font size
        font_sizes = []
        
        # First pass to determine normal text font size
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Skip if likely not main content (too small or unusual)
                            if span["size"] > 5:  # Filter out very small text
                                font_sizes.append(span["size"])
        
        # Calculate median font size as the reference for normal text
        font_sizes.sort()
        if font_sizes:
            normal_font_size = font_sizes[len(font_sizes) // 2]
            logger.info(f"Detected normal font size: {normal_font_size:.2f}")
        else:
            normal_font_size = 12  # Default fallback
            logger.warning(f"Could not detect font size, using default: {normal_font_size}")
        
        # Second pass to extract content with heading detection
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        is_heading = False
                        heading_level = 0
                        
                        # Extract text and detect if this line is a heading
                        for span in line["spans"]:
                            current_font_size = span["size"]
                            current_font_name = span["font"]
                            is_bold = "bold" in current_font_name.lower() or "heavy" in current_font_name.lower()
                            
                            # Check if this span might be a heading
                            size_ratio = current_font_size / normal_font_size
                            
                            # Determine heading level based on font size ratio and styling
                            if size_ratio > self.heading_detection_threshold:
                                is_heading = True
                                if size_ratio > 1.8:  # h1
                                    heading_level = 1
                                elif size_ratio > 1.5:  # h2
                                    heading_level = 2
                                elif size_ratio > 1.2:  # h3
                                    heading_level = 3
                                
                                # Additional boost for bold text
                                if is_bold and heading_level > 1:
                                    heading_level -= 1  # Bold makes it one level higher
                                
                            line_text += span["text"]
                            
                            # Track for change detection
                            prev_font_size = current_font_size
                            prev_font_name = current_font_name
                            prev_is_bold = is_bold
                        
                        # Clean the line text
                        line_text = line_text.strip()
                        
                        # Skip empty lines
                        if not line_text:
                            continue
                        
                        # Add markdown heading syntax if detected as heading
                        if is_heading and line_text:
                            # Ensure there's a line break before headings (except at beginning of document)
                            if markdown_text and not markdown_text.endswith('\n\n'):
                                markdown_text += '\n\n'
                            
                            heading_prefix = '#' * heading_level
                            markdown_text += f"{heading_prefix} {line_text}\n\n"
                        else:
                            markdown_text += line_text + " "
        
        # Clean up any trailing whitespace or excessive newlines
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
        markdown_text = markdown_text.strip()
        
        logger.info(f"Converted PDF to markdown with detected headings")
        return markdown_text
    
    def split_text(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Split the markdown text by headers to create chunks with hierarchical structure.
        
        Args:
            markdown_text: Markdown text with headers
            
        Returns:
            List of dictionaries with content and metadata
        """
        logger.info("Splitting text by headers")
        
        # Use LangChain's MarkdownHeaderTextSplitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        
        # Get initial splits based on headers
        header_splits = markdown_splitter.split_text(markdown_text)
        
        # Process the splits to ensure they're not too large
        final_splits = []
        
        for doc in header_splits:
            content = doc.page_content
            metadata = doc.metadata
            
            # If content is smaller than max size, keep as is
            if len(content) <= self.max_chunk_size:
                final_splits.append({"content": content, "metadata": metadata})
            else:
                # Further split large content while preserving metadata
                sub_chunks = self.fallback_splitter.split_text(content)
                for i, chunk in enumerate(sub_chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i + 1
                    chunk_metadata["total_chunks"] = len(sub_chunks)
                    final_splits.append({"content": chunk, "metadata": chunk_metadata})
        
        logger.info(f"Created {len(final_splits)} semantic chunks")
        return final_splits
    
    def process_pdf(self, pdf_path: str, additional_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a PDF file by extracting text with heading structure and creating semantic chunks.
        
        Args:
            pdf_path: Path to the PDF file
            additional_metadata: Additional metadata to add to each chunk
            
        Returns:
            List of dictionaries with content and metadata
        """
        # Extract text with heading information
        markdown_text = self.detect_headings_from_pdf(pdf_path)
        
        # Split by semantic structure
        chunks = self.split_text(markdown_text)
        
        # Add additional metadata to each chunk
        if additional_metadata:
            for chunk in chunks:
                chunk["metadata"].update(additional_metadata)
        
        return chunks


def process_document_with_semantic_chunking(
    file_obj: Any,
    metadata: Dict[str, Any],
    collection_name: str = "combined_text_collection",
    min_chunk_size: int = 500,
    max_chunk_size: int = 3000,
    chunk_overlap: int = 300,
    embedding_model: str = "all-MiniLM-L6-v2",
    milvus_host: str = "localhost",
    milvus_port: str = "19530"
) -> Dict[str, Any]:
    """
    Process a document using semantic chunking based on document structure.
    
    Args:
        file_obj: File object (from streamlit upload or bytes)
        metadata: Document metadata to include with chunks
        collection_name: Milvus collection name
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        chunk_overlap: Chunk overlap
        embedding_model: HuggingFace embedding model name
        milvus_host: Milvus host
        milvus_port: Milvus port
        
    Returns:
        Dictionary with success status, message, and chunk information
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Handle both Streamlit UploadedFile and bytes
            if hasattr(file_obj, 'getvalue'):
                # Streamlit UploadedFile
                tmp_file.write(file_obj.getvalue())
            elif isinstance(file_obj, bytes):
                # Bytes
                tmp_file.write(file_obj)
            else:
                # File path
                with open(file_obj, 'rb') as f:
                    tmp_file.write(f.read())
                
            pdf_path = tmp_file.name
        
        # Initialize the semantic processor
        processor = SemanticPDFProcessor(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Process the PDF to get semantic chunks
        chunks = processor.process_pdf(pdf_path, additional_metadata=metadata)
        
        # Convert to LangChain Document format for vectorization
        documents = [
            Document(
                page_content=chunk["content"],
                metadata=chunk["metadata"]
            ) for chunk in chunks
        ]
        
        # Create embeddings and store in Milvus
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Store in Milvus with configurable collection name
        vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            connection_args={"host": milvus_host, "port": milvus_port}
        )
        
        # Clean up temporary file
        try:
            os.unlink(pdf_path)
        except:
            pass
        
        # Extract document structure for reporting
        chunk_structure = {}
        for chunk in chunks:
            if "Header 1" in chunk["metadata"]:
                h1 = chunk["metadata"]["Header 1"]
                if h1 not in chunk_structure:
                    chunk_structure[h1] = []
                
                if "Header 2" in chunk["metadata"]:
                    h2 = chunk["metadata"]["Header 2"]
                    if h2 not in chunk_structure[h1]:
                        chunk_structure[h1].append(h2)
        
        return {
            "success": True,
            "chunk_count": len(chunks),
            "message": f"Successfully processed and indexed {len(chunks)} semantic chunks in collection '{collection_name}'",
            "chunk_structure": chunk_structure
        }
        
    except Exception as e:
        logger.error(f"Error processing document with semantic chunking: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }


# Optional function for direct command line usage
def process_pdf_file(
    pdf_path: str,
    metadata: Dict[str, Any],
    output_collection: str = "semantic_pdf_chunks",
    max_chunk_size: int = 3000
) -> Dict[str, Any]:
    """
    Convenience function to process a PDF file directly.
    
    Args:
        pdf_path: Path to PDF file
        metadata: Metadata to include with chunks
        output_collection: Milvus collection name
        max_chunk_size: Maximum chunk size
        
    Returns:
        Dictionary with success status and processing info
    """
    return process_document_with_semantic_chunking(
        pdf_path,
        metadata,
        collection_name=output_collection,
        max_chunk_size=max_chunk_size
    )


# Example usage
if __name__ == "__main__":
    # Example usage when the module is run directly
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python semantic_chunker.py <pdf_file_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Example metadata
    metadata = {
        "module_code": "EXAMPLE",
        "module_name": "Example Module",
        "lecture_number": 1,
        "lecture_title": "Introduction to Semantic Chunking",
        "source_type": "Lecture",
        "source": os.path.basename(pdf_path)
    }
    
    result = process_pdf_file(pdf_path, metadata)
    
    if result["success"]:
        print(f"Success: {result['message']}")
        print(f"Document structure:")
        for h1, h2_list in result.get("chunk_structure", {}).items():
            print(f"- {h1}")
            for h2 in h2_list:
                print(f"  - {h2}")
    else:
        print(f"Error: {result['message']}")
