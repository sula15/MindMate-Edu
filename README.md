# Enhanced RAG System with Personalization & Anxiety Detection

A comprehensive multimodal Retrieval-Augmented Generation (RAG) system designed for educational environments, featuring intelligent document processing, student personalization, and mental health support.

## 🌟 Features

### 📚 Multimodal RAG System
- **Smart Document Search**: Query both text content and images from educational materials
- **Lecture-Specific Citations**: Automatic referencing with specific lecture numbers and sources
- **Visual Content Retrieval**: Find relevant diagrams, charts, and figures
- **Multi-format Support**: Process PDF and PPTX files with automatic conversion

### 👤 Student Personalization
- **6 Learning Styles**: Detailed, Concise, Bulleted, ELI5, Visual, and Quiz-based responses
- **Learning Analytics**: Track study time, interaction patterns, and performance metrics
- **Personalized Recommendations**: AI-driven suggestions based on learning behavior
- **Progress Monitoring**: Comprehensive dashboard for academic progress

### 🧠 Anxiety Detection & Wellness
- **Multimodal Assessment**: Analyze text input and voice recordings for anxiety detection
- **Real-time Feedback**: Immediate mental health status assessment
- **Wellness Resources**: Personalized coping strategies and recommendations
- **Historical Tracking**: Monitor anxiety levels and wellness trends over time

### 🔧 Administrative Tools
- **Bulk Document Processing**: Efficient upload and processing of educational materials
- **Collection Management**: Easy management of database collections
- **System Monitoring**: Real-time status of all system components
- **Duplicate Prevention**: Automatic detection and prevention of duplicate documents

## 🏗️ System Architecture

```
Frontend (Streamlit) → RAG Engine (LangChain + CLIP) → Databases (MongoDB + Milvus)
                    ↓
Student Personalization ← Analytics Engine → Anxiety Detection Module
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **MongoDB**: Running on localhost:27017
- **Milvus**: Running on localhost:19530 
- **LibreOffice**: For PPTX to PDF conversion
- **CUDA GPU**: Optional, for faster processing
- **Gemini API Key**: Required for LLM functionality

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/enhanced-rag-system.git
cd enhanced-rag-system
```

### 2. Setup Python Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements2.txt
```

### 3. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libreoffice-headless libsndfile1-dev mongodb
sudo systemctl start mongodb
```

#### macOS
```bash
brew install libreoffice libsndfile mongodb-community
brew services start mongodb-community
```

#### Windows
- Install LibreOffice from [official website](https://www.libreoffice.org/)
- Install MongoDB from [official website](https://www.mongodb.com/try/download/community)
- Add LibreOffice to system PATH

### 4. Setup Milvus Database
```bash
# Download and start Milvus using Docker
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d

# Verify installation
docker-compose ps
```

### 5. Configure Environment
Create a `.env` file in the project root:
```env
# Required: Gemini API Key for LLM functionality
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Database configurations (defaults shown)
MONGODB_URI=mongodb://localhost:27017/

# Optional: Custom paths for models
ANXIETY_MODEL_PATH=best_multimodal_model.pth
```

### 6. Download Models
```bash
# Create models directory
mkdir saved_models

# Models will be downloaded automatically on first run
# Or manually place pre-trained models:
# - best_multimodal_model.pth (anxiety detection)
# - svd_model.pkl (recommendation system)
```

## 🎯 Usage

### Main Application
```bash
streamlit run main.py
```
**Access at**: http://localhost:8501

### Admin Interface
```bash
streamlit run admin_interface.py
```
**Access at**: http://localhost:8502

### Standalone Anxiety Detection
```bash
streamlit run anxiety_detection.py
```

## 📖 Getting Started

### For Students

1. **Login/Register**
   - Use sidebar to create your student account
   - Enter Student ID and Name
   - Select your preferred learning style

2. **Start Learning**
   - Ask questions about course materials
   - Get personalized responses based on your learning style
   - View relevant images and diagrams
   - Provide feedback to improve recommendations

3. **Monitor Progress**
   - Check "Learning Analytics" tab for insights
   - View study time and interaction patterns
   - Track satisfaction rates across modules

4. **Wellness Check**
   - Use anxiety detection for mental health assessment
   - Get personalized wellness recommendations
   - Track your mental health trends

### For Administrators

1. **Upload Documents**
   - Access admin interface at port 8502
   - Configure document metadata (Module ID, Lecture Code, etc.)
   - Upload PDF or PPTX files
   - Choose processing type (Text, Images, or Both)

2. **Manage Collections**
   - Monitor database collections
   - View document counts and statistics
   - Manage storage and cleanup

3. **System Monitoring**
   - Check database connection status
   - Monitor system performance
   - View processing logs

## 🎨 Learning Styles

| Style | Description | Best For |
|-------|-------------|----------|
| **Detailed** | Comprehensive explanations with examples and context | In-depth understanding |
| **Concise** | Brief, focused responses with key points | Quick reviews |
| **Bulleted** | Information organized in clear bullet points | Structured learning |
| **ELI5** | Simple explanations with analogies | Complex topics |
| **Visual** | Emphasis on diagrams and visual content | Visual learners |
| **Quiz** | Interactive practice questions and answers | Active learning |

## 🔧 Configuration

### Document Processing Settings
```python
# Advanced configuration options in admin interface
{
    "image_weight": 0.3,           # Weight for image embeddings
    "text_weight": 0.7,            # Weight for text embeddings  
    "similarity_threshold": 0.98,   # Duplicate detection threshold
    "batch_size": 8,               # Processing batch size
    "output_dim": 512,             # Embedding dimensions
    "use_dim_reduction": True,     # Enable dimension reduction
    "use_embedding_alignment": True # Align embeddings for better results
}
```

### Database Collections
- **MongoDB Collections**:
  - `pdf_images_7`: Image metadata and base64 data
  - `student_profiles`: Student information and preferences
  - `student_interactions`: Learning interaction history
  - `anxiety_db`: Mental health assessments

- **Milvus Collections**:
  - `combined_text_collection`: Text embeddings
  - `combined_embeddings_7`: Image embeddings

## 🧪 Testing

### Verify Installation
```bash
# Test database connections
python -c "
import pymongo; 
from pymilvus import connections; 
print('MongoDB:', pymongo.MongoClient().server_info()['version']);
connections.connect('default', host='localhost', port='19530');
print('Milvus: Connected')
"

# Test model loading
python -c "
import torch;
from transformers import BertTokenizer;
print('PyTorch:', torch.__version__);
print('CUDA available:', torch.cuda.is_available());
print('BERT tokenizer loaded successfully')
"
```

### Manual Testing Checklist
- [ ] Upload a sample PDF document via admin interface
- [ ] Create a student profile and select learning style
- [ ] Ask a question about the uploaded document
- [ ] Verify proper citations and image retrieval
- [ ] Test anxiety detection with text input
- [ ] Check learning analytics dashboard

## 🚨 Troubleshooting

### Database Connection Issues
```bash
# Check MongoDB status
sudo systemctl status mongodb
# If not running:
sudo systemctl start mongodb

# Check Milvus status  
docker-compose ps
# If containers are down:
docker-compose up -d
```

### Model Loading Problems
```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Clear cache and reinstall
pip cache purge
pip uninstall torch transformers
pip install torch transformers

# For CUDA issues, install specific version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### PPTX Conversion Errors
```bash
# Ubuntu: Ensure LibreOffice is installed
sudo apt-get install libreoffice-headless
which soffice  # Should return path

# macOS: Check installation
brew list libreoffice
which soffice

# Windows: Add to PATH
# Add C:\Program Files\LibreOffice\program to system PATH
```

### Memory Issues
- Reduce `batch_size` in processing configuration
- Close other applications to free up memory
- Use CPU-only processing if GPU memory is insufficient
- Process documents one at a time for large files

### Performance Optimization
- Enable GPU processing for faster embeddings
- Create database indexes for frequently queried fields
- Regular cleanup of old interaction data
- Monitor disk space for large document collections

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor
- **Network**: Internet connection for model downloads

### Recommended Requirements
- **RAM**: 16GB or higher
- **Storage**: 50GB+ for large document collections
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM
- **CPU**: 8+ cores for faster processing

## 🔒 Security & Privacy

- Student data is stored locally in your MongoDB instance
- No personal data is sent to external services except for LLM API calls
- Anxiety assessments are kept confidential
- All file uploads are processed locally
- Session data is managed securely

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Create an issue on GitHub for bugs or feature requests
- **Documentation**: Check this README and inline code comments
- **Community**: Discussions tab for questions and community support

## 🙏 Acknowledgments

- HuggingFace for transformer models
- LangChain for RAG framework  
- OpenAI for CLIP model
- Milvus for vector database
- Streamlit for web interface
- MongoDB for document storage

---

**Built for educational excellence and student well-being** 🎓💙
