import streamlit as st
import torch
import librosa # type: ignore
import numpy as np
import tempfile
import os
from transformers import BertTokenizer
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import whisper  # Added for audio transcription

# Load environment variables
load_dotenv()

# MongoDB connection
def get_mongo_client():
    """Create and return a MongoDB client connection"""
    # Get MongoDB connection string from environment variables
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        # Fallback for development/testing only
        st.warning("MongoDB URI not found in environment variables. Using fallback connection for development.")
        mongo_uri = "mongodb://localhost:27017/"
        
    client = MongoClient(mongo_uri)
    return client

# Load tokenizer and model
@st.cache_resource
def load_anxiety_model():
    """Load and cache the anxiety detection model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Import inside function to avoid circular imports
    from model import MultimodalFusion
    model = MultimodalFusion()
    model.load_state_dict(torch.load("best_multimodal_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

# ✅ Load whisper model for audio transcription
@st.cache_resource
def load_whisper():
    """Load and cache the Whisper model for speech recognition"""
    return whisper.load_model("base")

# Text Preprocessing
def tokenize_and_pad(text, tokenizer, max_length=128):
    """Tokenize and pad the input text for the model"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['attention_mask']

# Audio Preprocessing
def preprocess_audio(audio_path, sample_rate=16000, n_mels=64, duration=2.5):
    """Process audio file for the model"""
    waveform, sr = librosa.load(audio_path, sr=None)
    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)

    num_samples = int(sample_rate * duration)
    if len(waveform) < num_samples:
        waveform = np.pad(waveform, (0, num_samples - len(waveform)))
    else:
        waveform = waveform[:num_samples]

    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_tensor = torch.tensor(mel_spec_db).unsqueeze(0)
    return mel_spec_tensor

# Predict anxiety level
anxiety_levels = ["No Anxiety", "Mild Anxiety", "Moderate Anxiety", "Severe Anxiety"]
def get_anxiety_response(text, audio_file, tokenizer, model):
    """Predict anxiety level from text and/or audio input"""
    if text:
        input_ids, attention_mask = tokenize_and_pad(text, tokenizer)
        audio_tensor = torch.zeros((1, 64, 79)) if not audio_file else preprocess_audio(audio_file)
    else:
        input_ids, attention_mask = torch.zeros((1, 128), dtype=torch.long), torch.zeros((1, 128), dtype=torch.long)
        audio_tensor = preprocess_audio(audio_file) if audio_file else torch.zeros((1, 64, 79))

    with torch.no_grad():
        output = model(input_ids, attention_mask, audio_tensor)
        predicted_label = torch.argmax(output).item()

    msg_map = [
        "You seem to be doing well. Keep up the good work!",
        "It seems like you are having mild anxiety. Consider talking to a counselor or engaging in relaxation techniques.",
        "It seems like you are experiencing moderate anxiety. It might be helpful to talk to a mental health professional.",
        "It seems like you are experiencing severe anxiety. It's important to seek help from a professional as soon as possible."
    ]
        # Add this line before returning
    st.session_state.last_detected_anxiety_level = anxiety_levels[predicted_label]
    
    return anxiety_levels[predicted_label], msg_map[predicted_label]

# Save to MongoDB
def save_anxiety_assessment(query, anxiety_level, has_audio=False):
    """Save anxiety assessment to MongoDB"""
    client = get_mongo_client()
    db = client["anxiety_db"]
    collection = db["current_anxiety_level"]
    
    data = {
        "type": "current_anxiety_level",  # Added type field to match what get_current_anxiety_level() expects
        "level": anxiety_level,
        "query": query,
        "has_audio": has_audio,
        "timestamp": datetime.now()
    }
    collection.insert_one(data)
    client.close()

# ✅ FIXED: New audio recording function using Streamlit's audio_input
def process_audio_input(audio_file, whisper_model):
    """Process audio input using Streamlit's audio_input widget and Whisper"""
    if audio_file is None:
        return None, ""
    
    # Ensure directory for storing audio responses
    AUDIO_SAVE_PATH = "audio_responses"
    os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)
    
    try:
        # Define the save path for the recorded audio
        audio_save_path = os.path.join(AUDIO_SAVE_PATH, "recorded_response.wav")

        # Save the recorded audio file locally
        with open(audio_save_path, "wb") as f:
            f.write(audio_file.getbuffer())

        # Transcribe the recorded audio using Whisper
        transcription = whisper_model.transcribe(audio_save_path)
        transcript = transcription["text"]
        
        return audio_save_path, transcript
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, ""

# Get anxiety resources based on level
def get_anxiety_resources(anxiety_level):
    """Return resources based on anxiety level"""
    if anxiety_level == "No Anxiety":
        return None
    elif anxiety_level == "Mild Anxiety":
        return """
        - Try some deep breathing exercises
        - Take a short walk outside
        - Practice mindfulness meditation for 5-10 minutes
        - Consider talking to a friend or family member
        """
    elif anxiety_level == "Moderate Anxiety":
        return """
        - Consider scheduling a session with a counselor
        - Try the 5-4-3-2-1 grounding technique
        - Use progressive muscle relaxation
        - Limit caffeine and alcohol intake
        - Maintain regular sleep schedule
        """
    else:  # Severe Anxiety
        return """
        - Please reach out to a mental health professional
        - Contact your university's counseling services
        - National Anxiety Helpline: 1-800-XXX-XXXX
        - Crisis Text Line: Text HOME to 741741
        - Consider scheduling an appointment with your doctor
        """

# Render anxiety detection UI
def render_anxiety_detection_ui():
    """Render the complete anxiety detection UI component"""
    st.title("🧠 Anxiety Detection & Wellness Check")
    st.markdown("This tool can help assess your anxiety levels through text or voice input.")
    
    # Initialize session state variables if they don't exist
    if 'anxiety_chat_history' not in st.session_state:
        st.session_state.anxiety_chat_history = []
        
    # ✅ FIXED: Updated session state for new audio handling
    if "text_input_val" not in st.session_state:
        st.session_state.text_input_val = ""
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "voice_path" not in st.session_state:
        st.session_state.voice_path = ""
    
    # Get the anxiety model, tokenizer, and whisper model
    try:
        tokenizer, anxiety_model = load_anxiety_model()
        whisper_model = load_whisper()  # ✅ FIXED: Load Whisper model
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.warning("The anxiety detection functionality is not available. Please ensure the model files are present.")
        model_loaded = False
    
    if model_loaded:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("Share how you're feeling through text or voice:")
        
        # Display previous interactions
        with st.container(height=400):
            for entry in st.session_state.anxiety_chat_history:
                user_text = entry.get("input_text", "")
                anxiety_level = entry.get("anxiety_level", "")
                response = entry.get("response", "")
                
                if user_text:
                    with st.chat_message("user"):
                        st.markdown(user_text)
                
                with st.chat_message("assistant"):
                    st.markdown(f"**Assessed Level:** {anxiety_level}")
                    st.markdown(response)
        
        # ✅ FIXED: New input layout with Streamlit's audio_input
        col1, col2, col3 = st.columns([1, 4, 1])
        
        # ✅ FIXED: Replace record button with audio_input widget
        audio_file = col1.audio_input("🎤 Record")
        submit = col3.button("Send")
        
        # ✅ FIXED: Process audio input using Whisper
        voice_path = None
        if audio_file is not None:
            st.audio(audio_file)  # Play back the recorded audio
            voice_path, transcript = process_audio_input(audio_file, whisper_model)
            if transcript:
                st.session_state.transcript = transcript
                st.session_state.voice_path = voice_path
                st.session_state.text_input_val = transcript
        
        # Text input
        text_input = col2.text_input("Your message", key="anxiety_text_input", 
                                    value=st.session_state.text_input_val)
        
        # Handle submission
        if submit:
            # Prioritize voice input if available
            if st.session_state.transcript and st.session_state.voice_path:
                user_query = st.session_state.transcript
                audio_file_path = st.session_state.voice_path
            else:
                user_query = text_input
                audio_file_path = None
            
            if not user_query and not audio_file_path:
                st.warning("Please enter a message or record your voice.")
            else:
                # Get anxiety assessment
                with st.spinner("Analyzing..."):
                    try:
                        anxiety_level, response_message = get_anxiety_response(
                            user_query, audio_file_path, tokenizer, anxiety_model
                        )
                        
                        # Add to chat history
                        st.session_state.anxiety_chat_history.append({
                            "input_text": user_query,
                            "anxiety_level": anxiety_level,
                            "response": response_message,
                            "timestamp": datetime.now().isoformat(),
                            "has_audio": audio_file_path is not None
                        })
                        
                        # If user is logged in, store in student profile
                        student_id = st.session_state.get("student_id")
                        if student_id:
                            try:
                                from student_personalization import StudentPersonalization, PersonalizationConfig
                                personalization = StudentPersonalization(PersonalizationConfig())
                                # Store anxiety assessment in student profile
                                personalization.db[personalization.config.profiles_collection].update_one(
                                    {"_id": student_id},
                                    {"$push": {"anxiety_assessments": {
                                        "timestamp": datetime.now(),
                                        "level": anxiety_level,
                                        "input_text": user_query,
                                        "has_audio": audio_file_path is not None
                                    }}}
                                )
                            except Exception as profile_err:
                                st.error(f"Error storing anxiety assessment in profile: {profile_err}")
                        
                        # Also save to the anxiety database
                        try:
                            save_anxiety_assessment(user_query, anxiety_level, audio_file_path is not None)
                        except Exception as db_err:
                            st.error(f"Error saving to anxiety database: {db_err}")
                        
                        # Display results
                        with st.chat_message("user"):
                            st.markdown(user_query)
                        
                        with st.chat_message("assistant"):
                            st.markdown(f"**Assessed Level:** {anxiety_level}")
                            st.markdown(response_message)
                            
                            # Add resources based on anxiety level
                            resources = get_anxiety_resources(anxiety_level)
                            if resources:
                                st.markdown("### Helpful Resources")
                                st.markdown(resources)
                    except Exception as assess_error:
                        st.error(f"Error during anxiety assessment: {assess_error}")
                
                # ✅ FIXED: Clear session state after submission
                st.session_state.text_input_val = ""
                st.session_state.transcript = ""
                st.session_state.voice_path = ""
                        
                # Rerun to update UI
                st.rerun()
        
        # Add some information about the assessment
        with st.expander("About Anxiety Assessment", expanded=False):
            st.markdown("""
            ### How This Works
            
            This anxiety assessment tool uses a multimodal approach to detect potential signs of anxiety:
            
            1. **Text Analysis**: When you type or speak, the system analyzes linguistic patterns associated with different anxiety levels.
            
            2. **Voice Analysis** (when using the microphone): The system can detect acoustic patterns in your voice that may indicate anxiety.
            
            3. **Combined Assessment**: The results from both analyses are combined for a more accurate assessment.
            
            ### Privacy Note
            
            Your assessments are kept private. If you're logged in, your assessment history is stored in your profile to help track patterns over time, which can be viewed in the Analytics tab.
            
            ### Important Disclaimer
            
            This tool is not a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing severe anxiety, please consult with a qualified healthcare provider.
            """)
            
        # Add a section for wellness tips
        with st.expander("Wellness Tips", expanded=False):
            st.markdown("""
            ### Quick Anxiety-Reduction Techniques
            
            #### 1. Deep Breathing (5-5-5)
            - Inhale deeply for 5 seconds
            - Hold for 5 seconds
            - Exhale for 5 seconds
            - Repeat 5 times
            
            #### 2. Progressive Muscle Relaxation
            - Tense and then relax each muscle group
            - Start from your toes and work up to your head
            - Hold tension for 5 seconds, then release for 10 seconds
            
            #### 3. 5-4-3-2-1 Grounding
            - 5 things you can see
            - 4 things you can touch
            - 3 things you can hear
            - 2 things you can smell
            - 1 thing you can taste
            
            #### 4. Mindfulness Minute
            - Focus only on your breathing for 60 seconds
            - Notice thoughts without judgment, then return to breath
            - Say "breathing in" and "breathing out" mentally
            """)