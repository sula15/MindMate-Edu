import streamlit as st
import speech_recognition as sr
import torch
import librosa
import numpy as np
import tempfile
from transformers import BertTokenizer
from model import MultimodalFusion

# UI Setup
st.set_page_config(page_title="Anxiety Chatbot", layout="centered")

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MultimodalFusion()
    model.load_state_dict(torch.load("best_multimodal_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Text Preprocessing
def tokenize_and_pad(text, max_length=128):
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

# Predict
anxiety_levels = ["No Anxiety", "Mild Anxiety", "Moderate Anxiety", "Severe Anxiety"]
def get_response(text, audio_file):
    if text:
        input_ids, attention_mask = tokenize_and_pad(text)
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
    return anxiety_levels[predicted_label], msg_map[predicted_label]


st.title("🧠 Anxiety Detection Chatbot")
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("Enter a text or record your voice below:")
st.markdown("<br>", unsafe_allow_html=True)


chat_history = st.session_state.get("chat_history", [])

# --- Initialize session state for text input ---
if "text_input_val" not in st.session_state:
    st.session_state.text_input_val = ""

# Input Area
col1, col2, col3 = st.columns([1, 4, 1])
record = col1.button("🎤 Record")
submitted = col3.button("Send")

voice_path = None
if record:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording... Speak now!")
        audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        st.success("Recording complete!")
        try:
            transcript = recognizer.recognize_google(audio_data)
            st.session_state.transcript = transcript
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_data.get_wav_data())
                voice_path = f.name
        except Exception as e:
            st.error(f"Speech Recognition Error: {e}")
            transcript = ""
else:
    transcript = st.session_state.get("transcript", "")

# Handle submission
if submitted or (transcript and record):
    query = st.session_state.text_input_val or transcript
    audio_file = voice_path if voice_path else None

    if not query and not audio_file:
        st.warning("Please enter a message or record your voice.")
    else:
        anxiety_level, message = get_response(query, audio_file)
        st.markdown(f"**🧑‍🎓 You:** {query}")
        st.markdown(f"**🤖 Bot:** {message}")

    # ✅ Clear input and transcript
    st.session_state.text_input_val = ""
    st.session_state.transcript = ""

# --- Render text input (AFTER clearing) ---
text_input = col2.text_input(
    "",
    key="text_input_val",
    label_visibility="collapsed"
)