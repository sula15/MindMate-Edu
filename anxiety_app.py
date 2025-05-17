import streamlit as st
import pandas as pd
import time
import datetime
import joblib
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

# MongoDB setup
client = MongoClient("mongodb+srv://sandunikavi09:SLWhTunJGitcSxGO@cluster0.y6ppduj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["anxiety_db"]
collection = db["historical_data"]

# Load saved model and data
svd_model = joblib.load("svd_model.pkl")
suggestion_latent = joblib.load("suggestion_latent.pkl")

# Load training data
df = pd.read_excel("responses.xlsx", sheet_name='Form Responses 1')

feedback_map = {'Highly Satisfied': 3, 'Satisfied': 2, 'Unsatisfied': 1}
for level in ['Severe', 'Moderate', 'Mild', 'None']:
    col = f'Feedback on suggestions ({level})'
    if col in df.columns:
        df[col] = df[col].map(feedback_map)

suggestions = df[['Severe (Anxiety Level)\n', 'Moderate (Anxiety Level)',
                  'Mild (Anxiety Level)', 'None (Anxiety Level)']].stack().reset_index(drop=True)
feedback = df[['Feedback on suggestions (Severe)', 'Feedback on suggestions (Moderate)',
               'Feedback on suggestions (Mild)', 'Feedback on suggestions (None)']].stack().reset_index(drop=True)

interaction_matrix = pd.DataFrame({
    'suggestion': suggestions,
    'feedback': feedback
}).dropna()

suggestion_feedback_matrix = pd.get_dummies(interaction_matrix['suggestion']).multiply(interaction_matrix['feedback'], axis="index").fillna(0)
suggestion_names = suggestion_feedback_matrix.columns.tolist()

def get_recommendations(past_anxiety_level):
    user_vector = np.zeros(len(suggestion_names))
    relevant_suggestions = interaction_matrix[interaction_matrix['feedback'] > 0]['suggestion'].unique()

    for idx, name in enumerate(suggestion_names):
        if name in relevant_suggestions:
            user_vector[idx] = np.random.randint(1, 4)

    user_latent = svd_model.transform([user_vector])
    similarities = cosine_similarity(user_latent, suggestion_latent)
    top_indices = similarities[0].argsort()[::-1][:min(3, len(suggestion_names))]
    return [suggestion_names[i] for i in top_indices if i < len(suggestion_names)]

# Retrieve last past_anxiety_level from MongoDB
def get_last_past_anxiety_level():
    last_feedback = list(collection.find().sort("timestamp", -1).limit(1))  # Convert the cursor to a list
    if len(last_feedback) > 0:
        return last_feedback[0].get("current_anxiety_level", "Mild")
    else:
        return "Mild"
    
# Retrieve current anxiety level from MongoDB
def get_current_anxiety_level():
    current_anxiety = collection.find_one({"type": "current_anxiety_level"})  # Assuming 'type' is the key to differentiate collections
    if current_anxiety:
        return current_anxiety.get("level", "Moderate")
    else:
        return "Moderate"

# Initialize session state
if "countdown_completed" not in st.session_state:
    st.session_state.countdown_completed = False
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# Retrieve past and current anxiety levels
past_anxiety_level = get_last_past_anxiety_level()
current_anxiety_level = get_current_anxiety_level()

# Countdown and show recommendations
if not st.session_state.countdown_completed:
    countdown_time = 10  # seconds
    st.session_state.recommendations = get_recommendations(current_anxiety_level)

    # Placeholders
    progress_bar = st.progress(0)
    countdown_text = st.empty()
    recommendations_placeholder = st.empty()

    # Display recommendations immediately
    with recommendations_placeholder.container():
        st.markdown("### Personalized suggestions for your well-being")
        for rec in st.session_state.recommendations:
            st.write("- " + rec)

    # Start countdown with progress bar
    for second in range(countdown_time):
        percent = (second + 1) / countdown_time
        countdown_text.markdown(f"Please wait: {countdown_time - second} seconds remaining ⏳")
        progress_bar.progress(percent)
        time.sleep(1)

    countdown_text.empty()
    progress_bar.empty()
    st.session_state.countdown_completed = True

# Show feedback form after countdown
if st.session_state.countdown_completed:
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    if not st.session_state.feedback_submitted:
        feedback_rating = st.radio("### How satisfied are you with the recommendations?",
                                   ("Highly Satisfied", "Satisfied", "Unsatisfied"))

        if st.button("Submit Feedback"):
            feedback_data = {
                "timestamp": datetime.datetime.now(),
                "current_anxiety_level": current_anxiety_level,
                "past_anxiety_level": past_anxiety_level,
                "recommendations": st.session_state.recommendations,
                "feedback": feedback_rating
            }
            collection.insert_one(feedback_data)
            st.success("Thank you for your feedback!")
            st.session_state.feedback_submitted = True
    else:
        st.info("✅ You have already submitted your feedback. Thank you!")
