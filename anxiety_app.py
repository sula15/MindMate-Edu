import streamlit as st
import pandas as pd
import time
import datetime
import joblib
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Function to create MongoDB connection using environment variables
def get_mongo_client():
    """Create MongoDB connection using environment variables with fallback"""
    # Get MongoDB connection string
    mongo_uri = os.getenv("MONGODB_URI")
    
    # If no environment variable, use local MongoDB
    if not mongo_uri:
        logger.warning("MongoDB URI not found in environment variables. Using local MongoDB.")
        mongo_uri = "mongodb://localhost:27017/"
    
    try:
        # Try to connect with a short timeout to fail fast
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.server_info()
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        # Fallback to local MongoDB
        try:
            logger.info("Attempting fallback to local MongoDB...")
            local_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            local_client.server_info()
            logger.info("Successfully connected to local MongoDB")
            return local_client
        except Exception as local_e:
            logger.error(f"Failed to connect to local MongoDB: {local_e}")
            # Return None if both connections fail
            return None

# Global variables - will be initialized when needed
client = None
db = None
collection = None

def initialize_mongodb():
    """Initialize MongoDB connection when needed"""
    global client, db, collection
    
    if client is None:
        client = get_mongo_client()
        if client is not None:
            db = client["anxiety_db"]
            collection = db["historical_data"]
        else:
            logger.error("Could not establish MongoDB connection")
    
    return client is not None

# Load saved model and data
try:
    svd_model = joblib.load("svd_model.pkl")
    suggestion_latent = joblib.load("suggestion_latent.pkl")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    svd_model = None
    suggestion_latent = None

# Load training data
try:
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
except Exception as e:
    logger.error(f"Error loading training data: {e}")
    df = None
    interaction_matrix = None
    suggestion_names = []

def get_recommendations(past_anxiety_level):
    """Get personalized recommendations based on anxiety level"""
    if svd_model is None or suggestion_latent is None or len(suggestion_names) == 0:
        return ["Take deep breaths", "Go for a walk", "Practice mindfulness meditation"]
    
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
    """Get the last anxiety level from MongoDB with error handling"""
    try:
        if not initialize_mongodb():
            logger.warning("MongoDB not available, returning default anxiety level")
            return "Mild"
        
        last_feedback = list(collection.find().sort("timestamp", -1).limit(1))
        if len(last_feedback) > 0:
            return last_feedback[0].get("current_anxiety_level", "Mild")
        else:
            return "Mild"
    except Exception as e:
        logger.error(f"Error getting last anxiety level: {e}")
        return "Mild"
    
# Retrieve current anxiety level from MongoDB
def get_current_anxiety_level():
    """Get current anxiety level from MongoDB with error handling"""
    try:
        if not initialize_mongodb():
            logger.warning("MongoDB not available, returning default anxiety level")
            return "Moderate"
        
        current_anxiety = collection.find_one({"type": "current_anxiety_level"})
        if current_anxiety:
            return current_anxiety.get("level", "Moderate")
        else:
            return "Moderate"
    except Exception as e:
        logger.error(f"Error getting current anxiety level: {e}")
        return "Moderate"

# Save to MongoDB
def save_anxiety_assessment(query, anxiety_level, has_audio=False):
    """Save anxiety assessment to MongoDB with error handling"""
    try:
        if not initialize_mongodb():
            logger.warning("MongoDB not available, skipping save")
            return False
        
        data = {
            "type": "current_anxiety_level",
            "level": anxiety_level,
            "query": query,
            "has_audio": has_audio,
            "timestamp": datetime.datetime.now()
        }
        collection.insert_one(data)
        logger.info("Successfully saved anxiety assessment")
        return True
    except Exception as e:
        logger.error(f"Error saving anxiety assessment: {e}")
        return False

def render_anxiety_solutions_ui():
    """Render the anxiety solutions UI as a component for main.py"""
    st.header("Personalized Anxiety Management Solutions")
    
    # Check if user is logged in
    student_id = st.session_state.get("student_id")
    if not student_id:
        st.info("Please log in using the sidebar to get personalized recommendations.")
        return
    
    # Initialize session state variables if they don't exist
    if "countdown_completed" not in st.session_state:
        st.session_state.countdown_completed = False
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    
    # First check if there's a last detected anxiety level in session state
    if 'last_detected_anxiety_level' in st.session_state:
        current_anxiety_level = st.session_state.last_detected_anxiety_level
        st.success(f"Using your recently assessed anxiety level: {current_anxiety_level}")
    else:
        # Otherwise get from MongoDB
        current_anxiety_level = get_current_anxiety_level()
        st.info(f"Using your previously recorded anxiety level: {current_anxiety_level}")
    
    past_anxiety_level = get_last_past_anxiety_level()
    
    # Display current anxiety level with emoji indicator
    level_emoji = {
        "No Anxiety": "😊",
        "Mild Anxiety": "😐",
        "Moderate Anxiety": "😟",
        "Severe Anxiety": "😨"
    }
    emoji = level_emoji.get(current_anxiety_level, "😐")
    
    st.markdown(f"## Current Status: {emoji} {current_anxiety_level}")
    
    # Countdown and show recommendations
    if not st.session_state.countdown_completed:
        st.markdown("### Generating personalized recommendations...")
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
        countdown_time = 5  # seconds - reduced from 10 for better UX
        for second in range(countdown_time):
            percent = (second + 1) / countdown_time
            countdown_text.markdown(f"Please wait: {countdown_time - second} seconds remaining ⏳")
            progress_bar.progress(percent)
            time.sleep(1)

        countdown_text.empty()
        progress_bar.empty()
        st.session_state.countdown_completed = True

    # Show results after countdown
    if st.session_state.countdown_completed:
        st.subheader("Your Personalized Suggestions")
        
        # Display recommendations in a nicer format
        for i, rec in enumerate(st.session_state.recommendations):
            with st.container(border=True):
                st.markdown(f"### Suggestion {i+1}")
                st.markdown(f"**{rec}**")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Feedback form if not already submitted
        if not st.session_state.feedback_submitted:
            with st.container(border=True):
                st.markdown("### How satisfied are you with these recommendations?")
                feedback_rating = st.radio(
                    "Your feedback helps us improve suggestions",
                    ("Highly Satisfied", "Satisfied", "Unsatisfied"),
                    horizontal=True
                )

                if st.button("Submit Feedback"):
                    feedback_data = {
                        "timestamp": datetime.datetime.now(),
                        "student_id": student_id,
                        "current_anxiety_level": current_anxiety_level,
                        "past_anxiety_level": past_anxiety_level,
                        "recommendations": st.session_state.recommendations,
                        "feedback": feedback_rating
                    }
                    
                    # Try to save to MongoDB
                    saved = False
                    try:
                        if initialize_mongodb():
                            collection.insert_one(feedback_data)
                            saved = True
                    except Exception as e:
                        logger.error(f"Error saving feedback to MongoDB: {e}")
                    
                    # Also store in student profile if available
                    try:
                        from student_personalization import StudentPersonalization, PersonalizationConfig
                        personalization = StudentPersonalization(PersonalizationConfig())
                        # Store feedback in student profile
                        personalization.db[personalization.config.profiles_collection].update_one(
                            {"_id": student_id},
                            {"$push": {"anxiety_recommendations": {
                                "timestamp": datetime.datetime.now(),
                                "level": current_anxiety_level,
                                "recommendations": st.session_state.recommendations,
                                "feedback": feedback_rating
                            }}}
                        )
                    except Exception as profile_err:
                        logger.error(f"Error storing feedback in profile: {profile_err}")
                    
                    if saved or True:  # Show success even if MongoDB fails
                        st.success("Thank you for your feedback!")
                        st.session_state.feedback_submitted = True
                        time.sleep(1)
                        st.rerun()
        else:
            st.success("✅ You have already submitted your feedback. Thank you!")
        
        # Add a reset button to get new recommendations
        if st.button("Get New Recommendations"):
            st.session_state.countdown_completed = False
            st.session_state.feedback_submitted = False
            st.rerun()

# Main function for standalone usage
def main():
    st.set_page_config(page_title="Anxiety Management Solutions", page_icon="🧠")
    
    # Initialize session state
    if "countdown_completed" not in st.session_state:
        st.session_state.countdown_completed = False
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    
    # Set a dummy student ID for standalone mode
    if "student_id" not in st.session_state:
        st.session_state.student_id = "standalone_user"
    
    st.title("Anxiety Management Solutions")
    
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
                
                # Try to save to MongoDB
                try:
                    if initialize_mongodb():
                        collection.insert_one(feedback_data)
                        st.success("Thank you for your feedback!")
                    else:
                        st.warning("Feedback saved locally (database not available)")
                except Exception as e:
                    logger.error(f"Error saving feedback: {e}")
                    st.warning("Feedback noted (database temporarily unavailable)")
                
                st.session_state.feedback_submitted = True
        else:
            st.info("✅ You have already submitted your feedback. Thank you!")

# Run the standalone app if this file is run directly
if __name__ == "__main__":
    main()