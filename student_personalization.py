import pymongo
import pandas as pd
import numpy as np
import datetime
import uuid
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
import sys

# Try to import Streamlit, gracefully handle if not available
try:
    import streamlit as st
except ImportError:
    st = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LearningStyle:
    """Learning style definition"""
    id: str
    name: str
    description: str
    prompt_template: str


@dataclass
class PersonalizationConfig:
    """Configuration for personalization module"""
    # MongoDB settings
    mongodb_uri: str = "mongodb://localhost:27017/"
    student_db: str = "student_analytics"
    profiles_collection: str = "student_profiles"
    interactions_collection: str = "student_interactions"
    analytics_collection: str = "student_analytics"
    
    # Learning styles
    learning_styles: List[LearningStyle] = field(default_factory=list)
    
    # Analytics settings
    analytics_update_frequency: int = 24  # Hours between analytics updates
    time_spent_threshold: int = 300  # Max seconds to consider for a query (5 min)
    
    # Session timeout
    session_timeout: int = 1800  # 30 minutes in seconds
    
    def __post_init__(self):
        """Initialize default learning styles if none provided"""
        if not self.learning_styles:
            self.learning_styles = [
                LearningStyle(
                    id="detailed",
                    name="Detailed Explanation",
                    description="Comprehensive, thorough explanations with examples and deep context",
                    prompt_template="Please provide a detailed and comprehensive explanation with examples. " +
                                  "Include background context, theory, and practical examples. " +
                                  "Structure your answer with clear sections and thorough explanations."
                ),
                LearningStyle(
                    id="concise",
                    name="Concise Summary",
                    description="Brief, to-the-point explanations focusing on key concepts",
                    prompt_template="Please provide a concise, to-the-point explanation. " +
                                  "Focus only on the most important concepts and key takeaways. " +
                                  "Keep your answer brief and direct without unnecessary details."
                ),
                LearningStyle(
                    id="bulleted",
                    name="Bulleted List",
                    description="Information organized in easy-to-scan bullet points",
                    prompt_template="Please format your response as a bulleted list. " +
                                  "Organize information in clear, scannable bullet points with hierarchical structure. " +
                                  "Use headings where appropriate and keep each bullet point focused on a single concept."
                ),
                LearningStyle(
                    id="eli5",
                    name="Explain Like I'm 5",
                    description="Simple explanations using basic language and analogies",
                    prompt_template="Please explain this concept as if I'm a beginner with no background knowledge. " +
                                  "Use simple language, analogies, and everyday examples I can relate to. " +
                                  "Avoid technical jargon and complex terminology unless absolutely necessary and defined."
                ),
                LearningStyle(
                    id="visual",
                    name="Visual Learning",
                    description="Focus on diagrams, charts, and visual explanations",
                    prompt_template="Please emphasize visual examples in your explanation. " +
                                  "Refer to any diagrams, charts, or images that help illustrate the concepts. " +
                                  "Describe visual relationships and spatial information clearly, and suggest visual ways to remember the information."
                ),
                LearningStyle(
                    id="quiz",
                    name="Quiz-Based",
                    description="Information presented through practice questions",
                    prompt_template="Please structure your response as a series of practice questions with answers. " +
                                  "First, provide a brief overview of the key concepts, then present 3-5 questions that test understanding. " +
                                  "Include the answers with explanations at the end."
                )
            ]


class StudentPersonalization:
    """
    Manages student personalization features including learning style preferences,
    interaction tracking, and analytics generation
    """
    
    def __init__(self, config: Optional[PersonalizationConfig] = None):
        """Initialize with configuration"""
        self.config = config or PersonalizationConfig()
        
        # Initialize MongoDB connections
        self._init_db_connections()
        
        # Initialize session state if using with Streamlit
        self._init_session_state()
    
    def _init_db_connections(self):
        """Initialize database connections"""
        try:
            self.client = pymongo.MongoClient(self.config.mongodb_uri)
            self.db = self.client[self.config.student_db]
            
            # Create collections if they don't exist
            if self.config.profiles_collection not in self.db.list_collection_names():
                self.db.create_collection(self.config.profiles_collection)
            
            if self.config.interactions_collection not in self.db.list_collection_names():
                self.db.create_collection(self.config.interactions_collection)
                # Create indexes for efficient querying
                self.db[self.config.interactions_collection].create_index([("student_id", 1)])
                self.db[self.config.interactions_collection].create_index([("timestamp", -1)])
                self.db[self.config.interactions_collection].create_index([("module_id", 1)])
            
            if self.config.analytics_collection not in self.db.list_collection_names():
                self.db.create_collection(self.config.analytics_collection)
                # Create indexes for analytics
                self.db[self.config.analytics_collection].create_index([("student_id", 1)])
                self.db[self.config.analytics_collection].create_index([("module_id", 1)])
            
            logger.info("Successfully connected to MongoDB and initialized collections")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables if available"""
        # Check if we're running in a Streamlit context
        if 'streamlit' in sys.modules and st is not None:
            # Only initialize if session_state is available
            if not hasattr(st, 'session_state'):
                return
                
            # Set default values if keys don't exist
            if 'student_id' not in st.session_state:
                st.session_state.student_id = None
            if 'learning_style' not in st.session_state:
                st.session_state.learning_style = None
            if 'interaction_start_time' not in st.session_state:
                st.session_state.interaction_start_time = None
            if 'current_interaction_id' not in st.session_state:
                st.session_state.current_interaction_id = None
            if 'last_activity_time' not in st.session_state:
                st.session_state.last_activity_time = datetime.datetime.now()
    
    def get_learning_styles(self) -> List[Dict[str, str]]:
        """
        Get all available learning styles
        
        Returns:
            List of learning style dictionaries with id, name, and description
        """
        return [
            {
                "id": style.id,
                "name": style.name,
                "description": style.description
            }
            for style in self.config.learning_styles
        ]
    
    def get_learning_style_by_id(self, style_id: str) -> Optional[LearningStyle]:
        """
        Get a learning style by ID
        
        Args:
            style_id: The ID of the learning style
            
        Returns:
            LearningStyle object or None if not found
        """
        for style in self.config.learning_styles:
            if style.id == style_id:
                return style
        return None
    
    def create_student_profile(self, student_id: str, name: str, email: str = None, 
                              default_learning_style: str = "detailed") -> Dict[str, Any]:
        """
        Create a new student profile
        
        Args:
            student_id: Unique identifier for the student
            name: Student's name
            email: Student's email (optional)
            default_learning_style: Default learning style ID
            
        Returns:
            The created student profile document
        """
        # Check if student already exists
        existing_student = self.db[self.config.profiles_collection].find_one({"_id": student_id})
        if existing_student:
            logger.info(f"Student profile already exists for {student_id}")
            return existing_student
        
        # Create new student profile
        now = datetime.datetime.now()
        student_profile = {
            "_id": student_id,
            "name": name,
            "email": email,
            "default_learning_style": default_learning_style,
            "created_at": now,
            "last_active": now,
            "preferences": {},
            "modules_accessed": [],
            "total_interactions": 0
        }
        
        try:
            self.db[self.config.profiles_collection].insert_one(student_profile)
            logger.info(f"Created new student profile for {student_id}")
            return student_profile
        except Exception as e:
            logger.error(f"Error creating student profile: {e}")
            raise
    
    def get_student_profile(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a student profile
        
        Args:
            student_id: ID of the student
            
        Returns:
            Student profile document or None if not found
        """
        try:
            profile = self.db[self.config.profiles_collection].find_one({"_id": student_id})
            return profile
        except Exception as e:
            logger.error(f"Error retrieving student profile: {e}")
            return None
        
    
    def update_learning_style_preference(self, student_id: str, learning_style_id: str) -> bool:
        """
        Update a student's preferred learning style
        
        Args:
            student_id: ID of the student
            learning_style_id: ID of the preferred learning style
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if learning style exists
            if not self.get_learning_style_by_id(learning_style_id):
                logger.error(f"Learning style {learning_style_id} not found")
                return False
            
            # Update student profile
            result = self.db[self.config.profiles_collection].update_one(
                {"_id": student_id},
                {
                    "$set": {
                        "default_learning_style": learning_style_id,
                        "last_active": datetime.datetime.now()
                    }
                }
            )
            
            # Update session state if using Streamlit
            if 'streamlit' in sys.modules and st is not None:
                try:
                    if hasattr(st, 'session_state') and hasattr(st.session_state, 'student_id'):
                        st.session_state.learning_style = learning_style_id
                except Exception as session_error:
                    logger.error(f"Error updating session state: {session_error}")
                
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating learning style preference: {e}")
            return False
    
    def start_interaction(self, student_id: str, query: str, 
                         learning_style_id: Optional[str] = None,
                         module_id: Optional[str] = None,
                         lecture_code: Optional[str] = None) -> str:
        """
        Start tracking a new student interaction
        
        Args:
            student_id: ID of the student
            query: The student's query
            learning_style_id: Learning style for this interaction (optional)
            module_id: Module ID related to this interaction (optional)
            lecture_code: Lecture code related to this interaction (optional)
            
        Returns:
            Interaction ID for the new interaction
        """
        # Get student profile
        profile = self.get_student_profile(student_id)
        if not profile:
            logger.error(f"Student profile not found for {student_id}")
            return None
        
        # Use default learning style if none provided
        if not learning_style_id:
            learning_style_id = profile.get("default_learning_style", "detailed")
        
        # Generate interaction ID
        interaction_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        
        # Create interaction document
        interaction = {
            "_id": interaction_id,
            "student_id": student_id,
            "query": query,
            "learning_style_id": learning_style_id,
            "module_id": module_id,
            "lecture_code": lecture_code,
            "start_time": now,
            "end_time": None,
            "time_spent": None,
            "feedback": None,
            "helpful": None,
            "follow_up_queries": [],
            "retrieved_sources": [],
            "timestamp": now
        }
        
        try:
            # Store interaction
            self.db[self.config.interactions_collection].insert_one(interaction)
            
            # Update student profile
            self.db[self.config.profiles_collection].update_one(
                {"_id": student_id},
                {
                    "$set": {"last_active": now},
                    "$inc": {"total_interactions": 1},
                    "$addToSet": {"modules_accessed": module_id} if module_id else {}
                }
            )
            
            # Update session state if using Streamlit
            if 'streamlit' in sys.modules and st is not None:
                try:
                    if hasattr(st, 'session_state'):
                        st.session_state.interaction_start_time = now
                        st.session_state.current_interaction_id = interaction_id
                        st.session_state.last_activity_time = now
                except Exception as session_error:
                    logger.error(f"Error updating session state: {session_error}")
            
            logger.info(f"Started interaction {interaction_id} for student {student_id}")
            return interaction_id
        except Exception as e:
            logger.error(f"Error starting interaction: {e}")
            return None
    
    def end_interaction(self, interaction_id: str, retrieved_sources: List[Dict] = None,
                       feedback: str = None, helpful: bool = None) -> bool:
        """
        End a student interaction and record metrics
        
        Args:
            interaction_id: ID of the interaction to end
            retrieved_sources: List of sources retrieved for this query (optional)
            feedback: Student feedback text (optional)
            helpful: Whether the student found the response helpful (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get interaction
            interaction = self.db[self.config.interactions_collection].find_one({"_id": interaction_id})
            if not interaction:
                logger.error(f"Interaction {interaction_id} not found")
                return False
            
            now = datetime.datetime.now()
            start_time = interaction.get("start_time")
            
            # Calculate time spent
            if start_time:
                time_spent = (now - start_time).total_seconds()
                # Cap time spent at threshold to avoid skewing analytics
                time_spent = min(time_spent, self.config.time_spent_threshold)
            else:
                time_spent = 0
            
            # Update interaction
            update_data = {
                "$set": {
                    "end_time": now,
                    "time_spent": time_spent
                }
            }
            
            if retrieved_sources:
                update_data["$set"]["retrieved_sources"] = retrieved_sources
            
            if feedback is not None:
                update_data["$set"]["feedback"] = feedback
                
            if helpful is not None:
                update_data["$set"]["helpful"] = helpful
            
            # Apply updates
            result = self.db[self.config.interactions_collection].update_one(
                {"_id": interaction_id},
                update_data
            )
            
            # Clear session state if using Streamlit
            if 'streamlit' in sys.modules and st is not None:
                try:
                    if hasattr(st, 'session_state'):
                        st.session_state.interaction_start_time = None
                        st.session_state.current_interaction_id = None
                        st.session_state.last_activity_time = now
                except Exception as session_error:
                    logger.error(f"Error updating session state: {session_error}")
            
            # Trigger analytics update
            student_id = interaction.get("student_id")
            if student_id:
                self._update_student_analytics(student_id)
            
            logger.info(f"Ended interaction {interaction_id}")
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error ending interaction: {e}")
            return False
    
    def add_follow_up_query(self, interaction_id: str, follow_up_query: str) -> bool:
        """
        Add a follow-up query to an existing interaction
        
        Args:
            interaction_id: ID of the parent interaction
            follow_up_query: The follow-up query text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.db[self.config.interactions_collection].update_one(
                {"_id": interaction_id},
                {
                    "$push": {
                        "follow_up_queries": {
                            "query": follow_up_query,
                            "timestamp": datetime.datetime.now()
                        }
                    }
                }
            )
            
            # Update last activity time if using Streamlit
            if 'streamlit' in sys.modules and st is not None:
                try:
                    if hasattr(st, 'session_state'):
                        st.session_state.last_activity_time = datetime.datetime.now()
                except Exception as session_error:
                    logger.error(f"Error updating session state: {session_error}")
                
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error adding follow-up query: {e}")
            return False
    
    def check_session_timeout(self) -> bool:
        """
        Check if the current session has timed out
        
        Returns:
            True if session has timed out, False otherwise
        """
        # First check if we're in a Streamlit context
        if 'streamlit' not in sys.modules or st is None:
            return False
            
        # Then check if session_state exists and is properly initialized
        if not hasattr(st, 'session_state'):
            return False
            
        # Check if last_activity_time is set
        if not hasattr(st.session_state, 'last_activity_time') or not st.session_state.last_activity_time:
            return False
            
        try:
            # Get last activity time
            last_activity = st.session_state.last_activity_time
            now = datetime.datetime.now()
            time_diff = (now - last_activity).total_seconds()
            
            # If session has timed out, end current interaction
            if time_diff > self.config.session_timeout:
                if hasattr(st.session_state, 'current_interaction_id') and st.session_state.current_interaction_id:
                    self.end_interaction(st.session_state.current_interaction_id)
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking session timeout: {e}")
            return False
    
    def update_last_activity(self):
        """Update the last activity timestamp"""
        # Check if we're in a Streamlit context
        if 'streamlit' not in sys.modules or st is None:
            return
            
        # Check if session_state exists
        if not hasattr(st, 'session_state'):
            return
            
        # Update the last activity time
        try:
            st.session_state.last_activity_time = datetime.datetime.now()
        except Exception as e:
            logger.error(f"Error updating last activity: {e}")
    
    def _update_student_analytics(self, student_id: str) -> bool:
        """
        Update analytics for a student
        
        Args:
            student_id: ID of the student
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all interactions for this student
            interactions = list(self.db[self.config.interactions_collection].find(
                {"student_id": student_id}
            ))
            
            if not interactions:
                logger.info(f"No interactions found for student {student_id}")
                return False
            
            # Calculate analytics
            now = datetime.datetime.now()
            
            # Group interactions by module
            module_interactions = {}
            for interaction in interactions:
                module_id = interaction.get("module_id")
                if not module_id:
                    continue
                    
                if module_id not in module_interactions:
                    module_interactions[module_id] = []
                    
                module_interactions[module_id].append(interaction)
            
            # Calculate overall analytics
            total_interactions = len(interactions)
            completed_interactions = sum(1 for i in interactions if i.get("end_time") is not None)
            total_time_spent = sum(i.get("time_spent", 0) for i in interactions if i.get("time_spent") is not None)
            avg_time_per_query = total_time_spent / max(completed_interactions, 1)
            
            # Get helpful ratings
            helpful_ratings = [i.get("helpful") for i in interactions if i.get("helpful") is not None]
            satisfaction_rate = sum(1 for r in helpful_ratings if r) / max(len(helpful_ratings), 1) if helpful_ratings else None
            
            # Learning style preferences
            style_counts = {}
            for interaction in interactions:
                style = interaction.get("learning_style_id")
                if style:
                    style_counts[style] = style_counts.get(style, 0) + 1
                    
            preferred_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Module-specific analytics
            module_analytics = []
            for module_id, mod_interactions in module_interactions.items():
                mod_total = len(mod_interactions)
                mod_completed = sum(1 for i in mod_interactions if i.get("end_time") is not None)
                mod_time_spent = sum(i.get("time_spent", 0) for i in mod_interactions if i.get("time_spent") is not None)
                mod_avg_time = mod_time_spent / max(mod_completed, 1)
                
                # Helpful ratings for this module
                mod_helpful = [i.get("helpful") for i in mod_interactions if i.get("helpful") is not None]
                mod_satisfaction = sum(1 for r in mod_helpful if r) / max(len(mod_helpful), 1) if mod_helpful else None
                
                # Learning style preferences for this module
                mod_style_counts = {}
                for interaction in mod_interactions:
                    style = interaction.get("learning_style_id")
                    if style:
                        mod_style_counts[style] = mod_style_counts.get(style, 0) + 1
                        
                mod_preferred_styles = sorted(mod_style_counts.items(), key=lambda x: x[1], reverse=True)
                
                module_analytics.append({
                    "module_id": module_id,
                    "total_interactions": mod_total,
                    "completed_interactions": mod_completed,
                    "total_time_spent": mod_time_spent,
                    "avg_time_per_query": mod_avg_time,
                    "satisfaction_rate": mod_satisfaction,
                    "preferred_styles": mod_preferred_styles[:2] if mod_preferred_styles else []
                })
            
            # Create or update analytics document
            analytics_doc = {
                "student_id": student_id,
                "last_updated": now,
                "total_interactions": total_interactions,
                "completed_interactions": completed_interactions,
                "total_time_spent": total_time_spent,
                "avg_time_per_query": avg_time_per_query,
                "satisfaction_rate": satisfaction_rate,
                "preferred_styles": preferred_styles[:3] if preferred_styles else [],
                "module_analytics": module_analytics,
                "interaction_history": [
                    {
                        "date": i.get("start_time"),
                        "module_id": i.get("module_id"),
                        "lecture_code": i.get("lecture_code"),
                        "time_spent": i.get("time_spent"),
                        "helpful": i.get("helpful")
                    }
                    for i in interactions if i.get("start_time") is not None
                ]
            }
            
            # Upsert the analytics document
            self.db[self.config.analytics_collection].update_one(
                {"student_id": student_id},
                {"$set": analytics_doc},
                upsert=True
            )
            
            logger.info(f"Updated analytics for student {student_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating student analytics: {e}")
            return False
    
    def get_student_analytics(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analytics for a student
        
        Args:
            student_id: ID of the student
            
        Returns:
            Analytics document or None if not found
        """
        try:
            analytics = self.db[self.config.analytics_collection].find_one({"student_id": student_id})
            
            # If analytics don't exist or are outdated, update them
            if not analytics or (datetime.datetime.now() - analytics.get("last_updated", datetime.datetime.min)).total_seconds() > self.config.analytics_update_frequency * 3600:
                self._update_student_analytics(student_id)
                analytics = self.db[self.config.analytics_collection].find_one({"student_id": student_id})
                
            return analytics
        except Exception as e:
            logger.error(f"Error retrieving student analytics: {e}")
            return None
    
    def get_module_analytics(self, student_id: str, module_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analytics for a specific module for a student
        
        Args:
            student_id: ID of the student
            module_id: ID of the module
            
        Returns:
            Module analytics or None if not found
        """
        try:
            analytics = self.get_student_analytics(student_id)
            if not analytics:
                return None
                
            module_analytics = next(
                (m for m in analytics.get("module_analytics", []) if m.get("module_id") == module_id),
                None
            )
            
            return module_analytics
        except Exception as e:
            logger.error(f"Error retrieving module analytics: {e}")
            return None
    
    def get_learning_path_recommendations(self, student_id: str, module_id: str) -> List[Dict[str, Any]]:
        """
        Generate personalized learning path recommendations based on analytics
        
        Args:
            student_id: ID of the student
            module_id: Current module ID
            
        Returns:
            List of recommended learning resources
        """
        try:
            # Get student analytics
            analytics = self.get_student_analytics(student_id)
            if not analytics:
                return []
                
            # Get module analytics
            module_analytics = self.get_module_analytics(student_id, module_id)
            
            # Create recommendations based on analytics
            recommendations = []
            
            # Check if module analytics exist
            if not module_analytics:
                return []
                
            # Simple recommendation based on time spent
            avg_time = module_analytics.get("avg_time_per_query", 0)
            
            # More detailed resources for those who spend less time on queries
            if avg_time < 20:  # Less than 20 seconds per query
                recommendations.append({
                    "type": "resource",
                    "title": "Detailed Study Guide",
                    "description": "This comprehensive guide covers all the fundamental concepts",
                    "reason": "You seem to be moving quickly through material. Consider spending more time on key concepts."
                })
            
            # If satisfaction rate is low, recommend different learning style
            satisfaction_rate = module_analytics.get("satisfaction_rate", None)
            if satisfaction_rate is not None and satisfaction_rate < 0.7:
                # Get current preferred style
                current_styles = module_analytics.get("preferred_styles", [])
                current_style_id = current_styles[0][0] if current_styles else "detailed"
                
                # Recommend a different style
                for style in self.config.learning_styles:
                    if style.id != current_style_id:
                        recommendations.append({
                            "type": "learning_style",
                            "style_id": style.id,
                            "title": f"Try {style.name} Learning Style",
                            "description": style.description,
                            "reason": "This learning style might help you understand concepts better."
                        })
                        break
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating learning path recommendations: {e}")
            return []
    
    def format_query_with_learning_style(self, query: str, learning_style_id: str) -> str:
        """
        Format a query with the appropriate learning style prompt template
        
        Args:
            query: The original query
            learning_style_id: ID of the learning style to use
            
        Returns:
            Formatted query with learning style instructions
        """
        learning_style = self.get_learning_style_by_id(learning_style_id)
        if not learning_style:
            # Default to detailed style if not found
            learning_style = self.get_learning_style_by_id("detailed")
            if not learning_style:
                return query
        
        # Format using the prompt template
        formatted_query = f"{query}\n\n[Learning Style: {learning_style.prompt_template}]"
        return formatted_query
    

    def generate_student_report(self, student_id: str) -> Dict[str, Any]:
        """Generate a comprehensive report of student learning activity"""
        try:
            analytics = self.get_student_analytics(student_id)
            if not analytics:
                return {"error": "No analytics found for this student"}
            
            profile = self.get_student_profile(student_id)
            if not profile:
                return {"error": "Student profile not found"}
            
            # Basic student info with safe conversions
            total_time = analytics.get("total_time_spent", 0)
            satisfaction = analytics.get("satisfaction_rate")
            
            report = {
                "student_name": profile.get("name", "Unknown"),
                "student_id": student_id,
                "total_study_time": total_time / 60 if total_time is not None else 0,
                "total_interactions": analytics.get("total_interactions", 0),
                "satisfaction_rate": satisfaction * 100 if satisfaction is not None else None,
                "preferred_learning_style": None,
                "modules_activity": [],
                "learning_style_usage": [],
                "daily_activity": [],
                "areas_for_improvement": [],
                "strengths": []
            }
            
            # Add preferred learning style if available
            preferred_styles = analytics.get("preferred_styles", [])
            if preferred_styles:
                style_id = preferred_styles[0][0]
                style = self.get_learning_style_by_id(style_id)
                if style:
                    report["preferred_learning_style"] = {
                        "id": style.id,
                        "name": style.name,
                        "description": style.description
                    }
            
            # Module activity data with safe conversions
            for module in analytics.get("module_analytics", []):
                mod_time = module.get("total_time_spent", 0)
                mod_satisfaction = module.get("satisfaction_rate")
                
                report["modules_activity"].append({
                    "module_id": module.get("module_id", "Unknown"),
                    "time_spent": mod_time / 60 if mod_time is not None else 0,
                    "interactions": module.get("total_interactions", 0),
                    "avg_time_per_query": module.get("avg_time_per_query", 0),
                    "satisfaction_rate": mod_satisfaction * 100 if mod_satisfaction is not None else None
                })
            
            # Learning style usage data
            style_usage = {}
            interaction_history = analytics.get("interaction_history", [])
            
            for interaction in interaction_history:
                style_id = interaction.get("learning_style_id", "unknown")
                style_usage[style_id] = style_usage.get(style_id, 0) + 1
            
            for style_id, count in style_usage.items():
                style = self.get_learning_style_by_id(style_id)
                style_name = style.name if style else "Unknown"
                report["learning_style_usage"].append({
                    "style_id": style_id,
                    "style_name": style_name,
                    "count": count
                })
            
            # Daily activity data with safe conversions
            daily_activity = {}
            for interaction in interaction_history:
                date = interaction.get("date")
                if not date:
                    continue
                
                date_str = date.strftime("%Y-%m-%d")
                if date_str not in daily_activity:
                    daily_activity[date_str] = {
                        "date": date_str,
                        "count": 0,
                        "time_spent": 0
                    }
                
                daily_activity[date_str]["count"] += 1
                
                int_time = interaction.get("time_spent")
                if int_time is not None:
                    daily_activity[date_str]["time_spent"] += int_time / 60
            
            report["daily_activity"] = list(daily_activity.values())
            
            # Identify areas for improvement
            module_times = [(m["module_id"], m["time_spent"]) 
                            for m in report["modules_activity"] if m["time_spent"] is not None]
            if module_times:
                module_times.sort(key=lambda x: x[1])
                for module_id, time_spent in module_times[:2]:
                    if time_spent < 30:
                        report["areas_for_improvement"].append({
                            "module_id": module_id,
                            "metric": "time_spent",
                            "value": time_spent,
                            "suggestion": f"Consider spending more time studying {module_id}."
                        })
            
            # Identify strengths
            module_satisfactions = [(m["module_id"], m["satisfaction_rate"]) 
                                    for m in report["modules_activity"] 
                                    if m["satisfaction_rate"] is not None]
            if module_satisfactions:
                module_satisfactions.sort(key=lambda x: x[1], reverse=True)
                for module_id, satisfaction in module_satisfactions[:2]:
                    if satisfaction > 80:
                        report["strengths"].append({
                            "module_id": module_id,
                            "metric": "satisfaction_rate",
                            "value": satisfaction,
                            "comment": f"You're performing well in {module_id} with high satisfaction."
                        })
            
            return report
        except Exception as e:
            logger.error(f"Error generating student report: {e}")
            return {"error": f"Error generating report: {str(e)}"}
