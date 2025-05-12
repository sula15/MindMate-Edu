import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import uuid
from typing import Dict, List, Any, Optional

# Import the personalization module
from student_personalization import StudentPersonalization, PersonalizationConfig, LearningStyle

def initialize_personalization_module():
    """Initialize the personalization module with default configuration"""
    config = PersonalizationConfig()
    return StudentPersonalization(config)

def render_login_ui():
    """Render the student login UI"""
    st.subheader("Student Login")
    
    # Initialize session state if needed
    if 'student_id' not in st.session_state:
        st.session_state.student_id = None
    if 'student_name' not in st.session_state:
        st.session_state.student_name = None
    
    # If already logged in, show logout option
    if st.session_state.student_id:
        st.write(f"Logged in as: **{st.session_state.student_name}** (ID: {st.session_state.student_id})")
        
        if st.button("Logout"):
            # Clear session state
            st.session_state.student_id = None
            st.session_state.student_name = None
            st.session_state.learning_style = None
            st.rerun()
    else:
        # Login form
        with st.form("login_form"):
            col1, col2 = st.columns(2)
            with col1:
                student_id = st.text_input("Student ID")
            with col2:
                student_name = st.text_input("Name")
                
            email = st.text_input("Email (optional)")
            
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if not student_id or not student_name:
                    st.error("Please enter both Student ID and Name.")
                else:
                    # Initialize personalization module
                    personalization = initialize_personalization_module()
                    
                    # Create or get student profile
                    profile = personalization.create_student_profile(
                        student_id=student_id,
                        name=student_name,
                        email=email
                    )
                    
                    if profile:
                        # Set session state
                        st.session_state.student_id = student_id
                        st.session_state.student_name = student_name
                        st.session_state.learning_style = profile.get("default_learning_style")
                        
                        st.success(f"Welcome, {student_name}!")
                        st.rerun()
                    else:
                        st.error("Failed to create or retrieve student profile.")

def render_learning_style_selector():
    """Render the learning style selector"""
    if not st.session_state.get("student_id"):
        return
    
    # Initialize personalization module
    personalization = initialize_personalization_module()
    
    # Get all learning styles
    styles = personalization.get_learning_styles()
    
    # Get current learning style
    current_style_id = st.session_state.get("learning_style", "detailed")
    
    st.subheader("Select Your Learning Style")
    
    # Create columns for the learning styles
    cols = st.columns(3)
    
    # Display each learning style as a selectable card
    for i, style in enumerate(styles):
        with cols[i % 3]:
            # Determine if this style is selected
            is_selected = style["id"] == current_style_id
            
            # Create card with background color if selected
            with st.container(border=True):
                st.markdown(f"### {style['name']}")
                st.markdown(style["description"])
                
                # Button to select this style
                if st.button("Select", key=f"style_{style['id']}", 
                            disabled=is_selected,
                            use_container_width=True):
                    # Update student profile
                    success = personalization.update_learning_style_preference(
                        student_id=st.session_state.student_id,
                        learning_style_id=style["id"]
                    )
                    
                    if success:
                        st.session_state.learning_style = style["id"]
                        st.success(f"Learning style updated to {style['name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to update learning style.")
                
                # Show "Current" indicator
                if is_selected:
                    st.success("Current Style")

def format_student_query(query, student_id=None, module_id=None):
    """
    Format a student query with learning style preferences
    
    Args:
        query: The original query
        student_id: Student ID (optional)
        module_id: Module ID (optional)
        
    Returns:
        Formatted query and interaction ID
    """
    if not student_id or not st.session_state.get("student_id"):
        return query, None
    
    try:
        # Initialize personalization module
        personalization = initialize_personalization_module()
        
        # Get current learning style
        learning_style_id = st.session_state.get("learning_style", "detailed")
        
        # Start tracking interaction
        interaction_id = personalization.start_interaction(
            student_id=student_id,
            query=query,
            learning_style_id=learning_style_id,
            module_id=module_id
        )
        
        # Format query with learning style
        formatted_query = personalization.format_query_with_learning_style(
            query=query,
            learning_style_id=learning_style_id
        )
        
        return formatted_query, interaction_id
    except Exception as e:
        st.error(f"Error formatting query: {e}")
        return query, None

def end_student_interaction(interaction_id, retrieved_sources=None, feedback=None, helpful=None):
    """
    End a student interaction and record metrics
    
    Args:
        interaction_id: The interaction ID
        retrieved_sources: List of sources retrieved (optional)
        feedback: Student feedback (optional)
        helpful: Whether the response was helpful (optional)
        
    Returns:
        True if successful, False otherwise
    """
    if not interaction_id:
        return False
    
    try:
        # Initialize personalization module
        personalization = initialize_personalization_module()
        
        # End interaction
        success = personalization.end_interaction(
            interaction_id=interaction_id,
            retrieved_sources=retrieved_sources,
            feedback=feedback,
            helpful=helpful
        )
        
        return success
    except Exception as e:
        st.error(f"Error ending interaction: {e}")
        return False

def render_feedback_ui(interaction_id):
    """
    Render a feedback UI for the student
    
    Args:
        interaction_id: The interaction ID
    """
    if not interaction_id:
        return
    
    with st.expander("**Provide Feedback**", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Add a unique key based on the interaction_id
            feedback = st.text_area(
                "How can we improve our answers?", 
                key=f"feedback_{interaction_id}"  # This makes the key unique
            )
        
        with col2:
            # Also add a unique key for the radio button
            helpful = st.radio(
                "Was this helpful?", 
                ["Yes", "No"], 
                index=0,
                key=f"helpful_{interaction_id}"  # This makes the key unique
            )
        
        # And for the button
        if st.button("Submit Feedback", key=f"submit_{interaction_id}"):
            # Convert helpful to boolean
            helpful_bool = helpful == "Yes"
            
            # Submit feedback
            success = end_student_interaction(
                interaction_id=interaction_id,
                feedback=feedback,
                helpful=helpful_bool
            )
            
            if success:
                st.success("Thank you for your feedback!")
            else:
                st.error("Failed to submit feedback. Please try again.")

def render_analytics_dashboard():
    """Render a student analytics dashboard"""
    if not st.session_state.get("student_id"):
        st.warning("Please log in to view your analytics.")
        return
    
    st.header("Your Learning Analytics")
    
    try:
        # Initialize personalization module
        personalization = initialize_personalization_module()
        
        # Get student analytics
        analytics = personalization.get_student_analytics(st.session_state.student_id)
        
        if not analytics:
            st.info("No analytics data available yet. Start learning to generate insights!")
            return
        
        # Generate analytics report
        report = personalization.generate_student_report(st.session_state.student_id)
        
        if "error" in report:
            st.error(report["error"])
            return
        
        # Display overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Study Time", f"{report['total_study_time']:.1f} min")
        
        with col2:
            st.metric("Total Interactions", f"{report['total_interactions']}")
        
        with col3:
            if report['satisfaction_rate'] is not None:
                st.metric("Satisfaction Rate", f"{report['satisfaction_rate']:.1f}%")
            else:
                st.metric("Satisfaction Rate", "N/A")
        
        with col4:
            if report['preferred_learning_style']:
                st.metric("Learning Style", report['preferred_learning_style']['name'])
            else:
                st.metric("Learning Style", "Not set")
        
        # Show areas for improvement
        if report['areas_for_improvement']:
            st.subheader("Areas for Improvement")
            for area in report['areas_for_improvement']:
                st.info(area['suggestion'])
        
        # Show strengths
        if report['strengths']:
            st.subheader("Your Strengths")
            for strength in report['strengths']:
                st.success(strength['comment'])
        
        # Create tabs for different analytics
        tab1, tab2, tab3 = st.tabs(["Module Activity", "Learning Styles", "Daily Activity"])
        
        with tab1:
            # Module activity visualization
            if report['modules_activity']:
                # Convert to DataFrame
                df_modules = pd.DataFrame(report['modules_activity'])
                
                # Create bar chart for time spent by module
                fig = px.bar(
                    df_modules, 
                    x='module_id', 
                    y='time_spent',
                    labels={'module_id': 'Module', 'time_spent': 'Time Spent (minutes)'},
                    title='Time Spent by Module',
                    color='satisfaction_rate',
                    color_continuous_scale='RdYlGn',
                    hover_data=['interactions', 'avg_time_per_query', 'satisfaction_rate']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed module data
                st.dataframe(df_modules)
            else:
                st.info("No module activity data available yet.")
        
        with tab2:
            # Learning style usage visualization
            if report['learning_style_usage']:
                # Convert to DataFrame
                df_styles = pd.DataFrame(report['learning_style_usage'])
                
                # Create pie chart for learning style usage
                fig = px.pie(
                    df_styles,
                    values='count',
                    names='style_name',
                    title='Learning Style Usage'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No learning style usage data available yet.")
        
        with tab3:
            # Daily activity visualization
            if report['daily_activity']:
                # Convert to DataFrame
                df_daily = pd.DataFrame(report['daily_activity'])
                
                # Sort by date
                df_daily['date'] = pd.to_datetime(df_daily['date'])
                df_daily = df_daily.sort_values('date')
                
                # Create line chart for daily activity
                fig = px.line(
                    df_daily,
                    x='date',
                    y='time_spent',
                    markers=True,
                    labels={'date': 'Date', 'time_spent': 'Time Spent (minutes)'},
                    title='Daily Study Activity'
                )
                
                # Add interaction count as bar chart
                fig.add_trace(
                    go.Bar(
                        x=df_daily['date'],
                        y=df_daily['count'],
                        name='Interactions',
                        yaxis='y2',
                        opacity=0.5
                    )
                )
                
                # Update layout for dual y-axes
                fig.update_layout(
                    yaxis2=dict(
                        title='Interactions',
                        overlaying='y',
                        side='right'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily activity data available yet.")
    
    except Exception as e:
        st.error(f"Error rendering analytics dashboard: {e}")

def render_learning_path_recommendations(module_id=None):
    """
    Render learning path recommendations for a student
    
    Args:
        module_id: Current module ID (optional)
    """
    if not st.session_state.get("student_id"):
        return
    
    if not module_id:
        return
    
    try:
        # Initialize personalization module
        personalization = initialize_personalization_module()
        
        # Get recommendations
        recommendations = personalization.get_learning_path_recommendations(
            student_id=st.session_state.student_id,
            module_id=module_id
        )
        
        if not recommendations:
            return
        
        st.subheader("Personalized Recommendations")
        
        for i, rec in enumerate(recommendations):
            with st.container(border=True):
                st.markdown(f"#### {rec['title']}")
                st.markdown(rec['description'])
                st.markdown(f"*{rec['reason']}*")
                
                # For learning style recommendations, add a button to switch
                if rec['type'] == 'learning_style':
                    if st.button(f"Switch to this Style", key=f"rec_{i}"):
                        success = personalization.update_learning_style_preference(
                            student_id=st.session_state.student_id,
                            learning_style_id=rec['style_id']
                        )
                        
                        if success:
                            st.session_state.learning_style = rec['style_id']
                            st.success(f"Learning style updated to {rec['title']}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to update learning style.")
    
    except Exception as e:
        st.error(f"Error rendering recommendations: {e}")

# Function to integrate with the main app
def add_personalization_to_sidebar():
    """Add personalization components to the sidebar"""
    st.sidebar.title("Student Personalization")
    
    # Login UI
    render_login_ui()
    
    # Learning style selector (if logged in)
    if st.session_state.get("student_id"):
        st.sidebar.divider()
        render_learning_style_selector()
        
        # Link to analytics
        st.sidebar.divider()
        if st.sidebar.button("View My Learning Analytics", use_container_width=True):
            st.session_state.show_analytics = True
        
        # Check session timeout
        try:
            personalization = initialize_personalization_module()
            personalization.check_session_timeout()
            personalization.update_last_activity()
        except Exception as e:
            st.sidebar.error(f"Session error: {e}")