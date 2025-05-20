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
    current_style_id = st.session_state.get("learning_style")

    # If learning style is not set in session, get from profile
    if not current_style_id:
        student_id = st.session_state.get("student_id")
        if student_id:
            profile = personalization.get_student_profile(student_id)
            if profile and "default_learning_style" in profile:
                current_style_id = profile.get("default_learning_style")
                # Update session state
                st.session_state.learning_style = current_style_id

    # If still not set, default to "detailed"
    if not current_style_id:
        current_style_id = "detailed"
        # Update session state
        st.session_state.learning_style = current_style_id

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

        # Get current learning style from session state
        learning_style_id = st.session_state.get("learning_style")

        # Debug output (temporarily add this)
        st.sidebar.write(f"Debug - Using learning style: {learning_style_id}")

        # If learning_style_id is None or empty, get from profile
        if not learning_style_id:
            # Get the student profile
            profile = personalization.get_student_profile(student_id)
            if profile and "default_learning_style" in profile:
                learning_style_id = profile.get("default_learning_style")
                # Update session state
                st.session_state.learning_style = learning_style_id
                st.sidebar.write(f"Debug - Retrieved style from profile: {learning_style_id}")

        # If still not set, use "detailed" as default
        if not learning_style_id:
            learning_style_id = "detailed"
            # Update session state
            st.session_state.learning_style = learning_style_id
            st.sidebar.write("Debug - Using default 'detailed' style")

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
        import traceback
        st.sidebar.code(traceback.format_exc(), language="python")
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

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

def render_analytics_dashboard():
    """Render a student analytics dashboard with fixed visualizations"""
    if not st.session_state.get("student_id"):
        st.warning("Please log in to view your analytics.")
        return

    st.header("Your Learning Analytics")

    try:
        # Initialize personalization module
        from student_personalization import StudentPersonalization, PersonalizationConfig
        personalization = StudentPersonalization(PersonalizationConfig())

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

        # Create tabs for different analytics
        tab1, tab2, tab3 = st.tabs(["Module Activity", "Learning Styles", "Daily Activity"])

        with tab1:
            # Module activity visualization
            if report['modules_activity']:
                st.subheader("Time Spent by Module")

                # Add data debugging
                #st.write("Raw module data (for debugging):")
                #st.json(report['modules_activity'])

                # Make sure all fields exist
                clean_modules = []
                for mod in report['modules_activity']:
                    clean_mod = mod.copy()
                    # Ensure all required fields exist with defaults
                    if 'module_id' not in clean_mod or not clean_mod['module_id']:
                        clean_mod['module_id'] = 'Unknown'
                    if 'time_spent' not in clean_mod or clean_mod['time_spent'] is None:
                        clean_mod['time_spent'] = 0
                    if 'interactions' not in clean_mod or clean_mod['interactions'] is None:
                        clean_mod['interactions'] = 0
                    if 'avg_time_per_query' not in clean_mod or clean_mod['avg_time_per_query'] is None:
                        clean_mod['avg_time_per_query'] = 0
                    if 'satisfaction_rate' not in clean_mod or clean_mod['satisfaction_rate'] is None:
                        clean_mod['satisfaction_rate'] = 50  # Neutral default

                    clean_modules.append(clean_mod)

                # Convert to DataFrame
                df_modules = pd.DataFrame(clean_modules)

                # Display the DataFrame for debugging
                #st.write("Cleaned DataFrame:")
                #st.dataframe(df_modules)

                # Basic bar chart without color scale if satisfaction_rate is missing
                fig = px.bar(
                    df_modules,
                    x='module_id',
                    y='time_spent',
                    labels={'module_id': 'Module', 'time_spent': 'Time Spent (minutes)'},
                    title='Time Spent by Module'
                )

                # Try using color only if all values exist
                try:
                    if 'satisfaction_rate' in df_modules.columns and not df_modules['satisfaction_rate'].isnull().any():
                        fig = px.bar(
                            df_modules,
                            x='module_id',
                            y='time_spent',
                            labels={'module_id': 'Module', 'time_spent': 'Time Spent (minutes)'},
                            title='Time Spent by Module',
                            color='satisfaction_rate',
                            color_continuous_scale='RdYlGn'
                        )
                except Exception as e:
                    st.warning(f"Could not apply color scale: {e}")

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No module activity data available yet.")

        with tab2:
            # Learning style usage visualization
            if report['learning_style_usage']:
                st.subheader("Learning Style Usage")

                # Ensure all data is valid
                clean_styles = []
                for style in report['learning_style_usage']:
                    clean_style = style.copy()
                    if 'style_name' not in clean_style or not clean_style['style_name']:
                        clean_style['style_name'] = f"Style {clean_style.get('style_id', 'Unknown')}"
                    if 'count' not in clean_style or clean_style['count'] is None:
                        clean_style['count'] = 0

                    clean_styles.append(clean_style)

                # Convert to DataFrame
                df_styles = pd.DataFrame(clean_styles)

                # Calculate total and percentages explicitly
                total_count = df_styles['count'].sum()
                df_styles['percentage'] = df_styles['count'] / total_count * 100

                # Put detailed data in an expander to save space
                with st.expander("View detailed data", expanded=False):
                    st.write(f"Total learning style interactions: {total_count}")
                    st.dataframe(df_styles)

                # Use two columns for side-by-side charts
                col1, col2 = st.columns(2)

                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.cm as cm
                    import numpy as np

                    # Define a modern color palette
                    colors = plt.cm.Pastel1(np.linspace(0, 1, len(df_styles)))

                    # First column: Pie chart
                    with col1:
                        # Create a pie chart with matplotlib - more modern styling
                        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#f0f2f6')
                        wedges, texts, autotexts = ax.pie(
                            df_styles['count'],
                            labels=None,  # Remove labels from pie
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors,
                            shadow=False,
                            wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 2}  # Donut style with white borders
                        )

                        # Style the percentage text
                        for autotext in autotexts:
                            autotext.set_color('black')
                            autotext.set_fontsize(9)
                            autotext.set_weight('bold')

                        # Equal aspect ratio ensures that pie is drawn as a circle
                        ax.axis('equal')

                        # Add a clean title
                        plt.title('Learning Style Distribution', fontsize=12, pad=10)

                        # Add a clean legend
                        ax.legend(
                            df_styles['style_name'],
                            title="Learning Styles",
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1),
                            fontsize=8
                        )

                        plt.tight_layout()
                        st.pyplot(fig)

                    # Second column: Bar chart
                    with col2:
                        # Create a bar chart with matplotlib - more modern styling
                        fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#f0f2f6')

                        # Create horizontal bars for better readability with long labels
                        bars = ax2.barh(
                            df_styles['style_name'],
                            df_styles['count'],
                            color=colors,
                            edgecolor='white',
                            linewidth=1
                        )

                        # Add counts inside the bars
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax2.text(
                                width / 2,  # Center of bar
                                bar.get_y() + bar.get_height()/2,  # Vertical center of bar
                                f'{int(width)}',  # Count value
                                ha='center',
                                va='center',
                                color='black',
                                fontweight='bold',
                                fontsize=9
                            )

                        # Set background color
                        ax2.set_facecolor('#f0f2f6')

                        # Remove top and right spines for cleaner look
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['left'].set_color('#cccccc')
                        ax2.spines['bottom'].set_color('#cccccc')

                        # Add a grid for easier reading
                        ax2.grid(axis='x', linestyle='--', alpha=0.3)

                        # Clean up axis labels
                        plt.xlabel('Number of Interactions', fontsize=10)
                        plt.ylabel('')  # No y-label needed

                        # Clean title
                        plt.title('Frequency of Learning Styles', fontsize=12)

                        plt.tight_layout()
                        st.pyplot(fig2)

                except Exception as e:
                    st.error(f"Error creating charts: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
            else:
                st.info("No learning style usage data available yet.")

        with tab3:
            # Daily activity visualization
            if report['daily_activity']:
                st.subheader("Daily Study Activity")

                # Debug data
                #st.write("Raw daily activity data (for debugging):")
                #st.json(report['daily_activity'])

                # Clean and convert data
                clean_daily = []
                for day in report['daily_activity']:
                    clean_day = day.copy()
                    if 'date' not in clean_day or not clean_day['date']:
                        continue  # Skip entries without dates
                    if 'count' not in clean_day or clean_day['count'] is None:
                        clean_day['count'] = 0
                    if 'time_spent' not in clean_day or clean_day['time_spent'] is None:
                        clean_day['time_spent'] = 0

                    clean_daily.append(clean_day)

                if clean_daily:
                    # Convert to DataFrame
                    df_daily = pd.DataFrame(clean_daily)

                    # Sort by date
                    try:
                        df_daily['date'] = pd.to_datetime(df_daily['date'])
                        df_daily = df_daily.sort_values('date')
                    except Exception as e:
                        st.warning(f"Date sorting failed: {e}")

                    # Display the DataFrame for debugging
                    st.write("Cleaned DataFrame:")
                    st.dataframe(df_daily)

                    # Create basic line chart without dual axis if it's problematic
                    try:
                        fig = px.line(
                            df_daily,
                            x='date',
                            y='time_spent',
                            markers=True,
                            labels={'date': 'Date', 'time_spent': 'Time Spent (minutes)'},
                            title='Daily Study Activity'
                        )

                        # Try to add interaction count as bar chart if data is valid
                        if 'count' in df_daily.columns and not df_daily['count'].isnull().any():
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
                    except Exception as e:
                        st.error(f"Error creating chart: {e}")
                        # Fallback to simpler chart
                        st.info("Displaying fallback chart...")
                        try:
                            simple_fig = px.bar(df_daily, x='date', y='time_spent', title='Daily Activity')
                            st.plotly_chart(simple_fig, use_container_width=True)
                        except Exception as e2:
                            st.error(f"Even simple chart failed: {e2}")
                else:
                    st.warning("No valid daily activity data found.")
            else:
                st.info("No daily activity data available yet.")

    except Exception as e:
        st.error(f"Error rendering analytics dashboard: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")

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
    
    # Learning style selector is now in the Chat tab, not here
    
    # Link to analytics (always visible when logged in)
    if st.session_state.get("student_id"):
        st.sidebar.divider()
        
        # Show current learning style info
        current_style = st.session_state.get("learning_style")
        if current_style:
            style_name = "Unknown"
            # Get style name if available
            personalization = initialize_personalization_module()
            style = personalization.get_learning_style_by_id(current_style)
            if style:
                style_name = style.name
            st.sidebar.info(f"Active Style: {style_name}")
        
        # Analytics button
        if st.sidebar.button("View My Learning Analytics", use_container_width=True):
            st.session_state.show_analytics = True
        
        # Check session timeout
        try:
            personalization = initialize_personalization_module()
            personalization.check_session_timeout()
            personalization.update_last_activity()
        except Exception as e:
            st.sidebar.error(f"Session error: {e}")

def fix_unknown_learning_styles():
    """
    Utility function to fix unknown learning styles in the database.
    Call this from your main app once to update existing records.
    """
    st.write("### Learning Style Database Fix Utility")

    # Initialize personalization module
    personalization = initialize_personalization_module()

    # Only run if user confirms
    if st.button("Fix Unknown Learning Styles"):
        with st.spinner("Fixing unknown learning styles in database..."):
            try:
                # Get collections
                profiles_collection = personalization.db[personalization.config.profiles_collection]
                interactions_collection = personalization.db[personalization.config.interactions_collection]

                # 1. Find interactions with unknown learning style
                unknown_interactions = list(interactions_collection.find(
                    {"learning_style_id": {"$in": ["unknown", None, ""]}}
                ))

                count_fixed = 0

                # 2. Fix each interaction
                for interaction in unknown_interactions:
                    student_id = interaction.get("student_id")
                    if not student_id:
                        continue

                    # Get student profile to find preferred style
                    profile = profiles_collection.find_one({"_id": student_id})
                    if not profile:
                        continue

                    # Get default learning style from profile
                    default_style = profile.get("default_learning_style", "detailed")

                    # Update the interaction
                    interactions_collection.update_one(
                        {"_id": interaction["_id"]},
                        {"$set": {"learning_style_id": default_style}}
                    )

                    count_fixed += 1

                # 3. Update analytics
                if count_fixed > 0:
                    # Get list of affected students
                    student_ids = set(interaction.get("student_id") for interaction in unknown_interactions if interaction.get("student_id"))

                    # Regenerate analytics for each student
                    for student_id in student_ids:
                        personalization._update_student_analytics(student_id)

                st.success(f"Fixed {count_fixed} interactions with unknown learning style")

                # Show debug info
                if count_fixed > 0:
                    st.info("Please check your analytics again to see the updated data")

            except Exception as e:
                st.error(f"Error fixing learning styles: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")

import streamlit as st
import logging

def fix_learning_style_analytics():
    """
    Utility function to regenerate analytics and fix the learning style usage data
    """
    st.write("### Fix Learning Style Analytics")

    # Only run if there's a logged-in student
    if not st.session_state.get("student_id"):
        st.warning("Please log in to fix your analytics.")
        return

    student_id = st.session_state.get("student_id")

    if st.button("Regenerate My Analytics"):
        with st.spinner("Regenerating analytics..."):
            try:
                # Import modules
                from student_personalization import StudentPersonalization, PersonalizationConfig

                # Initialize personalization module
                personalization = StudentPersonalization(PersonalizationConfig())

                # Get interactions for this student
                interactions = list(personalization.db[personalization.config.interactions_collection].find(
                    {"student_id": student_id}
                ))

                # Display debug information about interactions
                st.write(f"Found {len(interactions)} interactions in the database.")

                # Show learning styles in interactions
                interaction_styles = {}
                for interaction in interactions:
                    style_id = interaction.get("learning_style_id")
                    if style_id:
                        interaction_styles[style_id] = interaction_styles.get(style_id, 0) + 1

                st.write("Learning styles found in interactions:")
                for style_id, count in interaction_styles.items():
                    style = personalization.get_learning_style_by_id(style_id)
                    style_name = style.name if style else style_id
                    st.write(f"- {style_name} ({style_id}): {count} interactions")

                # Force regenerate analytics
                success = personalization._update_student_analytics(student_id)

                if success:
                    st.success("Analytics regenerated successfully!")
                    st.info("Please check the Learning Analytics tab to see the updated data.")
                else:
                    st.error("Failed to regenerate analytics.")

            except Exception as e:
                st.error(f"Error regenerating analytics: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")
