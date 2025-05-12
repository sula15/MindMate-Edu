# database.py
import sqlite3
from datetime import datetime

# Connect to SQLite database (creates if it doesn't exist)
conn = sqlite3.connect("learning_analytics.db")
cursor = conn.cursor()

# Create tables for behavioral analytics
cursor.execute("""
CREATE TABLE IF NOT EXISTS Students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    full_name TEXT,
    email TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Queries (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    query_text TEXT NOT NULL,
    retrieval_style TEXT CHECK(retrieval_style IN ('detailed', 'short', 'bulleted', 'visual')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER,
    result_id INTEGER,
    clicked BOOLEAN DEFAULT 0,
    dwell_time INTEGER,
    feedback TEXT CHECK(feedback IN ('thumbs_up', 'thumbs_down', NULL)),
    FOREIGN KEY (query_id) REFERENCES Queries(query_id)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS QueryTrends (
    trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT UNIQUE,
    frequency INTEGER DEFAULT 1
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS LearningPatterns (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER UNIQUE,
    preferred_style TEXT,
    avg_query_length REAL,
    total_interactions INTEGER,
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);
""")

conn.commit()

# ==================== Behavioral Analytics Functions ====================
def add_student(username, full_name, email):
    """Add a new student to the database."""
    cursor.execute("""
    INSERT INTO Students (username, full_name, email)
    VALUES (?, ?, ?)
    """, (username, full_name, email))
    conn.commit()
    return cursor.lastrowid

def add_query(student_id, query_text, retrieval_style):
    """Add a new query to the database."""
    cursor.execute("""
    INSERT INTO Queries (student_id, query_text, retrieval_style)
    VALUES (?, ?, ?)
    """, (student_id, query_text, retrieval_style))
    conn.commit()
    return cursor.lastrowid

def add_interaction(query_id, result_id, clicked, dwell_time, feedback=None):
    """Add an interaction to the database."""
    cursor.execute("""
    INSERT INTO Interactions (query_id, result_id, clicked, dwell_time, feedback)
    VALUES (?, ?, ?, ?, ?)
    """, (query_id, result_id, clicked, dwell_time, feedback))
    conn.commit()

def add_feedback(query_id, result_id, feedback):
    """Add feedback for a specific interaction."""
    cursor.execute("""
    UPDATE Interactions
    SET feedback = ?
    WHERE query_id = ? AND result_id = ?
    """, (feedback, query_id, result_id))
    conn.commit()

def update_query_trends(query_text):
    """Update query trends in the database."""
    cursor.execute("""
    INSERT INTO QueryTrends (query_text) VALUES (?)
    ON CONFLICT(query_text) DO UPDATE SET frequency = frequency + 1
    """, (query_text,))
    conn.commit()

def update_learning_patterns(student_id):
    """Update learning patterns for a student."""
    # Calculate average query length and total interactions
    cursor.execute("""
    SELECT AVG(LENGTH(query_text)), COUNT(*)
    FROM Queries
    WHERE student_id = ?
    """, (student_id,))
    avg_query_length, total_interactions = cursor.fetchone()

    # Get preferred retrieval style
    cursor.execute("""
    SELECT retrieval_style, COUNT(*) as style_count
    FROM Queries
    WHERE student_id = ?
    GROUP BY retrieval_style
    ORDER BY style_count DESC
    LIMIT 1
    """, (student_id,))
    preferred_style = cursor.fetchone()[0]

    # Insert or update learning patterns
    cursor.execute("""
    INSERT INTO LearningPatterns (student_id, preferred_style, avg_query_length, total_interactions)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(student_id) DO UPDATE SET
        preferred_style = excluded.preferred_style,
        avg_query_length = excluded.avg_query_length,
        total_interactions = excluded.total_interactions
    """, (student_id, preferred_style, avg_query_length, total_interactions))
    conn.commit()

def get_latest_query_id():
    """Get the latest query_id for the current student."""
    cursor.execute("SELECT query_id FROM Queries ORDER BY query_id DESC LIMIT 1")
    return cursor.fetchone()[0]

def close_connection():
    """Close the database connection."""
    conn.close()