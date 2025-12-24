import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main { padding: 20px; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; }
    .danger-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; }
    h1 { color: #667eea; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('student_performance.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT,
        email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Student performance history table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        hours_studied REAL,
        previous_score REAL,
        sleep_hours REAL,
        attendance REAL,
        extracurricular BOOLEAN,
        predicted_score REAL,
        actual_score REAL,
        accuracy REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )''')
    
    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Database functions
def register_user(username, password, full_name, email):
    try:
        conn = sqlite3.connect('student_performance.db')
        c = conn.cursor()
        hashed_pwd = hash_password(password)
        c.execute('''INSERT INTO users (username, password, full_name, email) 
                    VALUES (?, ?, ?, ?)''', (username, hashed_pwd, full_name, email))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    conn = sqlite3.connect('student_performance.db')
    c = conn.cursor()
    hashed_pwd = hash_password(password)
    c.execute('SELECT user_id FROM users WHERE username = ? AND password = ?', 
              (username, hashed_pwd))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_user_info(user_id):
    conn = sqlite3.connect('student_performance.db')
    c = conn.cursor()
    c.execute('SELECT username, full_name, email FROM users WHERE user_id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    return result

def save_prediction(user_id, hours_studied, previous_score, sleep_hours, attendance, 
                    extracurricular, predicted_score, actual_score=None):
    conn = sqlite3.connect('student_performance.db')
    c = conn.cursor()
    accuracy = None
    if actual_score is not None:
        accuracy = 100 - abs(predicted_score - actual_score)
    
    c.execute('''INSERT INTO predictions 
                (user_id, hours_studied, previous_score, sleep_hours, attendance, 
                 extracurricular, predicted_score, actual_score, accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_id, hours_studied, previous_score, sleep_hours, attendance, 
               extracurricular, predicted_score, actual_score, accuracy))
    conn.commit()
    conn.close()

def get_user_predictions(user_id):
    conn = sqlite3.connect('student_performance.db')
    df = pd.read_sql_query('SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC', 
                          conn, params=(user_id,))
    conn.close()
    return df

def get_all_predictions():
    conn = sqlite3.connect('student_performance.db')
    df = pd.read_sql_query('SELECT p.*, u.username FROM predictions p JOIN users u ON p.user_id = u.user_id ORDER BY p.created_at DESC', 
                          conn)
    conn.close()
    return df

# Train ML model
def train_model():
    data = {
        'hours_studied': [2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
        'previous_score': [50, 60, 70, 80, 90, 55, 65, 75, 85, 95],
        'sleep_hours': [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
        'attendance': [60, 70, 80, 90, 95, 65, 75, 85, 90, 95],
        'extracurricular': [0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        'performance_score': [55, 65, 75, 85, 95, 60, 75, 85, 90, 98]
    }
    df = pd.DataFrame(data)
    X = df[['hours_studied', 'previous_score', 'sleep_hours', 'attendance', 'extracurricular']]
    y = df['performance_score']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

# Initialize database
init_db()

# Main app logic
if not st.session_state.logged_in:
    # Login/Register page
    st.title("📊 Student Performance Predictor")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 New User? Register")
        with st.form("register_form"):
            reg_username = st.text_input("Username", key="reg_user")
            reg_password = st.text_input("Password", type="password", key="reg_pwd")
            reg_full_name = st.text_input("Full Name", key="reg_name")
            reg_email = st.text_input("Email", key="reg_email")
            reg_submit = st.form_submit_button("Register ✨")
            
            if reg_submit:
                if not reg_username or not reg_password or not reg_full_name:
                    st.error("Please fill all fields!")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters!")
                elif register_user(reg_username, reg_password, reg_full_name, reg_email):
                    st.success("✅ Registration successful! Now login.")
                else:
                    st.error("❌ Username already exists!")
    
    with col2:
        st.markdown("### 🔐 Existing User? Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pwd")
            login_submit = st.form_submit_button("Login 🚀")
            
            if login_submit:
                user_id = verify_user(login_username, login_password)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = login_username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials!")

else:
    # Main application (after login)
    user_info = get_user_info(st.session_state.user_id)
    
    # Sidebar
    with st.sidebar:
        st.title(f"👤 {user_info[1]}")
        st.write(f"**Username:** {user_info[0]}")
        st.write(f"**Email:** {user_info[2]}")
        st.markdown("---")
        
        nav = st.radio("📌 Navigate", 
                       ["🏠 Dashboard", "🔮 Predict Score", "📈 My History", 
                        "📊 Analytics", "👥 Leaderboard", "⚙️ Settings"])
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
    
    # Dashboard
    if nav == "🏠 Dashboard":
        st.title("📊 Dashboard")
        
        user_preds = get_user_predictions(st.session_state.user_id)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(user_preds))
        
        with col2:
            if len(user_preds) > 0 and not user_preds['actual_score'].isna().all():
                avg_accuracy = user_preds['accuracy'].dropna().mean()
                st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
            else:
                st.metric("Avg Accuracy", "N/A")
        
        with col3:
            if len(user_preds) > 0:
                avg_pred_score = user_preds['predicted_score'].mean()
                st.metric("Avg Predicted Score", f"{avg_pred_score:.1f}")
            else:
                st.metric("Avg Predicted Score", "N/A")
        
        with col4:
            st.metric("Last Prediction", 
                     user_preds.iloc[0]['created_at'] if len(user_preds) > 0 else "None")
        
        st.markdown("---")
        
        # Recent predictions
        st.subheader("📋 Recent Predictions")
        if len(user_preds) > 0:
            st.dataframe(user_preds[['hours_studied', 'previous_score', 'sleep_hours', 
                                      'attendance', 'predicted_score', 'actual_score', 
                                      'accuracy', 'created_at']].head(5), use_container_width=True)
        else:
            st.info("No predictions yet. Go to 'Predict Score' to make your first prediction!")
    
    # Prediction page
    elif nav == "🔮 Predict Score":
        st.title("🔮 Predict Your Performance Score")
        
        model = train_model()
        
        col1, col2 = st.columns(2)
        
        with col1:
            hours_studied = st.slider("Hours Studied per Day", 0, 10, 5)
            sleep_hours = st.slider("Sleep Hours", 4, 12, 8)
            attendance = st.slider("Attendance %", 0, 100, 80)
        
        with col2:
            previous_score = st.slider("Previous Score", 0, 100, 70)
            extracurricular = st.checkbox("Participating in Extracurricular?")
        
        st.markdown("---")
        
        if st.button("🎯 Predict Score", use_container_width=True):
            X_input = np.array([[hours_studied, previous_score, sleep_hours, attendance, int(extracurricular)]])
            predicted = model.predict(X_input)[0]
            
            # Save to database
            save_prediction(st.session_state.user_id, hours_studied, previous_score, 
                          sleep_hours, attendance, extracurricular, predicted)
            
            st.success(f"✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Score", f"{predicted:.1f}/100")
            with col2:
                performance = "Excellent" if predicted >= 80 else "Good" if predicted >= 60 else "Average"
                st.metric("Performance Level", performance)
            with col3:
                recommendation = "Keep it up! 🚀" if predicted >= 80 else "Study harder! 💪"
                st.metric("Recommendation", recommendation)
    
    # History page
    elif nav == "📈 My History":
        st.title("📈 My Prediction History")
        
        user_preds = get_user_predictions(st.session_state.user_id)
        
        if len(user_preds) > 0:
            # Display table
            st.subheader("All Predictions")
            st.dataframe(user_preds[['hours_studied', 'previous_score', 'sleep_hours', 
                                      'attendance', 'predicted_score', 'actual_score', 
                                      'accuracy', 'created_at']], use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(user_preds, x='created_at', y='predicted_score', 
                             title='Predicted Scores Over Time', markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(user_preds, x='created_at', y='hours_studied',
                            title='Study Hours by Prediction', color='predicted_score')
                st.plotly_chart(fig, use_container_width=True)
            
            # Download data
            csv = user_preds.to_csv(index=False)
            st.download_button("⬇️ Download History as CSV", csv, "predictions.csv", "text/csv")
        else:
            st.info("No predictions yet!")
    
    # Analytics page
    elif nav == "📊 Analytics":
        st.title("📊 Analytics & Insights")
        
        user_preds = get_user_predictions(st.session_state.user_id)
        
        if len(user_preds) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Hours Studied", f"{user_preds['hours_studied'].mean():.1f} hrs")
                st.metric("Average Sleep", f"{user_preds['sleep_hours'].mean():.1f} hrs")
                st.metric("Average Attendance", f"{user_preds['attendance'].mean():.1f}%")
            
            with col2:
                st.metric("Average Predicted Score", f"{user_preds['predicted_score'].mean():.1f}")
                st.metric("Highest Score", f"{user_preds['predicted_score'].max():.1f}")
                st.metric("Lowest Score", f"{user_preds['predicted_score'].min():.1f}")
            
            st.markdown("---")
            
            # Correlation analysis
            st.subheader("📈 Correlation Analysis")
            fig = px.scatter(user_preds, x='hours_studied', y='predicted_score',
                            title='Hours Studied vs Predicted Score',
                            trendline='ols', size_max=10)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for analytics yet!")
    
    # Leaderboard page
    elif nav == "👥 Leaderboard":
        st.title("👥 Global Leaderboard")
        
        all_preds = get_all_predictions()
        
        if len(all_preds) > 0:
            # Top performers
            leaderboard = all_preds.groupby('username').agg({
                'predicted_score': 'mean',
                'hours_studied': 'mean',
                'accuracy': 'mean'
            }).round(2).sort_values('predicted_score', ascending=False).reset_index()
            
            leaderboard['Rank'] = range(1, len(leaderboard) + 1)
            st.subheader("🏆 Top Performers")
            st.dataframe(leaderboard[['Rank', 'username', 'predicted_score', 'hours_studied', 'accuracy']], 
                        use_container_width=True, hide_index=True)
            
            # Distribution chart
            fig = px.box(all_preds, x='username', y='predicted_score',
                        title='Score Distribution by User')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available!")
    
    # Settings page
    elif nav == "⚙️ Settings":
        st.title("⚙️ Settings")
        
        st.subheader("👤 Account Information")
        st.write(f"**Username:** {user_info[0]}")
        st.write(f"**Full Name:** {user_info[1]}")
        st.write(f"**Email:** {user_info[2]}")
        
        st.markdown("---")
        st.subheader("🔐 Change Password")
        
        with st.form("change_pwd_form"):
            old_pwd = st.text_input("Current Password", type="password")
            new_pwd = st.text_input("New Password", type="password")
            confirm_pwd = st.text_input("Confirm Password", type="password")
            change_submit = st.form_submit_button("Update Password")
            
            if change_submit:
                st.info("Password change feature coming soon!")
        
        st.markdown("---")
        st.subheader("⚠️ Danger Zone")
        
        if st.button("🗑️ Delete Account", use_container_width=True):
            st.warning("Account deletion feature coming soon!")
