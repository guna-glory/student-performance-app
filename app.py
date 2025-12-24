import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import io
import hashlib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Student Performance Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== DATABASE SETUP =====================
DB_FILE = "students.db"

def init_db():
    """Initialize SQLite database with all tables"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            department TEXT,
            semester INTEGER,
            marks1 INTEGER,
            marks2 INTEGER,
            marks3 INTEGER,
            attendance INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Admin users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # User authentication table (for student login)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_auth (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT,
            email TEXT,
            student_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    
    # Performance history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            exam_score REAL,
            exam_date TIMESTAMP,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    
    # Predictions log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            predicted_status TEXT,
            predicted_probability REAL,
            actual_status TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_default_admin():
    """Create default admin user if not exists"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        default_password = hash_password("admin123")
        cursor.execute(
            "INSERT INTO admin_users (username, password, email, full_name) VALUES (?, ?, ?, ?)",
            ("admin", default_password, "admin@school.com", "Admin User")
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    
    conn.close()

def verify_admin(username, password):
    """Verify admin credentials"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    hashed_password = hash_password(password)
    cursor.execute(
        "SELECT * FROM admin_users WHERE username = ? AND password = ?",
        (username, hashed_password)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    return result is not None

def get_all_students():
    """Fetch all students from database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM students")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    conn.close()
    
    if rows:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(columns=columns)

def add_student(name, age, gender, department, semester, marks1, marks2, marks3, attendance):
    """Add student to database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO students 
            (name, age, gender, department, semester, marks1, marks2, marks3, attendance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, age, gender, department, semester, marks1, marks2, marks3, attendance))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        st.error(f"Error adding student: {str(e)}")
        return False

def update_student(student_id, name, age, gender, department, semester, marks1, marks2, marks3, attendance):
    """Update student in database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE students 
            SET name=?, age=?, gender=?, department=?, semester=?, marks1=?, marks2=?, marks3=?, attendance=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        ''', (name, age, gender, department, semester, marks1, marks2, marks3, attendance, student_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        st.error(f"Error updating student: {str(e)}")
        return False

def delete_student(student_id):
    """Delete student from database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        cursor.execute("DELETE FROM performance_history WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM predictions_log WHERE student_id = ?", (student_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        st.error(f"Error deleting student: {str(e)}")
        return False

def upload_csv_to_db(df):
    """Upload CSV data to database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO students 
                (name, age, gender, department, semester, marks1, marks2, marks3, attendance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['Name'], int(row['Age']), row['Gender'], row['Department'], 
                  int(row['Semester']), int(row['Marks1']), int(row['Marks2']), 
                  int(row['Marks3']), int(row['Attendance'])))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        st.error(f"Error uploading CSV: {str(e)}")
        return False

def save_performance_history(student_id, exam_score, notes=""):
    """Save performance history"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO performance_history (student_id, exam_score, exam_date, notes)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
        ''', (student_id, exam_score, notes))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        st.error(f"Error saving performance: {str(e)}")
        return False

def get_performance_history(student_id):
    """Get performance history for a student"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM performance_history WHERE student_id = ? ORDER BY exam_date DESC
    ''', (student_id,))
    
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    
    if rows:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(columns=columns)

def save_prediction_log(student_id, predicted_status, predicted_probability):
    """Log ML prediction"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO predictions_log (student_id, predicted_status, predicted_probability)
            VALUES (?, ?, ?)
        ''', (student_id, predicted_status, predicted_probability))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        return False

def get_prediction_history():
    """Get all prediction history"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query('''
        SELECT pl.*, s.name FROM predictions_log pl
        JOIN students s ON pl.student_id = s.id
        ORDER BY pl.prediction_date DESC
    ''', conn)
    conn.close()
    return df

# ===================== THEME SYSTEM =====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Theme Settings")
theme_mode = st.sidebar.radio(
    "Select Theme:",
    options=["☀️ Light", "🌙 Dark"],
    horizontal=True,
    label_visibility="collapsed"
)

THEME = {
    "Light": {
        "bg_color": "#ffffff",
        "card_bg": "#f8f9fa",
        "text_color": "#1f2937",
        "text_secondary": "#6b7280",
        "border_color": "#e5e7eb",
        "accent": "#2563eb",
        "accent_light": "#dbeafe",
        "success": "#10b981",
        "danger": "#ef4444",
        "warning": "#f59e0b",
        "plotly_template": "plotly_white"
    },
    "Dark": {
        "bg_color": "#0f172a",
        "card_bg": "#1e293b",
        "text_color": "#f1f5f9",
        "text_secondary": "#cbd5e1",
        "border_color": "#334155",
        "accent": "#3b82f6",
        "accent_light": "#1e3a8a",
        "success": "#34d399",
        "danger": "#f87171",
        "warning": "#fbbf24",
        "plotly_template": "plotly_dark"
    }
}

active_theme = THEME["Light"] if "Light" in theme_mode else THEME["Dark"]

# ===================== CUSTOM CSS =====================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {active_theme['bg_color']};
    }}
    .stMarkdown, .stText, p {{
        color: {active_theme['text_color']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {active_theme['text_color']} !important;
        font-weight: 700;
    }}
    .card {{
        background-color: {active_theme['card_bg']};
        border: 1px solid {active_theme['border_color']};
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }}
    .stButton > button {{
        background-color: {active_theme['accent']} !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {{
        background-color: {active_theme['card_bg']} !important;
        color: {active_theme['text_color']} !important;
        border-color: {active_theme['border_color']} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== HELPER FUNCTIONS =====================
def calculate_total_marks(row):
    """Calculate total marks"""
    marks_cols = ['marks1', 'marks2', 'marks3']
    return row[marks_cols].sum()

def calculate_percentage(total_marks):
    """Calculate percentage"""
    return (total_marks / 300) * 100

def calculate_gpa(percentage):
    """Convert percentage to GPA"""
    if percentage >= 90:
        return 4.0
    elif percentage >= 85:
        return 3.7
    elif percentage >= 80:
        return 3.5
    elif percentage >= 75:
        return 3.0
    elif percentage >= 70:
        return 2.7
    elif percentage >= 65:
        return 2.5
    elif percentage >= 60:
        return 2.0
    elif percentage >= 55:
        return 1.7
    elif percentage >= 50:
        return 1.5
    elif percentage >= 40:
        return 1.0
    else:
        return 0.0

def get_pass_fail(total_marks):
    """Determine pass/fail"""
    return "PASS" if total_marks >= 120 else "FAIL"

def enrich_dataframe(df):
    """Add calculated columns to dataframe"""
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    df['total_marks'] = df.apply(calculate_total_marks, axis=1)
    df['percentage'] = df['total_marks'].apply(calculate_percentage)
    df['gpa'] = df['percentage'].apply(calculate_gpa)
    df['status'] = df['total_marks'].apply(get_pass_fail)
    
    return df

def train_ml_model(df):
    """Train Logistic Regression model"""
    try:
        df_model = df.copy()
        df_model['status_binary'] = (df_model['status'] == 'PASS').astype(int)
        
        unique_classes = df_model['status_binary'].unique()
        
        if len(unique_classes) < 2:
            st.warning("⚠️ Cannot train model: Need both PASS and FAIL students")
            return None, None, None, None
        
        le_gender = LabelEncoder()
        le_dept = LabelEncoder()
        
        df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'])
        df_model['dept_encoded'] = le_dept.fit_transform(df_model['department'])
        
        features = ['marks1', 'marks2', 'marks3', 'attendance', 'gender_encoded', 'dept_encoded']
        X = df_model[features]
        y = df_model['status_binary']
        
        if len(df_model) < 2:
            st.warning("⚠️ Need at least 2 students to train model")
            return None, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return model, metrics, le_gender, le_dept
    
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None, None, None, None

# ===================== INITIALIZATION =====================
init_db()
create_default_admin()

# ===================== LOGIN SYSTEM =====================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if not st.session_state.logged_in:
    st.markdown(
        """
        <div style="text-align: center; padding: 50px 0;">
            <h1 style="font-size: 2.5em;">📊 Student Performance Pro</h1>
            <p style="font-size: 1.2em; color: #666;">Admin Dashboard with ML Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        st.subheader("🔐 Admin Login")
        
        username = st.text_input("Username", placeholder="Enter username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")
        
        if st.button("Login", use_container_width=True, key="login_btn"):
            if verify_admin(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        
        st.markdown("---")
        st.info(
            """
            **Demo Credentials:**
            - Username: `admin`
            - Password: `admin123`
            """
        )

# ===================== MAIN APP (LOGGED IN) =====================
else:
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    
    st.sidebar.write(f"**Logged in as:** {st.session_state.username}")
    
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="margin: 0; font-size: 2.5em;">📊 Student Performance Pro</h1>
            <p style="color: #6b7280; font-size: 1.1em; margin: 10px 0 0 0;">Admin Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    page = st.radio(
        "📄 Navigation",
        ["📊 Dashboard", "🔍 Student Finder", "📈 Analytics", "🔄 Compare", "📊 Performance Tracking", "⚙️ Admin Panel", "🤖 ML Predictions"],
        horizontal=True,
        key="nav_page"
    )
    
    st.markdown("---")
    
    df = get_all_students()
    
    if len(df) == 0 and page != "⚙️ Admin Panel":
        st.warning("📭 No students in database. Use **⚙️ Admin Panel** to add students or upload CSV.")
    
    if len(df) > 0:
        df = enrich_dataframe(df)
        
        st.sidebar.markdown("---")
        st.sidebar.header("🎛️ Filters")
        
        genders = ['All'] + sorted(df['gender'].unique().tolist())
        departments = ['All'] + sorted(df['department'].unique().tolist())
        statuses = ['All', 'PASS', 'FAIL']
        
        selected_gender = st.sidebar.selectbox("Gender", genders, key="filter_gender")
        selected_department = st.sidebar.selectbox("Department", departments, key="filter_dept")
        selected_status = st.sidebar.selectbox("Status", statuses, key="filter_status")
        
        filtered_df = df.copy()
        
        if selected_gender != 'All':
            filtered_df = filtered_df[filtered_df['gender'] == selected_gender]
        if selected_department != 'All':
            filtered_df = filtered_df[filtered_df['department'] == selected_department]
        if selected_status != 'All':
            filtered_df = filtered_df[filtered_df['status'] == selected_status]
        
        if st.sidebar.button("🔄 Reset Filters", use_container_width=True, key="reset_btn"):
            st.rerun()
        
        st.sidebar.write(f"**📊 Results:** {len(filtered_df)} / {len(df)} students")
    else:
        filtered_df = pd.DataFrame()
    
    # ===================== PAGE: DASHBOARD =====================
    if page == "📊 Dashboard":
        if len(df) == 0:
            st.warning("📭 No students in database yet.")
            st.info("Go to **⚙️ Admin Panel** → **📥 Upload CSV** to add students")
        else:
            st.subheader("📊 Dashboard Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Total Students", len(filtered_df), f"{len(df)} in DB")
            with col2:
                st.metric("📈 Avg GPA", f"{filtered_df['gpa'].mean():.2f}", "0-4.0 scale")
            with col3:
                pass_pct = (filtered_df['status'] == 'PASS').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
                st.metric("✅ Pass Rate", f"{pass_pct:.1f}%", f"{(filtered_df['status'] == 'PASS').sum()} passed")
            with col4:
                st.metric("📍 Avg Attendance", f"{filtered_df['attendance'].mean():.1f}%", "Target: 75%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📈 GPA Distribution")
                fig_hist = px.histogram(filtered_df, x='gpa', nbins=15, 
                                       color_discrete_sequence=[active_theme['accent']])
                fig_hist.update_layout(template=active_theme['plotly_template'], height=400, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.write("### 📊 Marks by Department")
                dept_avg = filtered_df.groupby('department')[['marks1', 'marks2', 'marks3']].mean()
                dept_avg['avg'] = dept_avg.mean(axis=1)
                fig_bar = px.bar(dept_avg.reset_index(), x='department', y='avg', 
                                color='avg', color_continuous_scale='Viridis')
                fig_bar.update_layout(template=active_theme['plotly_template'], height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🎯 Marks vs Attendance")
                fig_scatter = px.scatter(filtered_df, x='attendance', y='total_marks', 
                                        color='status', 
                                        color_discrete_map={'PASS': active_theme['success'], 'FAIL': active_theme['danger']})
                fig_scatter.update_layout(template=active_theme['plotly_template'], height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                st.write("### 🥧 Pass/Fail Distribution")
                status_counts = filtered_df['status'].value_counts()
                fig_pie = px.pie(values=status_counts.values, names=status_counts.index,
                                color_discrete_map={'PASS': active_theme['success'], 'FAIL': active_theme['danger']})
                fig_pie.update_layout(template=active_theme['plotly_template'], height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # ===================== PAGE: STUDENT FINDER =====================
    elif page == "🔍 Student Finder":
        if len(df) == 0:
            st.warning("📭 No students in database yet.")
            st.info("Go to **⚙️ Admin Panel** → **📥 Upload CSV** to add students")
        else:
            st.subheader("🔍 Student Finder & Details")
            
            search_name = st.text_input("🔎 Search by name", placeholder="Enter student name...", key="search_name")
            
            if search_name:
                search_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False)]
            else:
                search_df = filtered_df
            
            if len(search_df) > 0:
                st.write(f"**Found {len(search_df)} student(s)**")
                
                display_cols = ['name', 'gender', 'department', 'marks1', 'marks2', 'marks3', 'total_marks', 'percentage', 'gpa', 'status', 'attendance']
                display_df = search_df[display_cols].copy()
                display_df['percentage'] = display_df['percentage'].round(2)
                display_df['gpa'] = display_df['gpa'].round(2)
                
                st.dataframe(display_df, use_container_width=True)
                
                st.markdown("---")
                
                selected_student = st.selectbox("Select student for details", search_df['name'].values, key="select_student")
                
                if selected_student:
                    s_data = search_df[search_df['name'] == selected_student].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Name", s_data['name'])
                    with col2: st.metric("Department", s_data['department'])
                    with col3: st.metric("Gender", s_data['gender'])
                    with col4: st.metric("Attendance", f"{s_data['attendance']}%")
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Exam 1", int(s_data['marks1']))
                    with col2: st.metric("Exam 2", int(s_data['marks2']))
                    with col3: st.metric("Exam 3", int(s_data['marks3']))
                    
                    st.markdown("---")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Total", f"{int(s_data['total_marks'])}/300")
                    with col2: st.metric("Percentage", f"{s_data['percentage']:.2f}%")
                    with col3: st.metric("GPA", f"{s_data['gpa']:.2f}")
                    with col4: 
                        status_emoji = "✅" if s_data['status'] == 'PASS' else "❌"
                        st.metric("Status", f"{status_emoji} {s_data['status']}")
            
            else:
                st.info("No students found")
    
    # ===================== PAGE: ANALYTICS =====================
    elif page == "📈 Analytics":
        if len(df) == 0:
            st.warning("📭 No students in database yet.")
            st.info("Go to **⚙️ Admin Panel** → **📥 Upload CSV** to add students")
        else:
            st.subheader("📈 Detailed Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Department Performance")
                dept_stats = filtered_df.groupby('department').agg({
                    'total_marks': ['mean', 'min', 'max'],
                    'gpa': 'mean'
                }).round(2)
                st.dataframe(dept_stats, use_container_width=True)
            
            with col2:
                st.write("### 👥 Gender Distribution")
                gender_stats = filtered_df.groupby('gender').agg({
                    'total_marks': 'mean',
                    'gpa': 'mean',
                    'attendance': 'mean'
                }).round(2)
                st.dataframe(gender_stats, use_container_width=True)
            
            st.markdown("---")
            
            st.write("### 📊 Pass Rate by Department")
            dept_pass = filtered_df.groupby('department')['status'].apply(
                lambda x: (x == 'PASS').sum() / len(x) * 100
            ).reset_index()
            dept_pass.columns = ['department', 'pass_rate']
            
            fig_pass = px.bar(dept_pass, x='department', y='pass_rate', 
                             color='pass_rate', color_continuous_scale='RdYlGn')
            fig_pass.update_layout(template=active_theme['plotly_template'], height=400)
            st.plotly_chart(fig_pass, use_container_width=True)
            
            st.markdown("---")
            
            st.write("### 📦 Marks Distribution by Department")
            fig_box = px.box(filtered_df, x='department', y='total_marks', color='department')
            fig_box.update_layout(template=active_theme['plotly_template'], height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # ===================== PAGE: COMPARE =====================
    elif page == "🔄 Compare":
        if len(df) == 0:
            st.warning("📭 No students in database yet.")
            st.info("Go to **⚙️ Admin Panel** → **📥 Upload CSV** to add students")
        elif len(filtered_df) < 2:
            st.warning("⚠️ Need at least 2 students to compare")
        else:
            st.subheader("🔄 Compare Students")
            
            col1, col2 = st.columns(2)
            
            with col1:
                student1 = st.selectbox("Student 1", filtered_df['name'].values, key="compare_s1")
            with col2:
                student2 = st.selectbox("Student 2", filtered_df['name'].values, key="compare_s2")
            
            if student1 and student2 and student1 != student2:
                s1 = filtered_df[filtered_df['name'] == student1].iloc[0]
                s2 = filtered_df[filtered_df['name'] == student2].iloc[0]
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 0.5, 1])
                
                with col1:
                    st.write(f"#### {s1['name']}")
                    st.metric("Total Marks", f"{int(s1['total_marks'])}/300")
                    st.metric("GPA", f"{s1['gpa']:.2f}")
                    st.metric("Percentage", f"{s1['percentage']:.2f}%")
                    st.metric("Attendance", f"{s1['attendance']}%")
                    st.metric("Status", "✅ PASS" if s1['status'] == 'PASS' else "❌ FAIL")
                
                with col2:
                    st.write("")
                
                with col3:
                    st.write(f"#### {s2['name']}")
                    st.metric("Total Marks", f"{int(s2['total_marks'])}/300")
                    st.metric("GPA", f"{s2['gpa']:.2f}")
                    st.metric("Percentage", f"{s2['percentage']:.2f}%")
                    st.metric("Attendance", f"{s2['attendance']}%")
                    st.metric("Status", "✅ PASS" if s2['status'] == 'PASS' else "❌ FAIL")
                
                st.markdown("---")
                
                st.write("### 📊 Exam Marks Comparison")
                comp_data = pd.DataFrame({
                    'Exam': ['Exam 1', 'Exam 2', 'Exam 3'],
                    student1: [s1['marks1'], s1['marks2'], s1['marks3']],
                    student2: [s2['marks1'], s2['marks2'], s2['marks3']]
                })
                
                fig_comp = px.bar(comp_data, x='Exam', y=[student1, student2], barmode='group',
                                 color_discrete_sequence=[active_theme['accent'], active_theme['success']])
                fig_comp.update_layout(template=active_theme['plotly_template'], height=400)
                st.plotly_chart(fig_comp, use_container_width=True)
            
            elif student1 == student2:
                st.warning("Select two different students")
    
    # ===================== PAGE: PERFORMANCE TRACKING =====================
    elif page == "📊 Performance Tracking":
        st.subheader("📊 Performance History & Trends")
        
        if len(df) == 0:
            st.warning("📭 No students in database yet.")
            st.info("Go to **⚙️ Admin Panel** to add students first.")
        else:
            # Select student
            selected_student = st.selectbox("📌 Select Student", df['name'].values, key="perf_select")
            student_data = df[df['name'] == selected_student].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Name", student_data['name'])
            with col2: st.metric("Department", student_data['department'])
            with col3: st.metric("Current GPA", f"{student_data['gpa']:.2f}")
            with col4: st.metric("Status", "✅ PASS" if student_data['status'] == 'PASS' else "❌ FAIL")
            
            st.markdown("---")
            
            # Add performance entry
            st.write("### ➕ Add Performance Entry")
            col1, col2 = st.columns(2)
            
            with col1:
                exam_score = st.number_input("Exam Score", min_value=0, max_value=100, value=85, key="perf_score")
            with col2:
                notes = st.text_input("Notes (optional)", placeholder="e.g., Improved time management", key="perf_notes")
            
            if st.button("💾 Save Performance Entry", use_container_width=True, key="save_perf"):
                if save_performance_history(student_data['id'], exam_score, notes):
                    st.success("✅ Performance entry saved!")
                    st.rerun()
            
            st.markdown("---")
            
            # Show performance history
            st.write("### 📈 Performance History")
            perf_history = get_performance_history(student_data['id'])
            
            if len(perf_history) > 0:
                st.dataframe(perf_history[['exam_score', 'exam_date', 'notes']].head(10), use_container_width=True)
                
                # Performance trend chart
                fig_trend = px.line(perf_history.iloc[::-1], y='exam_score', 
                                   title=f"Performance Trend - {selected_student}",
                                   markers=True)
                fig_trend.update_layout(template=active_theme['plotly_template'], height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                st.write(f"**Average Score:** {perf_history['exam_score'].mean():.2f}")
                st.write(f"**Highest Score:** {perf_history['exam_score'].max():.2f}")
                st.write(f"**Lowest Score:** {perf_history['exam_score'].min():.2f}")
            else:
                st.info("No performance history for this student yet.")
    
    # ===================== PAGE: ADMIN PANEL =====================
    elif page == "⚙️ Admin Panel":
        st.subheader("⚙️ Admin Panel")
        
        admin_tab1, admin_tab2, admin_tab3 = st.tabs(["➕ Add Student", "📥 Upload CSV", "📊 Manage Students"])
        
        # Tab 1: Add Student
        with admin_tab1:
            st.write("### ➕ Add New Student")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name", placeholder="John Doe", key="add_name")
                age = st.number_input("Age", min_value=15, max_value=30, value=20, key="add_age")
                gender = st.selectbox("Gender", ["Male", "Female"], key="add_gender")
            
            with col2:
                department = st.selectbox("Department", ["CSE", "ECE", "MECH"], key="add_dept")
                semester = st.number_input("Semester", min_value=1, max_value=8, value=4, key="add_sem")
                attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=85, key="add_att")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                marks1 = st.number_input("Marks 1", min_value=0, max_value=100, value=80, key="add_m1")
            with col2:
                marks2 = st.number_input("Marks 2", min_value=0, max_value=100, value=80, key="add_m2")
            with col3:
                marks3 = st.number_input("Marks 3", min_value=0, max_value=100, value=80, key="add_m3")
            
            if st.button("✅ Add Student", use_container_width=True, key="add_btn"):
                if name:
                    if add_student(name, age, gender, department, semester, marks1, marks2, marks3, attendance):
                        st.success(f"✅ Student '{name}' added successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to add student")
                else:
                    st.error("Please enter student name")
        
        # Tab 2: Upload CSV
        with admin_tab2:
            st.write("### 📥 Upload CSV to Database")
            st.info("✅ CSV must have columns: Name, Age, Gender, Department, Semester, Marks1, Marks2, Marks3, Attendance")
            
            uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="csv_upload")
            
            if uploaded_file is not None:
                try:
                    csv_df = pd.read_csv(uploaded_file)
                    
                    st.write("**Preview (first 5 rows):**")
                    st.dataframe(csv_df.head(), use_container_width=True)
                    
                    st.write(f"**Total rows to upload:** {len(csv_df)}")
                    
                    if st.button("📤 Upload to Database", use_container_width=True, key="csv_btn"):
                        if upload_csv_to_db(csv_df):
                            st.success(f"✅ {len(csv_df)} students uploaded successfully!")
                            st.balloons()
                            st.rerun()
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
                    st.info("Make sure your CSV has the correct column names and format")
        
        # Tab 3: Manage Students
        with admin_tab3:
            st.write("### 📊 Manage Students")
            
            if len(df) == 0:
                st.info("📭 No students in database. Add students using **➕ Add Student** or **📥 Upload CSV** tab first.")
            else:
                selected = st.selectbox("Select student", df['name'].values, key="manage_select")
                student = df[df['name'] == selected].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### ✏️ Edit Student")
                    new_name = st.text_input("Name", value=student['name'], key="edit_name")
                    new_age = st.number_input("Age", value=int(student['age']), key="edit_age")
                    new_gender = st.selectbox("Gender", ["Male", "Female"], 
                                             index=0 if student['gender'] == "Male" else 1, key="edit_gender")
                    new_dept = st.selectbox("Department", ["CSE", "ECE", "MECH"],
                                          index=["CSE", "ECE", "MECH"].index(student['department']), key="edit_dept")
                    new_sem = st.number_input("Semester", value=int(student['semester']), key="edit_sem")
                    new_m1 = st.number_input("Marks 1", value=int(student['marks1']), key="edit_m1")
                    new_m2 = st.number_input("Marks 2", value=int(student['marks2']), key="edit_m2")
                    new_m3 = st.number_input("Marks 3", value=int(student['marks3']), key="edit_m3")
                    new_att = st.number_input("Attendance", value=int(student['attendance']), key="edit_att")
                    
                    if st.button("💾 Update Student", use_container_width=True, key="update_btn"):
                        if update_student(student['id'], new_name, new_age, new_gender, new_dept, 
                                        new_sem, new_m1, new_m2, new_m3, new_att):
                            st.success("✅ Student updated!")
                            st.rerun()
                
                with col2:
                    st.write("#### 🗑️ Delete Student")
                    st.write(f"**Student:** {student['name']}")
                    st.write(f"**Department:** {student['department']}")
                    enriched = enrich_dataframe(pd.DataFrame([student]))
                    st.write(f"**GPA:** {enriched['gpa'].iloc[0]:.2f}")
                    st.write(f"**Status:** {enriched['status'].iloc[0]}")
                    
                    st.markdown("---")
                    
                    if st.button("🗑️ Delete Student", use_container_width=True, key="delete_btn"):
                        if delete_student(student['id']):
                            st.success("✅ Student deleted!")
                            st.rerun()
    
    # ===================== PAGE: ML PREDICTIONS =====================
    elif page == "🤖 ML Predictions":
        st.subheader("🤖 Machine Learning Predictions")
        
        if len(df) == 0:
            st.warning("📭 No students in database yet.")
            st.info("Go to **⚙️ Admin Panel** → **📥 Upload CSV** to add students")
        else:
            pass_count = (df['status'] == 'PASS').sum()
            fail_count = (df['status'] == 'FAIL').sum()
            
            st.write(f"**Database Statistics:** {len(df)} total | {pass_count} PASS | {fail_count} FAIL")
            
            if len(df) < 5:
                st.warning("⚠️ Need at least 5 students in database to train model")
            elif pass_count == 0 or fail_count == 0:
                st.error("❌ Cannot train model: Missing PASS or FAIL students")
                st.info("""
                **To fix this:**
                1. Go to Admin Panel → Add Student
                2. Add students with low marks (< 120 total) to create FAIL cases
                3. Add students with high marks (≥ 120 total) to create PASS cases
                4. Then return to this page
                """)
                
                st.markdown("---")
                st.write("### 📊 Current Status Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ PASS Students", pass_count)
                with col2:
                    st.metric("❌ FAIL Students", fail_count)
            
            else:
                st.write("### 🤖 Logistic Regression Model")
                
                model, metrics, le_gender, le_dept = train_ml_model(df)
                
                if model and metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Accuracy", f"{metrics['accuracy']:.2%}")
                    with col2:
                        st.metric("✅ Precision", f"{metrics['precision']:.2%}")
                    with col3:
                        st.metric("🎯 Recall", f"{metrics['recall']:.2%}")
                    with col4:
                        st.metric("⚖️ F1-Score", f"{metrics['f1']:.2%}")
                    
                    st.markdown("---")
                    
                    st.write("### 🔮 Predict Student Status")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pred_m1 = st.number_input("Exam 1 Marks", min_value=0, max_value=100, value=80, key="pred_m1")
                        pred_m2 = st.number_input("Exam 2 Marks", min_value=0, max_value=100, value=80, key="pred_m2")
                        pred_m3 = st.number_input("Exam 3 Marks", min_value=0, max_value=100, value=80, key="pred_m3")
                    
                    with col2:
                        pred_att = st.number_input("Attendance %", min_value=0, max_value=100, value=85, key="pred_att")
                        pred_gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
                        pred_dept = st.selectbox("Department", ["CSE", "ECE", "MECH"], key="pred_dept")
                    
                    if st.button("🔮 Predict", use_container_width=True, key="predict_btn"):
                        try:
                            gender_enc = le_gender.transform([pred_gender])[0]
                            dept_enc = le_dept.transform([pred_dept])[0]
                            
                            X_pred = np.array([[pred_m1, pred_m2, pred_m3, pred_att, gender_enc, dept_enc]])
                            prediction = model.predict(X_pred)[0]
                            probability = model.predict_proba(X_pred)[0]
                            
                            st.markdown("---")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                status = "✅ PASS" if prediction == 1 else "❌ FAIL"
                                st.metric("Prediction", status)
                            
                            with col2:
                                confidence = probability[prediction] * 100
                                st.metric("Confidence", f"{confidence:.1f}%")
                            
                            st.markdown("---")
                            
                            st.write("### 📊 Probability Distribution")
                            prob_data = pd.DataFrame({
                                'Status': ['PASS', 'FAIL'],
                                'Probability': [probability[1] * 100, probability[0] * 100]
                            })
                            
                            fig_prob = px.bar(prob_data, x='Status', y='Probability',
                                            color='Status',
                                            color_discrete_map={'PASS': active_theme['success'], 'FAIL': active_theme['danger']})
                            fig_prob.update_layout(template=active_theme['plotly_template'], height=300, showlegend=False)
                            st.plotly_chart(fig_prob, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
        
        # ===================== DOWNLOAD SECTION =====================
        if len(df) > 0:
            st.markdown("---")
            st.subheader("💾 Download Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_filtered = filtered_df.to_csv(index=False).encode()
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv_filtered,
                    file_name="filtered_students.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_filtered"
                )
            
            with col2:
                csv_all = df.to_csv(index=False).encode()
                st.download_button(
                    label="📥 Download All Data",
                    data=csv_all,
                    file_name="all_students.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_all"
                )
