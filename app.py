import streamlit as st
import pandas as pd
import os
from model import train_spam_model, predict_message

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Spam vs Ham Email Classifier",
    page_icon="📧",
    layout="wide"
)

# --- THEME COLORS (Original Cyberpunk) ---
MAIN_BG = "#21112D" 
ACCENT_PINK = "#FF007F" 
GLOW_PURPLE = "#7F00FF"
LIGHT_TEXT = "#FFFFFF"

# Apply Custom CSS for the premium design
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&display=swap');

    /* Global Styles */
    .stApp {{
        background: radial-gradient(circle at top right, #3D1C4A, {MAIN_BG}),
                    radial-gradient(circle at bottom left, #2D1A3D, {MAIN_BG});
        color: {LIGHT_TEXT};
        font-family: 'Space Grotesk', sans-serif;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: rgba(33, 17, 45, 0.9) !important;
        backdrop-filter: blur(20px);
        border-right: 2px solid {ACCENT_PINK};
    }}

    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {{
        color: {ACCENT_PINK} !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* Sidebar File Uploader Button */
    [data-testid="stFileUploader"] button {{
        background: linear-gradient(45deg, {ACCENT_PINK}, {GLOW_PURPLE}) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        transition: 0.3s !important;
        box-shadow: 0 5px 15px rgba(255, 0, 127, 0.3) !important;
        width: 100%;
    }}

    .main-header {{
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(to right, {ACCENT_PINK}, {GLOW_PURPLE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700;
        font-size: 4rem;
        letter-spacing: -2px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }}

    /* Input Field */
    .stTextArea textarea {{
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid {GLOW_PURPLE} !important;
        color: white !important;
        border-radius: 12px !important;
    }}

    /* Button Styling */
    .stButton > button {{
        background: linear-gradient(45deg, {ACCENT_PINK}, {GLOW_PURPLE}) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        transition: 0.3s;
        box-shadow: 0 5px 15px rgba(255, 0, 127, 0.3) !important;
        width: 100%;
    }}

    /* Result Badges */
    .result-badge {{
        padding: 20px 40px;
        border-radius: 15px;
        font-weight: 700;
        font-size: 2.5rem;
        display: inline-block;
        margin-bottom: 20px;
        text-transform: uppercase;
        text-align: center;
        width: 100%;
    }}
    .safe {{ 
        border: 2px solid #00FF9F; 
        color: #00FF9F; 
        box-shadow: 0 0 20px rgba(0, 255, 159, 0.2); 
    }}
    .spam {{ 
        border: 2px solid {ACCENT_PINK}; 
        color: {ACCENT_PINK}; 
        box-shadow: 0 0 20px rgba(255, 0, 127, 0.2); 
    }}

    .confidence-box {{
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }}

    /* History Cards */
    .history-card {{
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid {GLOW_PURPLE};
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        border: 1px solid rgba(127, 0, 255, 0.2);
    }}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []

if 'vectorizer' not in st.session_state or 'model' not in st.session_state:
    with st.status("Initializing AI System...", expanded=True) as status:
        st.write("Loading patterns from dataset.csv...")
        v, m, err = train_spam_model()
        if v and m:
            st.session_state.vectorizer = v
            st.session_state.model = m
            status.update(label="System Ready!", state="complete", expanded=False)
        else:
            status.update(label="Initialization Failed", state="error")
            st.error(f"Error: {err}")
            st.stop()

# --- SIDEBAR: RECENT HISTORY ---
with st.sidebar:
    st.markdown("## BATCH PROCESSING")
    uploaded_file = st.file_uploader("UPLOAD EMAILS CSV", type=["csv"])
    st.info("CSV should have a column named 'Message'")

    st.markdown("---")
    st.markdown("## RECENT SCANS")
    if not st.session_state.history:
        st.write("No history available.")
    else:
        for item in reversed(st.session_state.history[-5:]):
            label_color = "#00FF9F" if item['label'] == "Ham" else ACCENT_PINK
            st.markdown(f"""
            <div class="history-card">
                <span style="color:{label_color}; font-weight:bold;">{item['label'].upper()}</span> 
                <span style="color:#AAA;">({item['conf']:.1f}%)</span><br>
                <div style="color:#EEE; margin-top:5px; font-style:italic;">{item['text'][:40]}...</div>
            </div>
            """, unsafe_allow_html=True)

# --- MAIN APP LAYOUT ---
st.image("preview.png", use_column_width=True)
st.markdown('<h1 class="main-header">Spam vs Ham Email Classifier</h1>', unsafe_allow_html=True)

# Layout Columns
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    if uploaded_file is not None:
        st.subheader("BATCH ANALYSIS RESULTS")
        df_batch = pd.read_csv(uploaded_file)
        # Check for message column (case insensitive)
        msg_col = next((c for c in df_batch.columns if c.lower() in ['message', 'text', 'body']), None)
        
        if msg_col:
            with st.spinner("Analyzing contents..."):
                results = []
                confidences = []
                for msg in df_batch[msg_col]:
                    label, conf = predict_message(str(msg), st.session_state.vectorizer, st.session_state.model)
                    results.append(label)
                    confidences.append(f"{conf:.1f}%")
                
                df_batch['PREDICTION'] = results
                df_batch['CONFIDENCE'] = confidences
                st.dataframe(df_batch, use_container_width=True)
        else:
            st.error("Error: CSV must contain a 'Message' column.")
    else:
        email_input = st.text_area("ANALYZE MESSAGE CONTENT", placeholder="Paste email content here...", height=250)
        
        if st.button("VERIFY INTEGRITY"):
            input_text = email_input.strip()
            if input_text:
                # Basic validation: ensure it has at least 3 words and 15 characters
                if len(input_text.split()) < 3 or len(input_text) < 15:
                    st.error("⚠️ INPUT TOO SHORT: Please provide a more detailed email body (at least 3 words) to ensure accurate AI detection.")
                else:
                    with st.spinner("Detecting patterns..."):
                        label, confidence = predict_message(input_text, st.session_state.vectorizer, st.session_state.model)
                    
                    # Update History
                    st.session_state.history.append({
                        'text': email_input,
                        'label': label,
                        'conf': confidence
                    })
                    
                    # Display Result
                    badge_class = "safe" if label == "Ham" else "spam"
                    display_label = "SAFE (HAM)" if label == "Ham" else "SPAM DETECTED"
                    
                    st.markdown(f'<div class="result-badge {badge_class}">{display_label}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="confidence-box">
                            <h2 style="margin:0; font-size: 3rem; color: white;">{confidence:.1f}%</h2>
                            <p style="margin:0; color: #AAA; letter-spacing: 2px;">CONFIDENCE LEVEL</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze.")
