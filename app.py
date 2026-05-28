import os
import pandas as pd
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediChat",
    page_icon="💚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ── Load dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_medical_data():
    symptoms_df   = pd.read_csv("DiseaseAndSymptoms.csv")
    precaution_df = pd.read_csv("Disease_precaution.csv")

    symptom_cols    = [c for c in symptoms_df.columns   if c.startswith("Symptom_")]
    precaution_cols = [c for c in precaution_df.columns if c.startswith("Precaution_")]

    disease_symptoms = {}
    for _, row in symptoms_df.iterrows():
        d    = row["Disease"]
        syms = [str(row[c]).strip().replace("_", " ")
                for c in symptom_cols if pd.notna(row[c]) and str(row[c]).strip()]
        disease_symptoms.setdefault(d, set()).update(syms)

    disease_precautions = {}
    for _, row in precaution_df.iterrows():
        d     = row["Disease"]
        precs = [str(row[c]).strip()
                 for c in precaution_cols if pd.notna(row[c]) and str(row[c]).strip()]
        disease_precautions[d] = precs

    lines = []
    for disease, syms in disease_symptoms.items():
        precs = disease_precautions.get(disease, [])
        lines.append(
            f"Disease: {disease}\n"
            f"  Symptoms: {', '.join(sorted(syms))}\n"
            f"  Precautions: {'; '.join(precs) if precs else 'N/A'}"
        )
    return "\n\n".join(lines)

medical_context = load_medical_data()

SYSTEM_PROMPT = f"""You are MediChat, a helpful and compassionate medical information assistant.
You have access to a curated medical dataset containing diseases, their associated symptoms,
and recommended precautions.

IMPORTANT GUIDELINES:
- Always remind users that you are NOT a substitute for professional medical advice.
- Be empathetic, clear, and concise.
- If symptoms described match multiple diseases, list the top possibilities.
- Always recommend consulting a licensed doctor for diagnosis and treatment.
- Do not prescribe medications.

MEDICAL KNOWLEDGE BASE:
{medical_context}
"""

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #f2faf6 !important;
    color: #163a2b;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
.stDeployButton { display: none !important; visibility: hidden !important; }

/* ── Layout ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Top navbar ── */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 200;
    background: #0b6d51;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.navbar-logo {
    width: 36px; height: 36px;
    background: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.navbar-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.01em;
}
.navbar-sub {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.75);
    line-height: 1;
}
.navbar-clear {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    border-radius: 8px;
    padding: 0.4rem 0.85rem;
    font-size: 0.82rem;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    transition: background 0.2s;
}
.navbar-clear:hover { background: rgba(255,255,255,0.25); }

/* ── Page wrapper ── */
.page-wrap {
    max-width: 780px;
    margin: 0 auto;
    padding: 5.5rem 1.25rem 7rem;
    min-height: 100vh;
}

/* ── Welcome screen ── */
.welcome-hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.welcome-icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
    display: block;
}
.welcome-heading {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.5rem, 4vw, 2rem);
    color: #0b6d51;
    font-weight: 400;
    margin-bottom: 0.5rem;
}
.welcome-sub {
    font-size: 0.95rem;
    color: #5a8070;
    max-width: 420px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}

/* Suggestion cards */
.cards-label {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a8070;
    margin-bottom: 0.75rem;
    text-align: center;
}

/* Override Streamlit buttons used as cards */
div[data-testid="stButton"] > button {
    background: white !important;
    color: #0b6d51 !important;
    border: 1.5px solid #b8dece !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 0.9rem 1.1rem !important;
    text-align: left !important;
    width: 100% !important;
    line-height: 1.45 !important;
    transition: all 0.18s !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    cursor: pointer !important;
    white-space: normal !important;
}
div[data-testid="stButton"] > button:hover {
    background: #e6f5ee !important;
    border-color: #0b6d51 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(11,109,81,0.14) !important;
}

/* ── Chat bubbles ── */
.bubble-row {
    display: flex;
    gap: 0.65rem;
    margin-bottom: 1rem;
    align-items: flex-end;
}
.bubble-row.user-row { flex-direction: row-reverse; }

.avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
    flex-shrink: 0;
}
.av-bot  { background: #0b6d51; }
.av-user { background: #13a77b; }

.bubble {
    max-width: min(72%, 560px);
    padding: 0.85rem 1.1rem;
    font-size: 0.93rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
}
.bubble-bot {
    background: white;
    border: 1px solid #d5e8df;
    border-radius: 18px 18px 18px 4px;
    color: #1a2e25;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.bubble-user {
    background: #0b6d51;
    color: white;
    border-radius: 18px 18px 4px 18px;
}

/* ── Bottom input bar ── */
.input-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #f2faf6;
    border-top: 1px solid #c8e3d6;
    padding: 0.8rem 1.25rem 0.9rem;
    z-index: 200;
}
.input-bar-inner {
    max-width: 780px;
    margin: 0 auto;
}
.disclaimer-text {
    text-align: center;
    font-size: 0.72rem;
    color: #8aad9b;
    margin-top: 0.4rem;
}

/* Chat input pill */
[data-testid="stChatInputTextArea"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
}
div[data-testid="stChatInput"] > div {
    border-radius: 50px !important;
    border: 1.5px solid #b2dac8 !important;
    background: white !important;
    box-shadow: 0 2px 10px rgba(11,109,81,0.08) !important;
    padding: 0.1rem 0.5rem !important;
}
button[data-testid="stChatInputSubmitButton"] {
    background: #0b6d51 !important;
    border-radius: 50% !important;
}

/* ── Responsive ── */
@media (max-width: 600px) {
    .navbar { padding: 0.7rem 1rem; }
    .navbar-title { font-size: 1rem; }
    .page-wrap { padding: 5rem 0.85rem 6.5rem; }
    .bubble { max-width: 85%; font-size: 0.89rem; }
    .welcome-heading { font-size: 1.35rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "clear_trigger" not in st.session_state:
    st.session_state.clear_trigger = False

SUGGESTIONS = [
    "What are common symptoms of the flu?",
    "How can I prevent getting a cold?",
    "What should I do for a headache?",
    "When should I see a doctor for a fever?",
]

def ask_groq(messages):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content

def send_message(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        reply = ask_groq(st.session_state.messages)
    except Exception as e:
        reply = f"⚠️ Sorry, I encountered an error: {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-logo">💚</div>
        <div>
            <div class="navbar-title">MediChat</div>
            <div class="navbar-sub">Your Health Assistant</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Clear button in navbar via Streamlit (positioned via CSS)
col_nav1, col_nav2 = st.columns([6, 1])
with col_nav2:
    if st.button("🗑️ Clear", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

messages = st.session_state.messages

# Welcome screen
if not messages:
    st.markdown("""
    <div class="welcome-hero">
        <span class="welcome-icon">🩺</span>
        <div class="welcome-heading">How can I help you today?</div>
        <div class="welcome-sub">
            Describe your symptoms or ask about a condition and I'll provide
            information and precautions based on medical data.
        </div>
    </div>
    <p class="cards-label">Try asking about:</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for idx, suggestion in enumerate(SUGGESTIONS):
        with (col1 if idx % 2 == 0 else col2):
            if st.button(suggestion, key=f"sug_{idx}"):
                send_message(suggestion)
                st.rerun()

# Chat messages
for msg in messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="bubble-row user-row">
            <div class="avatar av-user">🧑</div>
            <div class="bubble bubble-user">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bubble-row">
            <div class="avatar av-bot">💚</div>
            <div class="bubble bubble-bot">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Fixed input bar ───────────────────────────────────────────────────────────
st.markdown('<div class="input-bar"><div class="input-bar-inner">', unsafe_allow_html=True)

if prompt := st.chat_input("Ask about symptoms, precautions..."):
    send_message(prompt)
    st.rerun()

st.markdown("""
    <p class="disclaimer-text">For informational purposes only. Always consult a healthcare professional.</p>
    </div></div>
""", unsafe_allow_html=True)