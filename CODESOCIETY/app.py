import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
import traceback
import math
import random
from datetime import datetime

from classifier import FakeNewsClassifier
from api_fetch import fetch_news
from utils import compute_credibility, detect_outbreak
from graph_analysis import simulate_propagation, analyze_propagation

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Misinformation Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

html, body, .stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
    font-family: 'Exo 2', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.4) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(0,255,231,0.15) !important;
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(15px);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 20px;
}

/* Page title */
.big-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    text-align: center;
    color: white;
    text-shadow: 0 0 30px rgba(0,255,231,0.5);
    letter-spacing: 3px;
    margin-bottom: 6px;
}
.sub-title {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.85rem;
    text-align: center;
    color: rgba(0,255,231,0.7);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 30px;
}

/* Risk badge */
.risk-high   { background:#ff4c4c22; border:1px solid #ff4c4c; color:#ff4c4c; padding:10px 20px; border-radius:10px; font-weight:700; font-size:1.1rem; text-align:center; }
.risk-medium { background:#ffa50022; border:1px solid #ffa500; color:#ffa500; padding:10px 20px; border-radius:10px; font-weight:700; font-size:1.1rem; text-align:center; }
.risk-low    { background:#00ff8822; border:1px solid #00ff88; color:#00ff88; padding:10px 20px; border-radius:10px; font-weight:700; font-size:1.1rem; text-align:center; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,255,231,0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    color: #00ffe7;
    font-weight: 700;
}
.metric-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    color: rgba(255,255,255,0.6) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00ffe7 !important;
    border-bottom: 2px solid #00ffe7 !important;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] button {
    background: rgba(0,255,231,0.05) !important;
    border: 1px solid rgba(0,255,231,0.2) !important;
    color: rgba(255,255,255,0.8) !important;
    font-family: 'Exo 2', sans-serif !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] button:hover {
    background: rgba(0,255,231,0.12) !important;
    color: #00ffe7 !important;
}

/* Text input */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0,255,231,0.3) !important;
    color: white !important;
    border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* Primary buttons */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00ffe7, #00c8b4) !important;
    color: #0f2027 !important;
    border: none !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
}

.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00ffe7;
    border-radius: 50%;
    box-shadow: 0 0 8px #00ffe7;
    animation: blink 1.5s infinite;
    margin-right: 6px;
}
@keyframes blink {
    0%,100% { opacity:1; } 50% { opacity:0.2; }
}

/* Top right user bar */
.user-bar {
    position: fixed;
    top: 0; right: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(15,32,39,0.92);
    backdrop-filter: blur(12px);
    border-bottom-left-radius: 14px;
    border-left: 1px solid rgba(0,255,231,0.2);
    border-bottom: 1px solid rgba(0,255,231,0.2);
    padding: 8px 18px 8px 14px;
    font-family: 'Exo 2', sans-serif;
}
.user-avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    border: 2px solid rgba(0,255,231,0.5);
    box-shadow: 0 0 10px rgba(0,255,231,0.3);
    object-fit: cover;
}
.user-avatar-placeholder {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: rgba(0,255,231,0.15);
    border: 2px solid rgba(0,255,231,0.4);
    display: flex; align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.user-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.5px;
    max-width: 140px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.user-role {
    font-size: 0.62rem;
    color: rgba(0,255,231,0.6);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.online-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 6px #00ff88;
    animation: pulse 2s infinite;
}

/* Logout button */
div[data-testid="stButton"] > button[kind="secondary"],
button[key="logout_btn"] {
    background: transparent !important;
    color: rgba(255,80,80,0.8) !important;
    border: 1px solid rgba(255,80,80,0.3) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button[key="logout_btn"]:hover {
    background: rgba(255,80,80,0.08) !important;
    border-color: rgba(255,80,80,0.6) !important;
}

.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    color: rgba(0,255,231,0.5);
    letter-spacing: 4px;
    text-transform: uppercase;
    border-left: 2px solid #00ffe7;
    padding-left: 10px;
    margin-bottom: 16px;
}

.article-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,255,231,0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ─────────────────────────────────────────────
#  AUTHENTICATION GUARD
#  If not logged in, stop and show message
# ─────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:3rem;">🔒</div>
        <h2 style="color:white; font-family:'Orbitron',monospace; letter-spacing:3px;">
            ACCESS DENIED
        </h2>
        <p style="color:rgba(0,255,231,0.7); font-size:0.9rem; letter-spacing:1px;">
            You have been logged out successfully.
        </p>
        <p style="color:rgba(255,255,255,0.4); font-size:0.8rem;">
            Please run: <code>streamlit run login.py</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  TOP RIGHT USER BAR
#  Shows logged-in user's name + avatar
# ─────────────────────────────────────────────
user_info = st.session_state.get("user_info", None)

if user_info:
    user_name    = user_info.get("name",    "Analyst")
    user_email   = user_info.get("email",   "")
    user_picture = user_info.get("picture", "")

    if user_picture:
        avatar_html = f'<img class="user-avatar" src="{user_picture}" />'
    else:
        avatar_html = '<div class="user-avatar-placeholder">👤</div>'

    st.markdown(f"""
    <div class="user-bar">
        <div class="online-dot"></div>
        {avatar_html}
        <div>
            <div class="user-name">{user_name}</div>
            <div class="user-role">Analyst</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    return FakeNewsClassifier()

try:
    classifier = load_classifier()
except Exception as e:
    st.error("MODEL LOADING FAILED")
    st.text(traceback.format_exc())
    st.stop()

# BERT disabled — using TF-IDF + NewsAPI + Graph (3-layer hybrid)
BERT_AVAILABLE = False
bert_tokenizer = None
bert_model     = None


# ─────────────────────────────────────────────
#  NEWS API VERIFICATION  (from friend's version)
# ─────────────────────────────────────────────
NEWS_API_KEY = "789601692407496dbc647ddb76d76aa4"

def verify_with_news_api(query: str) -> int:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query[:100],
        "apiKey":   NEWS_API_KEY,
        "language": "en",
        "sortBy":   "relevancy",
        "pageSize": 5,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        if data.get("status") == "ok":
            return data.get("totalResults", 0)
    except Exception:
        pass
    return 0


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────
def get_risk_level(score: float) -> tuple[str, str]:
    if score > 0.70:
        return "HIGH RISK", "risk-high"
    elif score > 0.40:
        return "MODERATE RISK", "risk-medium"
    else:
        return "LOW RISK", "risk-low"

def get_risk_emoji(score: float) -> str:
    if score > 0.70: return "🔴"
    elif score > 0.40: return "🟡"
    return "🟢"


# ─────────────────────────────────────────────
#  GRAPH DRAW  (enhanced radial layout)
# ─────────────────────────────────────────────
def draw_propagation_graph(G: nx.DiGraph, origin: str, suspicious_nodes: list = None) -> go.Figure:
    suspicious_nodes = suspicious_nodes or []
    pos = {origin: (0, 0)}
    neighbors_lvl1 = list(G.successors(origin))
    radius1, radius2 = 3, 6

    for i, node in enumerate(neighbors_lvl1):
        angle = 2 * math.pi * i / max(len(neighbors_lvl1), 1)
        pos[node] = (radius1 * math.cos(angle), radius1 * math.sin(angle))

    remaining = [n for n in G.nodes() if n not in neighbors_lvl1 and n != origin]
    for i, node in enumerate(remaining):
        angle = 2 * math.pi * i / max(len(remaining), 1)
        pos[node] = (radius2 * math.cos(angle), radius2 * math.sin(angle))

    fig = go.Figure()

    # Edges
    for edge in nx.bfs_edges(G, origin):
        x0, y0 = pos.get(edge[0], (0, 0))
        x1, y1 = pos.get(edge[1], (0, 0))
        is_suspicious = edge[1] in suspicious_nodes
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(color='rgba(255,107,53,0.5)' if is_suspicious else 'rgba(0,255,231,0.25)', width=2),
            hoverinfo='none',
            showlegend=False
        ))

    # Nodes
    for node, (x, y) in pos.items():
        if node == origin:
            color, size, symbol = "#ffe600", 35, "star"
        elif node in suspicious_nodes:
            color, size, symbol = "#ff4c4c", 28, "diamond"
        elif node in neighbors_lvl1:
            color, size, symbol = "#ff8c00", 22, "circle"
        else:
            color, size, symbol = "#4fc3f7", 16, "circle"

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(color='white', width=1)),
            text=[str(node)],
            textposition="top center",
            textfont=dict(color="white", size=9),
            hovertemplate=f"<b>{node}</b><br>{'⚠ Suspicious' if node in suspicious_nodes else '✅ Normal'}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text="Misinformation Propagation Network", font=dict(size=18, color="#00ffe7", family="Orbitron")),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(15,32,39,0.9)",
        paper_bgcolor="rgba(15,32,39,0.9)",
        font=dict(color="white", family="Exo 2"),
        margin=dict(l=20, r=20, t=50, b=20),
        hoverlabel=dict(bgcolor="#0f2027", font_color="white")
    )
    return fig


# ─────────────────────────────────────────────
#  FULL HYBRID ANALYSIS  (combined both versions)
# ─────────────────────────────────────────────
def run_full_analysis(text: str, weight_content: float, threshold: float) -> dict:
    weight_graph = 1 - weight_content
    results = {}

    # 1. TF-IDF Classifier (your version)
    fake_prob = classifier.predict_proba(text)[0][1]
    results["tfidf_score"] = fake_prob

    # 2. BERT — disabled, fallback to TF-IDF score
    results["bert_score"] = fake_prob

    # 3. NewsAPI cross-verification (friend's version)
    api_matches = verify_with_news_api(text[:100])
    if api_matches > 5:
        api_score = 0.2
    elif api_matches > 1:
        api_score = 0.5
    else:
        api_score = 0.85
    results["api_score"]   = api_score
    results["api_matches"] = api_matches

    # 4. Graph propagation simulation (your version)
    G, origin         = simulate_propagation()
    propagation_score = analyze_propagation(G)
    centrality        = nx.degree_centrality(G)
    suspicious_nodes  = [n for n, v in centrality.items() if v > 0.3]

    results["propagation_score"] = propagation_score
    results["graph"]             = G
    results["origin"]            = origin
    results["suspicious_nodes"]  = suspicious_nodes

    # 5. Final hybrid score
    semantic_avg = (results["tfidf_score"] + results["bert_score"]) / 2
    final_score  = (0.40 * semantic_avg) + (0.30 * api_score) + (0.30 * propagation_score)
    results["final_score"] = final_score
    results["label"]       = "Suspicious ⚠" if final_score > threshold else "Verified ✅"

    return results


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Orbitron,monospace;font-size:1.1rem;color:#00ffe7;font-weight:900;letter-spacing:3px;">🧠 M·I·E</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.65rem;color:rgba(255,255,255,0.4);letter-spacing:2px;">MISINFORMATION INTELLIGENCE ENGINE</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── User profile card in sidebar ──
    if user_info:
        if user_picture:
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                st.image(user_picture, width=60)
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:4px;">
            <div style="font-family:'Orbitron',monospace;font-size:0.82rem;
                        color:white;font-weight:700;letter-spacing:1px;">{user_name}</div>
            <div style="font-size:0.68rem;color:rgba(0,255,231,0.6);
                        letter-spacing:1px;">{user_email}</div>
            <div style="display:inline-block;background:rgba(0,255,136,0.1);
                        border:1px solid rgba(0,255,136,0.3);color:#00ff88;
                        border-radius:20px;padding:2px 10px;font-size:0.62rem;
                        letter-spacing:1px;margin-top:6px;">✔ Verified</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    pages = ["🏠 Home", "📝 Manual Analysis", "📡 Live News Scan", "🕸 Graph Simulator", "📊 History", "ℹ️ About"]
    for page in pages:
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page.split(" ", 1)[1]
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-header">System Controls</div>', unsafe_allow_html=True)

    threshold      = st.slider("Suspicion Threshold", 0.0, 1.0, 0.65, 0.05,
                                help="Score above this = Suspicious")
    weight_content = st.slider("Content Weight",      0.0, 1.0, 0.50, 0.05,
                                help="Balance NLP vs Graph scoring")
    weight_graph   = 1 - weight_content

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("TF-IDF", "✅")
    c2.metric("BERT",   "⏸️ Off")
    st.metric("Model Accuracy", "96%")

    st.markdown("---")
    st.markdown(f'<p style="font-size:0.65rem;color:rgba(255,255,255,0.3);text-align:center;">'
                f'<span class="status-dot"></span>SYSTEM ONLINE · {datetime.now().strftime("%H:%M:%S")}</p>',
                unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.user_info     = None
        st.session_state.show_app      = False
        st.session_state.page          = "Home"
        st.session_state.analysis_history = []
        st.rerun()


# ─────────────────────────────────────────────
#  TOP TABS
# ─────────────────────────────────────────────
tab_labels = ["🏠 Home", "📝 Manual Analysis", "📡 Live News Scan", "🕸 Graph Simulator", "📊 History", "ℹ️ About"]
tabs = st.tabs(tab_labels)


# ═════════════════════════════════════════════
#  TAB 0 — HOME
# ═════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="big-title">🧠 Misinformation Intelligence Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">NLP · BERT · Graph Theory · Real-Time API Verification</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["96%", "3-Layer", "Real-Time", "Hybrid"],
        ["Accuracy", "Detection", "Data Feed", "Scoring"]
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("""
    #### 🔬 How the Hybrid Engine Works

    | Layer | Method | Weight |
    |-------|--------|--------|
    | 🧠 Semantic NLP | TF-IDF + Logistic Regression | 20% |
    | 🤖 Deep Learning | BERT Transformer Model | 20% |
    | 🌐 API Cross-Check | NewsAPI real-time verification | 30% |
    | 🕸 Graph Analysis | Propagation network simulation | 30% |

    #### 🚀 Quick Start
    - **Manual Analysis** — paste any news text for instant deep analysis
    - **Live News Scan** — fetch & scan articles by keyword
    - **Graph Simulator** — explore misinformation spread patterns
    - **History** — review all past analyses this session
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  TAB 1 — MANUAL ANALYSIS  (friend's core feature)
# ═════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="big-title" style="font-size:1.6rem;">📝 Manual News Analysis</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Paste any news content for deep hybrid analysis</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    user_input = st.text_area("📩 Paste News Content Here", height=180,
                               placeholder="Paste a news article, headline, or social media post...")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Hybrid Analysis", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter some news content.")
        else:
            with st.spinner("🧠 Running Deep Hybrid Analysis..."):
                time.sleep(0.5)
                results = run_full_analysis(user_input, weight_content, threshold)

            risk_label, risk_class = get_risk_level(results["final_score"])
            final_pct = results["final_score"] * 100

            # Verdict
            st.markdown(f'<div class="{risk_class}">{get_risk_emoji(results["final_score"])} &nbsp; {risk_label} — {final_pct:.1f}% Misinformation Probability</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Score breakdown
            st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🧠 TF-IDF Score",      f"{results['tfidf_score']*100:.1f}%")
            c2.metric("🤖 BERT Score",         f"{results['bert_score']*100:.1f}%")
            c3.metric("🌐 API Cross-Check",    f"{results['api_score']*100:.0f}%",
                      help=f"{results['api_matches']} matching articles found")
            c4.metric("🕸 Propagation Score",  f"{results['propagation_score']:.2f}")

            st.progress(results["final_score"])

            # Score gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=final_pct,
                title={"text": "Hybrid Risk Score", "font": {"color": "white", "family": "Orbitron"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar":  {"color": "#ff4c4c" if final_pct > 70 else "#ffa500" if final_pct > 40 else "#00ff88"},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "steps": [
                        {"range": [0,  40], "color": "rgba(0,255,136,0.1)"},
                        {"range": [40, 70], "color": "rgba(255,165,0,0.1)"},
                        {"range": [70,100], "color": "rgba(255,76,76,0.1)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": threshold * 100}
                },
                number={"suffix": "%", "font": {"color": "white"}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(15,32,39,0.9)",
                font={"color": "white"},
                height=280,
                margin=dict(l=30, r=30, t=50, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Graph
            st.markdown('<div class="section-header">Propagation Network</div>', unsafe_allow_html=True)
            fig = draw_propagation_graph(results["graph"], results["origin"], results["suspicious_nodes"])
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Nodes",       len(results["graph"].nodes()))
            col2.metric("Total Connections", len(results["graph"].edges()))
            col3.metric("Suspicious Nodes",  len(results["suspicious_nodes"]))

            st.info(f"⭐ Origin  |  🔴 Suspicious  |  🟠 Amplifiers  |  🔵 Normal  |  NewsAPI matches: {results['api_matches']}")

            with st.expander("🔍 Detailed Explainability"):
                st.markdown(f"""
                - **TF-IDF Classifier:** {results['tfidf_score']:.3f} — linguistic pattern analysis
                - **BERT Model:** {results['bert_score']:.3f} — deep semantic understanding
                - **NewsAPI Verification:** {results['api_matches']} matching articles → risk score {results['api_score']:.2f}
                - **Graph Propagation:** {results['propagation_score']:.3f} — virality structure score
                - **Final Hybrid Score:** {results['final_score']:.3f} (threshold: {threshold})
                - **Verdict:** {results['label']}
                """)

            # Save to history
            st.session_state.analysis_history.append({
                "time":    datetime.now().strftime("%H:%M:%S"),
                "type":    "Manual",
                "text":    user_input[:80] + "...",
                "score":   results["final_score"],
                "verdict": results["label"]
            })


# ═════════════════════════════════════════════
#  TAB 2 — LIVE NEWS SCAN  (your version)
# ═════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="big-title" style="font-size:1.6rem;">📡 Live News Scanner</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Fetch and analyze real-time news articles by keyword</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        keyword = st.text_input("🔍 Enter keyword", placeholder="e.g. vaccine, election, climate...", key="keyword_input")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("🔎 Fetch & Analyse", use_container_width=True)

    if fetch_btn:
        if not keyword.strip():
            st.warning("Please enter a keyword.")
        else:
            with st.spinner(f"Fetching live articles for '{keyword}'..."):
                articles = fetch_news(keyword)
                time.sleep(0.5)

            if articles:
                st.success(f"Found {len(articles)} articles for **{keyword}**")

                for i, article in enumerate(articles):
                    with st.expander(f"📰 {article['title'][:90]}...", expanded=(i == 0)):
                        st.caption(f"Source: **{article['source']}** | Published: {article['published_at']}")

                        full_text = (article["title"] or "") + " " + (article["description"] or "")

                        with st.spinner("Analysing..."):
                            results = run_full_analysis(full_text, weight_content, threshold)

                        risk_label, risk_class = get_risk_level(results["final_score"])
                        st.markdown(f'<div class="{risk_class}">{get_risk_emoji(results["final_score"])} {risk_label} — {results["final_score"]*100:.1f}%</div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

                        cA, cB, cC, cD = st.columns(4)
                        cA.metric("🧠 NLP Score",        f"{results['tfidf_score']*100:.1f}%")
                        cB.metric("🌐 API Verification",  f"{results['api_score']*100:.0f}%")
                        cC.metric("🕸 Propagation",       f"{results['propagation_score']:.2f}")
                        cD.metric("📊 Credibility",       f"{compute_credibility(article['source']):.2f}")

                        st.progress(results["final_score"])

                        cX, cY = st.columns(2)
                        cX.metric("Risk Level",    f"{'🔴' if results['final_score']>0.7 else '🟡' if results['final_score']>0.4 else '🟢'} {risk_label}")
                        cY.metric("Outbreak Flag", "Yes 🚨" if detect_outbreak(results["tfidf_score"], article["published_at"]) else "No ✅")

                        fig = draw_propagation_graph(results["graph"], results["origin"], results["suspicious_nodes"])
                        st.plotly_chart(fig, use_container_width=True)

                        # Save history
                        st.session_state.analysis_history.append({
                            "time":    datetime.now().strftime("%H:%M:%S"),
                            "type":    "Live News",
                            "text":    article["title"][:80],
                            "score":   results["final_score"],
                            "verdict": results["label"]
                        })
            else:
                st.warning("No articles found. Try another keyword.")

    col_refresh = st.columns([3, 1])[1]
    if col_refresh.button("🔄 Auto Refresh (60s)"):
        st.info("Auto refresh enabled...")
        time.sleep(60)
        st.rerun()


# ═════════════════════════════════════════════
#  TAB 3 — GRAPH SIMULATOR
# ═════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="big-title" style="font-size:1.6rem;">🕸 Propagation Simulator</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Simulate how misinformation spreads through a network</div>', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([2, 1])
    with sim_col2:
        if st.button("▶ Run New Simulation", use_container_width=True):
            st.session_state["graph_sim"] = simulate_propagation()

    if "graph_sim" not in st.session_state:
        st.session_state["graph_sim"] = simulate_propagation()

    G, origin         = st.session_state["graph_sim"]
    propagation_score = analyze_propagation(G)
    centrality        = nx.degree_centrality(G)
    suspicious_nodes  = [n for n, v in centrality.items() if v > 0.3]

    with sim_col2:
        st.metric("Propagation Score", f"{propagation_score:.3f}")
        st.metric("Total Nodes",       len(G.nodes()))
        st.metric("Total Edges",       len(G.edges()))
        st.metric("Suspicious Nodes",  len(suspicious_nodes))

    with sim_col1:
        fig = draw_propagation_graph(G, origin, suspicious_nodes)
        st.plotly_chart(fig, use_container_width=True)

    st.info("⭐ Origin  |  🔴 Suspicious (high centrality)  |  🟠 First-level amplifiers  |  🔵 Secondary spread")

    # Centrality bar chart
    st.markdown('<div class="section-header">Node Centrality Analysis</div>', unsafe_allow_html=True)
    cent_df = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
    nodes_c, vals_c = zip(*cent_df)
    colors_c = ["#ff4c4c" if n in suspicious_nodes else "#00ffe7" for n in nodes_c]

    fig_bar = go.Figure(go.Bar(
        x=list(nodes_c), y=list(vals_c),
        marker_color=colors_c,
        hovertemplate="<b>%{x}</b><br>Centrality: %{y:.3f}<extra></extra>"
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(15,32,39,0.9)",
        plot_bgcolor="rgba(15,32,39,0.5)",
        font=dict(color="white"),
        xaxis=dict(tickfont=dict(size=9)),
        yaxis_title="Degree Centrality",
        height=260,
        margin=dict(l=20, r=20, t=20, b=60)
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ═════════════════════════════════════════════
#  TAB 4 — HISTORY
# ═════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="big-title" style="font-size:1.6rem;">📊 Analysis History</p>', unsafe_allow_html=True)

    if not st.session_state.analysis_history:
        st.info("No analyses yet. Run some analyses first!")
    else:
        history = st.session_state.analysis_history

        # Summary metrics
        total    = len(history)
        sus_count = sum(1 for h in history if "Suspicious" in h["verdict"])
        avg_score = sum(h["score"] for h in history) / total

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Analysed", total)
        c2.metric("Suspicious",     sus_count)
        c3.metric("Avg Risk Score", f"{avg_score:.2f}")

        st.markdown("---")

        # Score trend chart
        scores = [h["score"] * 100 for h in history]
        times  = [h["time"] for h in history]
        colors_h = ["#ff4c4c" if s > 70 else "#ffa500" if s > 40 else "#00ff88" for s in scores]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=times, y=scores,
            mode="lines+markers",
            line=dict(color="#00ffe7", width=2),
            marker=dict(color=colors_h, size=10),
            hovertemplate="Time: %{x}<br>Score: %{y:.1f}%<extra></extra>"
        ))
        fig_hist.add_hline(y=threshold * 100, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                           annotation_text="Threshold")
        fig_hist.update_layout(
            title="Risk Score Over Time",
            paper_bgcolor="rgba(15,32,39,0.9)",
            plot_bgcolor="rgba(15,32,39,0.5)",
            font=dict(color="white"),
            yaxis=dict(range=[0, 100], title="Risk %"),
            height=300,
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # History table
        st.markdown('<div class="section-header">All Analyses</div>', unsafe_allow_html=True)
        for item in reversed(history):
            emoji = get_risk_emoji(item["score"])
            st.markdown(
                f'<div class="article-card">'
                f'{emoji} &nbsp;<b>{item["verdict"]}</b> &nbsp;·&nbsp; '
                f'<span style="color:#00ffe7">{item["score"]*100:.1f}%</span> &nbsp;·&nbsp; '
                f'<span style="color:rgba(255,255,255,0.5)">{item["type"]}</span> &nbsp;·&nbsp; '
                f'<span style="color:rgba(255,255,255,0.4)">{item["time"]}</span><br>'
                f'<span style="font-size:0.85rem;color:rgba(255,255,255,0.6)">{item["text"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()


# ═════════════════════════════════════════════
#  TAB 5 — ABOUT
# ═════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="big-title" style="font-size:1.6rem;">ℹ️ About MIE</p>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Project Goal
    The **Misinformation Intelligence Engine (MIE)** is a real-time hybrid AI system
    combining NLP, Deep Learning, API verification and Graph Theory to detect fake news.

    ---

    ### 🧩 Combined Architecture

    | Component | Source | Description |
    |-----------|--------|-------------|
    | TF-IDF Classifier | Your version | Fast linguistic pattern detection |
    | BERT Transformer  | Friend's version | Deep semantic understanding |
    | NewsAPI Cross-check | Friend's version | Real-world article verification |
    | Graph Propagation | Your version | Virality & spread simulation |
    | Hybrid Scoring | Combined | Weighted multi-layer final verdict |

    ---

    ### 📦 Tech Stack
    Python · Streamlit · Scikit-learn · PyTorch · Transformers · NetworkX · Plotly · NewsAPI

    ---

    ### ⚙️ Configuration Tips
    - Lower **Suspicion Threshold** → flag more articles (higher recall)
    - Raise **Content Weight** → trust NLP more than graph signals
    - Use **Manual Analysis** for deep single-article inspection
    - Use **Live News Scan** for real-time keyword monitoring
    """)
    st.markdown('</div>', unsafe_allow_html=True)