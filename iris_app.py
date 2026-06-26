import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IRIS ML Explorer", page_icon="🌸", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #0f1117; color: #e8e8f0; }

    /* Form card */
    .form-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        max-width: 500px;
        margin: 2rem auto;
    }
    .form-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #c084fc;
        margin-bottom: 0.25rem;
    }
    .form-sub {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #13151f !important;
        border-right: 1px solid #2a2d3e;
    }
    [data-testid="stSidebar"] * { color: #e8e8f0 !important; }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        flex: 1;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #c084fc; }

    /* Prediction badge */
    .pred-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #c084fc);
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        margin: 0.5rem 0 1rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c084fc;
        margin: 1.5rem 0 0.75rem;
        border-left: 3px solid #7c3aed;
        padding-left: 0.75rem;
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #7c3aed, #c084fc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Load data & train model (cached) ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("IRIS.csv")

@st.cache_resource
def train_model(data):
    X = data.drop("species", axis=1)
    y = data["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    return rf, acc, X_test, y_test

# ── Session state ─────────────────────────────────────────────────────────────
if "unlocked" not in st.session_state:
    st.session_state.unlocked = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ══════════════════════════════════════════════════════════════════════════════
# GATE: Login / Info Form
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.unlocked:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="form-card">
            <div class="form-title">🌸 IRIS ML Explorer</div>
            <div class="form-sub">Fill in your details to access the dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("gate_form"):
            name    = st.text_input("Your Name", placeholder="e.g. Aditya Shah")
            purpose = st.selectbox("I'm here to…", [
                "Explore the data & charts",
                "Test the ML prediction model",
                "Both — show me everything",
            ])
            agree = st.checkbox("I understand this is a demo ML application")
            submitted = st.form_submit_button("Enter Dashboard →")

        if submitted:
            if not name.strip():
                st.error("Please enter your name.")
            elif not agree:
                st.error("Please accept the terms to continue.")
            else:
                st.session_state.unlocked  = True
                st.session_state.username  = name.strip()
                st.session_state.purpose   = purpose
                st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD (post-login)
# ══════════════════════════════════════════════════════════════════════════════
try:
    data = load_data()
except FileNotFoundError:
    st.error("❌ `IRIS.csv` not found. Place it in the same folder as this script.")
    st.stop()

model, accuracy, X_test, y_test = train_model(data)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👋 Hello, {st.session_state.username}!")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Charts", "🤖 Prediction"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**Model Accuracy**")
    st.markdown(f"<span style='font-size:1.6rem;font-weight:700;color:#c084fc'>{accuracy*100:.1f}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Dataset rows:** {len(data)}")
    st.markdown(f"**Features:** sepal & petal dimensions")
    st.markdown("---")
    if st.button("🚪 Log out"):
        st.session_state.unlocked = False
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Charts
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Charts":
    st.markdown('<div class="section-title">Data Visualisations</div>', unsafe_allow_html=True)

    chart_choice = st.selectbox(
        "Choose a chart",
        ["Scatter – Species vs Sepal Length",
         "Bar – Species vs Petal Length",
         "Histogram – Species Distribution"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")
    ax.tick_params(colors="#9ca3af")
    ax.xaxis.label.set_color("#9ca3af")
    ax.yaxis.label.set_color("#9ca3af")
    ax.title.set_color("#c084fc")

    if chart_choice == "Scatter – Species vs Sepal Length":
        scatter = ax.scatter(data["species"], data["sepal_length"],
                             c=pd.factorize(data["species"])[0],
                             cmap="viridis", alpha=0.7, edgecolors="none", s=60)
        ax.set_xlabel("Species")
        ax.set_ylabel("Sepal Length (cm)")
        ax.set_title("Species vs Sepal Length")
        plt.colorbar(scatter, ax=ax, label="Species index")

    elif chart_choice == "Bar – Species vs Petal Length":
        grouped = data.groupby("species")["petal_length"].mean()
        colors  = ["#7c3aed", "#c084fc", "#e879f9"]
        bars    = ax.bar(grouped.index, grouped.values, color=colors, edgecolor="#2a2d3e")
        ax.set_xlabel("Species")
        ax.set_ylabel("Avg Petal Length (cm)")
        ax.set_title("Average Petal Length per Species")
        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", color="#e8e8f0", fontsize=9)

    elif chart_choice == "Histogram – Species Distribution":
        species_codes, species_labels = pd.factorize(data["species"])
        ax.hist(species_codes, bins=10, color="#c084fc", edgecolor="#7c3aed", alpha=0.85)
        ax.set_xticks(range(len(species_labels)))
        ax.set_xticklabels(species_labels)
        ax.set_xlabel("Species")
        ax.set_ylabel("Count")
        ax.set_title("Species Distribution (Histogram)")

    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("📋 Raw dataset"):
        st.dataframe(data, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Prediction":
    st.markdown('<div class="section-title">Predict Iris Species</div>', unsafe_allow_html=True)
    st.markdown("Enter the four flower measurements and the Random Forest model will predict the species.")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=20.0, value=5.1, step=0.1)
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=20.0, value=1.4, step=0.1)
    with col2:
        sepal_width  = st.number_input("Sepal Width (cm)",  min_value=0.0, max_value=20.0, value=3.5, step=0.1)
        petal_width  = st.number_input("Petal Width (cm)",  min_value=0.0, max_value=20.0, value=0.2, step=0.1)

    if st.button("Predict Species 🔍"):
        input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        classes    = model.classes_

        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-badge">🌸 {prediction}</div>', unsafe_allow_html=True)

        st.markdown("**Confidence scores**")
        proba_df = pd.DataFrame({"Species": classes, "Probability": proba}).sort_values("Probability", ascending=False)
        for _, row in proba_df.iterrows():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(float(row["Probability"]))
            with col_b:
                st.markdown(f"`{row['Species']}` **{row['Probability']*100:.1f}%**")