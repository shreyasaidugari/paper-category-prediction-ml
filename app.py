import re
import joblib
import numpy as np
import streamlit as st


# ---------------------------------
# Load saved model files
# ---------------------------------
model = joblib.load("best_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

available_categories = list(label_encoder.classes_)


# ---------------------------------
# Text cleaning
# ---------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------
# Simple check for research-style text
# ---------------------------------
def research_style_score(text):
    research_words = [
        "paper", "research", "study", "proposed", "method", "model",
        "approach", "analysis", "result", "experiment", "dataset",
        "algorithm", "performance", "prediction", "classification",
        "training", "evaluation", "framework", "system", "learning",
        "network", "accuracy", "abstract", "technical"
    ]
    words = text.split()
    score = sum(1 for word in words if word in research_words)
    return score


# ---------------------------------
# Streamlit page config
# ---------------------------------
st.set_page_config(
    page_title="Research Paper Type Prediction",
    page_icon="📘",
    layout="wide"
)


# ---------------------------------
# Styling
# ---------------------------------
background_image = "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=1600&q=80"

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(rgba(8, 15, 28, 0.86), rgba(8, 15, 28, 0.92)),
            url("{background_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main .block-container {{
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    .hero {{
        background: linear-gradient(135deg, rgba(17, 94, 89, 0.78), rgba(30, 64, 175, 0.72));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 34px;
        margin-bottom: 24px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.28);
        backdrop-filter: blur(12px);
    }}

    .hero-title {{
        color: white;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }}

    .hero-subtitle {{
        color: #dbe7f3;
        font-size: 17px;
        line-height: 1.7;
        max-width: 850px;
    }}

    .panel {{
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        backdrop-filter: blur(10px);
    }}

    .metric-card {{
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px;
        margin-top: 12px;
        color: white;
    }}

    .result-card {{
        background: linear-gradient(135deg, #0f766e, #155e75);
        color: white;
        border-radius: 18px;
        padding: 20px 24px;
        margin-top: 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    }}

    .result-title {{
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 6px;
    }}

    .result-value {{
        font-size: 28px;
        font-weight: 700;
    }}

    .warning-card {{
        background: linear-gradient(135deg, #b45309, #92400e);
        color: white;
        border-radius: 18px;
        padding: 20px 24px;
        margin-top: 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    }}

    .side-note {{
        color: #dbe7f3;
        font-size: 14px;
        line-height: 1.7;
    }}

    .category-box {{
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 10px 12px;
        color: white;
        margin-bottom: 8px;
    }}

    textarea {{
        background-color: rgba(255,255,255,0.96) !important;
        color: #111827 !important;
        border-radius: 14px !important;
    }}

    .stButton > button {{
        width: 100%;
        border: none;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        font-size: 16px;
        font-weight: 600;
        color: white;
        background: linear-gradient(135deg, #f59e0b, #d97706);
    }}

    .stButton > button:hover {{
        color: white;
        background: linear-gradient(135deg, #d97706, #b45309);
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# Header
# ---------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Research Paper Type Prediction</div>
        <div class="hero-subtitle">
            This application predicts the category of a research paper using the labels available in the trained model.
            If the entered text does not look like research paper content, the system marks it as out of scope instead of showing a misleading prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# Layout
# ---------------------------------
left_col, right_col = st.columns([2.1, 1], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Input Text")
    user_input = st.text_area(
        "Paste the abstract, summary, or technical paper text below",
        height=300,
        placeholder="Example: In this paper, we propose a deep learning based framework for text classification using transformer representations..."
    )
    predict_button = st.button("Predict Research Paper Type")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Scope")
    st.markdown(
        """
        <div class="side-note">
        The system can only predict from categories used during training.
        If text is unrelated to research paper style content, it will be flagged as out of scope.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Available Categories")

    for label in available_categories:
        st.markdown(f'<div class="category-box">{label}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------
# Prediction logic
# ---------------------------------
if predict_button:
    if user_input.strip() == "":
        st.warning("Please enter some text before prediction.")
    else:
        cleaned_input = clean_text(user_input)
        word_count = len(cleaned_input.split())
        style_score = research_style_score(cleaned_input)

        input_tfidf = tfidf.transform([cleaned_input])
        prediction = model.predict(input_tfidf)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        confidence_score = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_tfidf)[0]
            confidence_score = float(np.max(probabilities))

        out_of_scope = False

        if word_count < 15:
            out_of_scope = True

        if style_score < 2:
            out_of_scope = True

        if confidence_score is not None and confidence_score < 0.60:
            out_of_scope = True

        if out_of_scope:
            st.markdown(
                """
                <div class="warning-card">
                    <div class="result-title">Status</div>
                    <div class="result-value">Out of Research Paper Scope</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "The entered text does not strongly match research paper style content or the trained paper categories."
            )

            if confidence_score is not None:
                st.write(f"Closest predicted category: `{predicted_label}`")
                st.write(f"Confidence score: `{confidence_score:.2f}`")

        else:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Predicted Research Paper Type</div>
                    <div class="result-value">{predicted_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <b>Word Count</b><br>{word_count}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_b:
                if confidence_score is not None:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <b>Confidence</b><br>{confidence_score:.2f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.success("Prediction completed successfully.")
