import streamlit as st
import google.generativeai as genai
import base64
import pandas as pd
import numpy as np
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import tempfile
import os
from gtts import gTTS
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Gemini API Setup
# ---------------------------
# Load centralized config (expects config.py in project root)
# config should expose `config` with attributes PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME
# and a helper `configure_genai_if_available(genai_module, cfg)` that returns True if configured.
try:
    from config import config, configure_genai_if_available  # type: ignore
except Exception:
    config = None
    configure_genai_if_available = None

configured = False
if configure_genai_if_available is not None:
    try:
        configured = configure_genai_if_available(genai, config)
    except Exception:
        configured = False

if not configured:
    st.warning(
        "Gemini API key not found or genai configuration failed. Gemini-powered features will be disabled. "
        "Set GEMINI_API_KEY in Streamlit Secrets or environment variables."
    )

# Use model names from config if available, otherwise fallback to defaults
if config is not None:
    PRIMARY_MODEL_NAME = getattr(config, "PRIMARY_MODEL_NAME", "gemini-2.5-flash")
    FALLBACK_MODEL_NAME = getattr(config, "FALLBACK_MODEL_NAME", "gemini-2.5-pro")
else:
    PRIMARY_MODEL_NAME = "gemini-2.5-flash"
    FALLBACK_MODEL_NAME = "gemini-2.5-pro"

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="MediAdvisor: AI Powered SmartHealth Suite", layout="wide")

# ---------------------------
# Global CSS (Including Fixed Sidebar and Title Styling)
# ---------------------------
import streamlit as st
import base64

# Load sidebar image
with open("sidebar.png", "rb") as image_file:
    sidebar_image = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <style>
        /* Sidebar with background image */
        [data-testid="stSidebar"] {{
            background-image: url("data:image/png;base64,{sidebar_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #0d47a1 !important;
        }}

        /* Make all sidebar text bold and blue */
        [data-testid="stSidebar"] * {{
            color: #0d47a1 !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSidebar"] .css-1d391kg {{
            color: #0d47a1; /* Fixed title color for "MediAdvisor Features" */
            font-weight: 900;
        }}

        [data-testid="stSidebar"] .css-1a32cs0,
        [data-testid="stSidebar"] .stSelectbox > div {{
            color: #0d47a1 !important; /* Sidebar text and selectbox font color */
            font-weight: 700 !important;
        }}

        .stApp {{
            background: linear-gradient(135deg, #f0f6ff, #e3f2fd, #f0f6ff);
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        .website-title {{
            text-align: center;
            font-size: 2.8rem;
            font-weight: 900;
            color: #0d47a1;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        }}

        .subtitle {{
            text-align: center;
            font-size: 1.2rem;
            color: #333;
            margin-bottom: 30px;
        }}

        h1, h2, h3 {{
            color: #0d47a1 !important;
            font-weight: 800 !important;
        }}

        h4, h5, h6 {{
            color: #444 !important;
            font-weight: 600 !important;
        }}

        input[type="text"] {{
            border: 2px solid #1977f3 !important;
            border-radius: 8px !important;
            padding: 10px !important;
            font-size: 1rem !important;
        }}

        .card {{
            background: #fff;
            border-radius: 15px;
            box-shadow: 0 8px 30px rgba(0,0,0,.1);
            padding-bottom: 1.7rem;
            border: 1.5px solid #f5f5f5;
            margin-bottom: 37px;
            transition: transform 0.2s;
        }}

        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0,0,0,.15);
        }}

        .card img {{
            border-radius: 15px 15px 0 0;
            width: 100%;
            height: 170px;
            object-fit: cover;
            margin-bottom: 0;
        }}

        .card-title {{
            font-size: 1.15rem;
            font-weight: 700;
            margin: 20px 0 10px 0;
            color: #1977f3;
            padding-left: 19px;
        }}

        .card-button {{
            margin-left: 19px;
            padding: 8px 21px;
            border-radius: 7px;
            color: white;
            background: #1977f3;
            border: none;
            font-weight: 500;
            font-size: 1rem;
            cursor: pointer;
        }}

        .card-button:hover {{
            background: #1558a0;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar for Feature Selection
# ---------------------------
st.sidebar.title("MediAdvisor Features")
feature = st.sidebar.selectbox(
    "Select Feature",
    [
        "Disease Information",
        "Healthcare Chatbot",
        "Treatment Cost Estimator",
        "Hospital Recommendation System",
        "Motivational Buddy"
    ]
)

# ---------------------------
# Disease Information Feature (from Disease.py)
# ---------------------------
if feature == "Disease Information":
    # Article Data
    ARTICLES = [
        {"title": "Asthma", "image_url": "images/asthma.jpg", "brief": "Asthma is a chronic condition that narrows and inflames airways, making breathing difficult."},
        {"title": "Fungal Nail Infections (Tinea Unguium)", "image_url": "images/fungal_nail.jpg", "brief": "Fungal nail infections cause discoloration and thickening of nails, mostly toenails."},
        {"title": "Diabetes Mellitus", "image_url": "images/diabetes.jpg", "brief": "Diabetes is a metabolic disorder leading to high blood sugar levels over prolonged periods."},
        {"title": "Hypertension (High Blood Pressure)", "image_url": "images/hypertension.png", "brief": "Hypertension occurs when blood pressure against artery walls is consistently too high."},
        {"title": "Coronary Artery Disease", "image_url": "images/coronary_artery.png", "brief": "Coronary artery disease is caused by plaque buildup in arteries supplying blood to the heart."},
        {"title": "Chronic Kidney Disease", "image_url": "images/chronic_kidney.jpeg", "brief": "CKD is a gradual loss of kidney function, impairing the ability to filter waste from blood."},
        {"title": "Stroke", "image_url": "images/stroke.jpeg", "brief": "Stroke occurs when blood flow to part of the brain is interrupted, leading to cell death."},
        {"title": "Chronic Obstructive Pulmonary Disease (COPD)", "image_url": "images/copd.jpeg", "brief": "COPD is a progressive lung disease causing obstructed airflow, chronic cough, and breathlessness."},
        {"title": "Osteoporosis", "image_url": "images/osteoporosis.jpeg", "brief": "Osteoporosis weakens bones, making them fragile and more likely to fracture."},
        {"title": "Arthritis", "image_url": "images/arthritis.png", "brief": "Arthritis refers to joint inflammation, causing pain, stiffness, and reduced mobility."},
        {"title": "Alzheimer’s Disease", "image_url": "images/alzheimers.jpg", "brief": "Alzheimer’s is a progressive neurological disorder that causes memory loss and cognitive decline."},
        {"title": "Parkinson’s Disease", "image_url": "images/parkinsons.jpg", "brief": "Parkinson’s is a nervous system disorder affecting movement, often causing tremors."},
        {"title": "Tuberculosis (TB)", "image_url": "images/tuberculosis.jpg", "brief": "TB is a bacterial infection that mainly affects the lungs but can spread to other organs."},
        {"title": "COVID-19", "image_url": "images/covid19.jpg", "brief": "COVID-19 is a contagious respiratory illness caused by the SARS-CoV-2 virus."},
        {"title": "Cancer", "image_url": "images/cancer.jpg", "brief": "Cancer is a group of diseases characterized by uncontrolled cell growth and spread to other parts of the body."},
        {"title": "Cardiovascular Disease (Heart Disease)", "image_url": "images/heart_disease.jpeg", "brief": "Cardiovascular disease includes conditions that affect the heart and blood vessels, often leading to heart attacks and strokes."},
        {"title": "Depression & Anxiety Disorders", "image_url": "images/depression_anxiety.webp", "brief": "Depression and anxiety disorders are common mental health conditions affecting mood, emotions, and daily functioning."},
        {"title": "Psoriasis", "image_url": "images/psoriasis.webp", "brief": "Psoriasis is a chronic autoimmune condition that causes rapid skin cell buildup, leading to scaling and inflammation."}
    ]

    # Helper Functions
    def generate_full_info(title):
        if title == "Asthma Overview":
            prompt = (
                f"You are an expert healthcare LLM. Write a highly detailed, structured article for patients about '{title}'. "
                "Include comprehensive headings for: Definition, Causes, Symptoms, Risk Factors, Diagnosis, Treatment, Prevention, "
                "and When to See a Doctor. Ensure the content is thorough, easy to understand, medically accurate, and at least 500 words long. "
                "Provide detailed explanations under each heading with examples and practical advice."
            )
        else:
            prompt = (
                f"You are an expert healthcare LLM. Write a detailed, structured article for patients about '{title}'. "
                "Include clear headings for: Definition, Causes, Symptoms, Risk Factors, Diagnosis, Treatment, Prevention, "
                "and When to See a Doctor. Make it easy to understand but medically accurate."
            )
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return "Detailed information not available."

    def generate_answer(user_q, full_text):
        prompt = (
            "You are a helpful health AI. Use this article content to answer the user's question concisely.\n"
            f"Article:\n{full_text}\nQuestion: {user_q}\nAnswer:"
        )
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return "Sorry, no answer found."

    @st.cache_data(show_spinner=True)
    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    def compute_perplexity(text):
        try:
            model_name = "gpt2"
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.eval()
            tokens = tokenizer.encode(text)
            chunk_size = 1024
            chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
            perplexities = []
            for chunk in chunks:
                input_ids = torch.tensor([chunk])
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    perplexities.append(torch.exp(loss).item())
            avg_perplexity = sum(perplexities) / len(perplexities)
            return round(avg_perplexity, 2)
        except Exception as e:
            print("Error computing perplexity:", e)
            return "N/A"

    # Session State
    if "page" not in st.session_state:
        st.session_state.page = "main"

    # Main Page (Cards)
    if st.session_state.page == "main":
        st.markdown('<div class="website-title">🩺 MediAdvisor : AI Powered SmartHealth Assistant ⛨ </div>', unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>⚕️ One Stop Solution for all your Health Needs ⚕️</p>", unsafe_allow_html=True)

        for row_start in range(0, len(ARTICLES), 3):
            cols = st.columns(3)
            for i, article in enumerate(ARTICLES[row_start:row_start+3]):
                with cols[i]:
                    base64_image = get_base64_image(article["image_url"])
                    st.markdown(
                        f"""
                        <div class="card">
                            <img src="data:image/jpeg;base64,{base64_image}" />
                            <div class="card-title">{article['title']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button("Read More", key=f"btn_{row_start+i}"):
                        st.session_state.page = "details"
                        st.session_state.selected_article_idx = row_start + i

    # Details Page
    elif st.session_state.page == "details":
        idx = st.session_state.selected_article_idx
        article = ARTICLES[idx]

        if st.button("⬅ Back to Homepage"):
            st.session_state.page = "main"

        st.markdown(f"<h2 style='text-align:center;'>{article['title']}</h2>", unsafe_allow_html=True)
        st.image(article["image_url"], use_container_width=True)
        st.write(article["brief"])

        if "full_info" not in st.session_state or st.session_state.full_info.get("title") != article["title"]:
            with st.spinner("🔍 Generating detailed article..."):
                st.session_state.full_info = {"title": article["title"], "text": generate_full_info(article["title"])}
        st.markdown(st.session_state.full_info["text"], unsafe_allow_html=True)

        with st.spinner("📊 Calculating Perplexity..."):
            perplexity_score = compute_perplexity(st.session_state.full_info["text"])
        st.info(f"Perplexity Score of this article: **{perplexity_score}**")

        q = st.text_input("💬 Ask something about this condition (symptoms, risks, management, etc.):")
        if q:
            with st.spinner("✍ Generating answer..."):
                answer = generate_answer(q, st.session_state.full_info["text"])
            st.success(answer)

    # Footer Section
    footer_html = """
    <div class="footer">
        <div class="footer-logo">.........</div>
        <div class="footer-features">
            <div class="footer-feature"><div class="footer-icon">✅</div><div class="footer-text">AI-Powered Health Insights</div></div>
            <div class="footer-feature"><div class="footer-icon">🕒</div><div class="footer-text">24/7 Availability</div></div>
            <div class="footer-feature"><div class="footer-icon">🌍</div><div class="footer-text">Multilingual Support</div></div>
            <div class="footer-feature"><div class="footer-icon">🔒</div><div class="footer-text">Data Privacy & Security</div></div>
        </div>
        <p class="footer-disclaimer">⚠️ Disclaimer: MediAdvisor provides AI-generated educational content and does not replace professional medical advice. Please consult a doctor for diagnosis or treatment.</p>
        <p class="footer-bottom">© 2025 MediAdvisor </p>
    </div>
    <style>
    .footer {margin-top: 50px; padding: 40px 20px 20px 20px; background: #f5f5f5; border-radius: 10px; text-align: center;}
    .footer-logo {font-size: 1.6rem; font-weight: 700; color: #0d47a1; margin-bottom: 25px;}
    .footer-features {display: flex; justify-content: center; gap: 60px; margin-bottom: 25px; flex-wrap: wrap;}
    .footer-feature {display: flex; flex-direction: column; align-items: center; max-width: 150px;}
    .footer-icon {font-size: 1.8rem; margin-bottom: 8px;}
    .footer-text {font-size: 1rem; font-weight: 500; color: #333;}
    .footer-disclaimer {font-size: 0.9rem; color: #555; margin-top: 15px;}
    .footer-bottom {font-size: 0.8rem; margin-top: 10px; color: #666;}
    </style>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# ---------------------------
# Healthcare Chatbot Feature (from healthcare_chatbot_agent.py)
# ---------------------------
elif feature == "Healthcare Chatbot":
    # Text-to-Speech (gTTS)
    def text_to_speech(text, lang_code="en", filename="response.mp3"):
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filename)
        return filename

    # Speech-to-Text
    def voice_input_to_text():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Speak now...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand your voice."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition."

    # Healthcare Chatbot Function
    def healthcare_chatbot_answer(question, lang_code, selected_lang):
        prompt = f"""
        You are an AI Healthcare Assistant.
        Answer the following user question in {selected_lang} language
        (use simple, clear, empathetic words).
        Format your response as bullet points for easier reading.
        Limit it to 4-6 points maximum.
        Always remind them to consult a doctor for serious or emergency issues.
        User Question: {question}
        """
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    # Feature-specific CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #e6f2f2;
            min-height: 100vh;
            padding: 4rem 4rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .block-container {
            background-color: white;
            border-radius: 16px;
            padding: 3rem 3rem 5rem 3rem;
            max-width: 700px;
            width: 100%;
            margin: 1rem auto;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 1.5rem;
        }
        .stSelectbox > div {
            border-radius: 8px;
        }
        .stTextInput > div > div > input {
            border: 2px solid #1abc9c;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #1abc9c;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.6rem 1.3rem;
            margin-top: 20px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #16a085;
            cursor: pointer;
        }
        .stAlert > div {
            background-color: #ecf0f1;
            color: #2c3e50;
        }
        .stSubheader {
            color: #1abc9c;
        }
        audio {
            margin-top: 10px;
        }
        hr {
            border: none;
            height: 1px;
            background-color: #ddd;
            margin-top: 3rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # App UI Layout
    st.title("🤖 AI Healthcare Assistant")
    st.write("Ask any healthcare-related question. You can type or use voice input 🎤.")

    # Language selection
    lang_map = {"English": "en", "Hindi": "hi", "Gujarati": "gu"}
    selected_lang = st.selectbox("🌐 Select Language:", list(lang_map.keys()))
    lang_code = lang_map[selected_lang]

    # User input (text)
    user_question = st.text_input("💬 Enter your healthcare-related question:")

    # Voice input button
    if st.button("🎤 Use Voice Input"):
        user_question = voice_input_to_text()
        st.success(f"🎤 You said: {user_question}")

    # Generate Answer
    if user_question:
        with st.spinner("Generating answer..."):
            answer = healthcare_chatbot_answer(user_question, lang_code, selected_lang)
        st.subheader("Answer:")
        st.markdown(answer)

        # Convert answer to speech
        if st.button("🔊 Hear Answer"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                filename = temp_file.name
                text_to_speech(answer, lang_code, filename)
                st.audio(filename, format="audio/mp3")
            if "old_audio" in st.session_state and os.path.exists(st.session_state["old_audio"]):
                try:
                    os.remove(st.session_state["old_audio"])
                except PermissionError:
                    pass
            st.session_state["old_audio"] = filename

# ---------------------------
# Disease Cost Estimator Feature (from disease_estimator.py)
# ---------------------------
elif feature == "Treatment Cost Estimator":
    # LLM Fallback
    def llm_fallback_estimate(disease_name):
        prompt = f"""
        Estimate the average treatment cost and typical cost range for the disease '{disease_name}' in India.
        Respond strictly as numbers in this format:
        Mean cost: XXXX
        Low cost: XXXX
        High cost: XXXX
        """
        try:
            model = genai.GenerativeModel(PRIMARY_MODEL_NAME)
            response = model.generate_content(prompt)
            text = response.text.strip()
            numbers = re.findall(r'\d+', text.replace(',', ''))
            if len(numbers) >= 3:
                mean_cost, low, high = map(int, numbers[:3])
                low = max(low, int(mean_cost * 0.7))
                high = min(high, int(mean_cost * 1.3))
                return {"mean_cost": mean_cost, "low": low, "high": high}
            else:
                st.warning("LLM returned unexpected format.")
                return None
        except Exception as e:
            st.error(f"LLM fallback failed: {e}")
            return None

    # Load & Clean Dataset
    @st.cache_data
    def load_and_clean(path):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        rename_map = {
            "DiseaseName": "disease",
            "HospitalClass": "hospital_class",
            "LengthOfStay_days": "stay_days",
            "RoomCharges": "room_charges",
            "MedicineCharges": "medicine_charges",
            "OtherCharges": "other_charges",
            "Insurance_Yes": "insurance",
            "PayerType": "payer",
            "InsurerPayment": "insurer_payment",
            "OutOfPocket": "out_of_pocket",
            "TotalHospitalReceived": "cost"
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        required_cols = ["disease", "cost", "stay_days", "hospital_class", "room_charges", "medicine_charges",
                        "other_charges", "insurance", "payer", "insurer_payment", "out_of_pocket"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        for col in ["stay_days", "room_charges", "medicine_charges", "other_charges",
                    "insurer_payment", "out_of_pocket", "cost"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df = df.dropna(subset=["disease", "cost"]).reset_index(drop=True)

        def identify_low_entry_diseases(df, threshold=5):
            counts = df['disease'].value_counts()
            return counts[counts < threshold].index.tolist()

        def augment_low_entry_diseases(df, low_entry_diseases, aug_factor=10, noise_std=0.15):
            augmented_rows = []
            for disease in low_entry_diseases:
                disease_rows = df[df['disease'] == disease]
                if len(disease_rows) > 0:
                    for _, row in disease_rows.iterrows():
                        for _ in range(aug_factor - 1):
                            new_row = row.copy()
                            numeric_cols = ['stay_days', 'room_charges', 'medicine_charges', 'other_charges',
                                          'insurer_payment', 'out_of_pocket', 'cost']
                            for col in numeric_cols:
                                if col in new_row and pd.notna(new_row[col]):
                                    noise = np.random.normal(0, noise_std * abs(new_row[col]))
                                    new_row[col] = max(0, new_row[col] + noise)
                            if 'hospital_class' in new_row:
                                new_row['hospital_class'] = np.random.choice(df['hospital_class'].dropna().unique())
                            if 'insurance' in new_row:
                                new_row['insurance'] = np.random.choice(['Yes', 'No'], p=[0.7, 0.3])
                            if 'payer' in new_row:
                                new_row['payer'] = np.random.choice(df['payer'].dropna().unique())
                            if all(c in new_row for c in ['insurer_payment', 'out_of_pocket']):
                                new_row['cost'] = new_row['insurer_payment'] + new_row['out_of_pocket']
                            augmented_rows.append(new_row)
            if augmented_rows:
                aug_df = pd.DataFrame(augmented_rows)
                df = pd.concat([df, aug_df], ignore_index=True)
            return df

        low_entry_diseases = identify_low_entry_diseases(df)
        if low_entry_diseases:
            df = augment_low_entry_diseases(df, low_entry_diseases)
        return df

    # Unified Model Training
    @st.cache_resource
    def train_and_select_best_model(df, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = []
        candidate_cols = ["disease", "hospital_class", "stay_days",
                         "room_charges", "medicine_charges", "other_charges",
                         "insurance", "payer", "insurer_payment", "out_of_pocket"]
        feature_cols = [c for c in candidate_cols if c in df.columns and c not in exclude_cols]
        if not feature_cols:
            raise ValueError("No feature columns available for model.")
        X = df[feature_cols].copy()
        y = np.log1p(df["cost"].values)
        cat_cols = [c for c in feature_cols if X[c].dtype == "object" or X[c].dtype.name == "category"]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        transformers = []
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
        if num_cols:
            transformers.append(("num", "passthrough", num_cols))
        preprocessor = ColumnTransformer(transformers, remainder="drop")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        models = {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(alpha=0.01, random_state=42),
            "DecisionTree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                                 max_features="sqrt", random_state=42),
            "SVR": SVR(kernel='rbf', C=100, epsilon=0.1)
        }
        results = {}
        best_model, best_r2, best_name = None, -np.inf, None
        for name, base_model in models.items():
            if name == "SVR":
                transformers_svr = []
                if cat_cols:
                    transformers_svr.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
                if num_cols:
                    transformers_svr.append(("num", StandardScaler(), num_cols))
                preprocessor_svr = ColumnTransformer(transformers_svr, remainder="drop")
                pipe = Pipeline([("preprocessor", preprocessor_svr), ("regressor", base_model)])
            else:
                pipe = Pipeline([("preprocessor", preprocessor), ("regressor", base_model)])
            pipe.fit(X_train, y_train)
            y_pred_log = pipe.predict(X_test)
            r2 = r2_score(y_test, y_pred_log)
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred_log)
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            results[name] = {"r2": r2, "mae": mae, "rmse": rmse, "model": pipe}
            if r2 > best_r2:
                best_r2 = r2
                best_model = pipe
                best_name = name
        return best_model, results, best_name, feature_cols

    # Streamlit UI
    st.markdown("""
    <style>
        .stApp {
            background-color: #f1edff;
        }
        h1 {
            font-size: 32px !important;
            font-weight: 700 !important;
            color: #0047AB !important;
        }
        h2 {
            font-size: 24px !important;
            font-weight: 600 !important;
            color: #1a237e !important;
        }
        h3, h4, h5, h6, p, div, span {
            color: #0047AB !important;
            font-weight: bold;
        }
        .stTextInput > div > div > input {
            background-color: white !important;
            color: #000000 !important;
            border: 1px solid #0047AB !important;
            border-radius: 6px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Treatment Cost Estimator Buddy")

    # Load dataset
    if "df" not in st.session_state:
        try:
            st.session_state.df = load_and_clean("final_synthetic_healthcare_costs.csv")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    # Train model
    if "best_model" not in st.session_state:
        with st.spinner("Training models..."):
            try:
                st.session_state.best_model, st.session_state.results, st.session_state.best_name, st.session_state.feature_cols = train_and_select_best_model(st.session_state.df)
            except Exception as e:
                st.error(f"Model training failed: {e}")
                st.stop()

    # Input
    disease_input = st.text_input("🔍 Enter disease name:", "")
    predict_button = st.button("Predict Treatment Cost Range")

    # Helper to build representative row
    def build_representative_row(rows_df, feature_cols):
        rep = {}
        for c in feature_cols:
            if c not in rows_df.columns:
                rep[c] = st.session_state.df[c].median() if c in st.session_state.df.columns and pd.api.types.is_numeric_dtype(st.session_state.df[c]) else ""
                continue
            if rows_df[c].dtype == "object" or rows_df[c].dtype.name == "category":
                mode_vals = rows_df[c].mode()
                rep[c] = mode_vals.iloc[0] if not mode_vals.empty else ""
            else:
                med = rows_df[c].median()
                if pd.isna(med):
                    med = st.session_state.df[c].median() if c in st.session_state.df.columns else 0
                rep[c] = med
        return pd.DataFrame([rep])[feature_cols] if feature_cols else pd.DataFrame()

    # Prediction flow
    if predict_button:
        if not disease_input.strip():
            st.warning("Please enter a disease name.")
        else:
            disease_norm = disease_input.strip().lower()
            disease_rows = st.session_state.df[st.session_state.df["disease"].str.lower() == disease_norm]
            headline_shown = False
            if st.session_state.best_model is not None and not disease_rows.empty and st.session_state.feature_cols:
                rep_input = build_representative_row(disease_rows, st.session_state.feature_cols)
                if not rep_input.empty:
                    try:
                        pred_log = st.session_state.best_model.predict(rep_input)
                        pred = float(np.expm1(pred_log)[0])
                        st.header(" Estimated Treatment Cost : ")
                        st.success(f"For **{disease_input}**: **₹{pred:,.0f}**")
                        preds_case_log = st.session_state.best_model.predict(disease_rows[st.session_state.feature_cols])
                        preds_case = np.expm1(preds_case_log)
                        low = float(np.percentile(preds_case, 10))
                        high = float(np.percentile(preds_case, 90))
                        st.write(f"Estimated Treatment Cost Range : **₹{low:,.0f} – ₹{high:,.0f}**")
                        headline_shown = True
                    except Exception as e:
                        pass
            if not headline_shown:
                llm_result = llm_fallback_estimate(disease_input)
                if llm_result:
                    st.header("Estimated Treatment Cost")
                    st.success(f"For **{disease_input}**: **₹{llm_result['mean_cost']:,.0f}**")
                    st.write(f"Estimated Treatment Cost Range: **₹{llm_result['low']:,.0f} – ₹{llm_result['high']:,.0f}**")

# ---------------------------
# Hospital Suggestion Feature (from updated hospital_suggestion.py)
# ---------------------------
elif feature == "Hospital Recommendation System":
    CSV_FILE_PATH = "mumbai_hospitals2.csv"

    # Load Data
    @st.cache_data
    def load_hospital_data():
        df = pd.read_csv(CSV_FILE_PATH)
        df.columns = df.columns.str.strip()
        df['Specialties'] = df['Specialties'].str.lower()
        df['Specialties_list'] = df['Specialties'].str.split(';')
        if "Type" in df.columns:
            df["Type"] = df["Type"].astype(str).str.strip().str.lower()
        return df

    hospital_df = load_hospital_data()

    # Match Hospitals
    def match_hospitals(df, disease):
        disease = disease.lower().strip()
        return df[df['Specialties_list'].apply(lambda specs: any(disease in spec for spec in specs))]

    # AI Suggestion
    def llm_suggest_hospital(disease):
        prompt = f"""
        A user is looking for treatment in Mumbai. Suggest a reputed hospital that treats "{disease}".
        Include: Name, Address, Specialty, and Contact (if possible).
        Separate suggestions for government and private hospitals if known.
        """
        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content([prompt])
            return response.text
        except Exception as e:
            return f"⚠️ AI suggestion failed: {e}"

    # Hospital Card UI
    def render_hospital_card(row):
        st.markdown(f"""
        <div style="background-color:#ffffff;
                    border:1px solid #E0E0E0;
                    padding:15px;
                    border-radius:8px;
                    margin-bottom:15px;
                    box-shadow:0 2px 5px rgba(0,0,0,0.05);">
            <h4 style="color:#4CAF50;margin-bottom:8px;">🏥 {row['Name']}</h4>
            <p><b>📍 Address:</b> {row['Address']}, {row['Area']}</p>
            <p><b>🩺 Specialties:</b> {row['Specialties']}</p>
            <p><b>⏰ Hours:</b> {row['OpenHours']} &nbsp; | &nbsp; <b>⭐ Rating:</b> {row['Rating']}</p>
            <p><b>📞 Contact:</b> {row['Contact']}</p>
        </div>
        """, unsafe_allow_html=True)

    # UI Start
    st.markdown("""
    <style>
        .stApp {
            background-color: #FFF6F0;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #FFF6F0;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color:#FFEBDD;
                padding:25px 20px;
                border-radius:10px;
                margin-bottom:25px;">
        <h2 style="color:#333333;">🏥 HOSPITAL RECOMMENDATION SYSTEM</h2>
        <p style="font-size:16px;color:#555555;">
            🩺 Get hospital suggestions based on your disease or condition! ⚚
        </p>
    </div>
    """, unsafe_allow_html=True)

    disease_input = st.text_input("Enter your disease or condition 🌐 (e.g., Cardiology, Neurology, Oncology):")

    if st.button("Suggest Hospitals 🏨") and disease_input:
        matched = match_hospitals(hospital_df, disease_input)
        st.markdown(f"""
        <div style="padding:10px 0 15px 0">
            <p style="font-size:16px;">
                🔎 <b>Searching for hospitals specializing in:</b>
                <span style="color:#1976D2;">{disease_input.title()}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        if not matched.empty:
            st.success(f"✅ Here are {len(matched)} hospitals for '{disease_input.title()}' for your reference...")
            gov_df = matched[matched['Type'] == 'government']
            priv_df = matched[matched['Type'] == 'private']
            st.markdown("### 🏛️ Government Hospitals")
            if gov_df.empty:
                st.info("No government hospitals found.")
            else:
                gov_rows = gov_df.to_dict(orient="records")
                for i in range(0, len(gov_rows), 3):
                    cols = st.columns(3)
                    for idx, row in enumerate(gov_rows[i:i+3]):
                        with cols[idx]:
                            render_hospital_card(row)
            st.markdown("### 🏢 Private Hospitals")
            if priv_df.empty:
                st.info("No private hospitals found.")
            else:
                priv_rows = priv_df.to_dict(orient="records")
                for i in range(0, len(priv_rows), 3):
                    cols = st.columns(3)
                    for idx, row in enumerate(priv_rows[i:i+3]):
                        with cols[idx]:
                            render_hospital_card(row)
        else:
            with st.spinner("🤖 Generating AI-powered recommendation..."):
                ai_result = llm_suggest_hospital(disease_input)
            st.markdown(f"""
            <div style="background-color:#F0FAFF;
                        border-left:6px solid #2196F3;
                        padding:25px 30px;
                        border-radius:12px;
                        margin-top:30px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.1);">
                <h4 style="color:#0D47A1;margin-bottom:15px;">🤖 AI Recommended Hospitals</h4>
                <div style="font-size:15px;color:#333333;line-height:1.8;">
                    {ai_result}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Static Side-by-Side Plots
    st.subheader("📊 Top Hospitals in Mumbai by Rating")
    private_hospitals = [
        "Lilavati Hospital", "Bombay Hospital", "Nanavati Super Speciality",
        "Hiranandani Hospital", "Wockhardt Hospital", "Breach Candy Hospital",
        "Jaslok Hospital", "Sahyadri Hospital", "Apollo Spectra", "SevenHills Hospital"
    ]
    private_ratings = [4.8, 4.7, 4.6, 4.5, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0]
    government_hospitals = [
        "King Edward Memorial Hospital", "Sir J.J. Hospital", "BYL Nair Hospital",
        "Sion Hospital", "KEM Trauma Center", "Lokmanya Tilak Hospital",
        "Nair Hospital Dental", "ESI Hospital", "Cooper Hospital", "Rural Hospital Dahanu"
    ]
    government_ratings = [4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7]
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8,6))
        colors = sns.color_palette("pink", len(private_hospitals))
        sns.barplot(x=private_ratings, y=private_hospitals, palette=colors, ax=ax1)
        ax1.set_xlabel("Rating ⭐")
        ax1.set_ylabel("Private Hospitals")
        ax1.set_title("Top 10 Private Hospitals in Mumbai")
        ax1.set_xticks([])
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8,6))
        colors = sns.color_palette("Blues_r", len(government_hospitals))
        sns.barplot(x=government_ratings, y=government_hospitals, palette=colors, ax=ax2)
        ax2.set_xlabel("Rating ⭐")
        ax2.set_ylabel("Government Hospitals")
        ax2.set_title("Top 10 Government Hospitals in Mumbai")
        ax2.set_xticks([])
        st.pyplot(fig2)

# ---------------------------
# Motivational Buddy Feature (from motivational_buddy.py)
# ---------------------------
elif feature == "Motivational Buddy":
    # Safe secrets/env access
    def get_secret_safe(key: str, default: str | None = None) -> str | None:
        try:
            return st.secrets[key]
        except Exception:
            return os.getenv(key, default)

    SYSTEM_INSTRUCTION = (
        "You are a warm, positive Motivational Healthcare Buddy. "
        "Respond with an uplifting, supportive message (4–6 sentences) that encourages healthy habits, "
        "self-care, and a positive mindset. Use simple, kind, and inspiring language with emojis. "
        "Do NOT provide medical diagnoses or prescriptions. "
        "Focus only on motivation, encouragement, and small actionable steps."
    )

    GENERATION_CONFIG = {
        "temperature": 0.8,
        "max_output_tokens": 1000,
    }

    def make_model(name: str):
        return genai.GenerativeModel(name, system_instruction=SYSTEM_INSTRUCTION)

    primary_model = make_model(PRIMARY_MODEL_NAME)
    fallback_model = make_model(FALLBACK_MODEL_NAME)

    # Background Setup
    def get_base64_of_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            st.warning("⚠️ Background image not found.")
            return ""
        except Exception as e:
            st.warning(f"⚠️ Could not load background image: {e}")
            return ""

    def set_background(image_path: str):
        encoded = get_base64_of_image(image_path)
        if encoded:
            st.markdown(
                f"""
<style>
[data-testid="stApp"] {{
    background-image: url("data:image/jpg;base64,{encoded}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    color: #e0f7fa;
}}
[data-testid="stApp"] * {{
    text-shadow: 1px 1px 2px rgba(0,0,0,0.6);
}}
.header {{
    background: rgba(0, 51, 102, 0.8);
    color: #ffffff;
    font-size: 34px;
    font-weight: 700;
    padding: 14px 28px;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}}
.subtext {{
    background-color: rgba(0, 0, 0, 0.55);
    color: #f1f1f1;
    padding: 12px 18px;
    font-size: 17px;
    border-radius: 10px;
}}
.main-text {{
    background-color: rgba(0, 0, 0, 0.5);
    color: #f2f2f2;
    padding: 14px 18px;
    font-size: 18px;
    font-weight: 500;
    margin-top: 20px;
}}
.motivation-box {{
    font-size: 18px;
    background: rgba(255, 255, 255, 0.95);
    color: #003366;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    font-weight: 500;
    line-height: 1.6;
    max-width: 900px;
    margin: 18px auto;
    text-align: left;
    white-space: pre-wrap;
}}
.quote-box {{
    font-size: 17px;
    background: rgba(0, 102, 204, 0.95);
    color: #ffffff;
    padding: 16px;
    margin-top: 15px;
    border-radius: 10px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.3);
    font-style: italic;
    text-align: center;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}}
.stButton > button {{
    background: #0072ff;
    color: white;
    font-weight: 600;
    padding: 0.5em 1.2em;
    font-size: 15px;
    border-radius: 8px;
    border: none;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.25);
    transition: all 0.3s ease-in-out;
}}
.stButton > button:hover {{
    background: #005bb5;
    transform: scale(1.03);
}}
.stTextInput input::placeholder {{
    color: #111111 !important;
    opacity: 1 !important;
}}
.stTextInput label {{
    color: #111111 !important;
    text-shadow: none !important;
}}
</style>
                """,
                unsafe_allow_html=True
            )

    set_background("images/motiv1.jpg")

    # JSON Extraction/Parsing
    def _extract_json_candidate(text: str) -> str | None:
        if not text:
            return None
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1)
        first = text.find('{')
        if first == -1:
            return None
        depth = 0
        for i in range(first, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[first:i+1]
        return None

    def _parse_model_json(text: str) -> dict | None:
        candidate = _extract_json_candidate(text)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                msg_m = re.search(r'"message"\s*:\s*"(?P<msg>(?:\\.|[^"\\])*)"', candidate, re.DOTALL)
                q_m = re.search(r'"quote"\s*:\s*"(?P<q>(?:\\.|[^"\\])*)"', candidate, re.DOTALL)
                result = {}
                if msg_m:
                    raw = msg_m.group("msg")
                    try:
                        result["message"] = json.loads(f'"{raw}"')
                    except Exception:
                        result["message"] = raw.replace('\\"', '"')
                if q_m:
                    rawq = q_m.group("q")
                    try:
                        result["quote"] = json.loads(f'"{rawq}"')
                    except Exception:
                        result["quote"] = rawq.replace('\\"', '"')
                if result:
                    return result
                return None
        return None

    def _extract_text(resp):
        if not resp:
            return ""
        try:
            txt = getattr(resp, "text", None)
            if isinstance(txt, str) and txt.strip():
                return txt
        except Exception:
            pass
        try:
            candidates = getattr(resp, "candidates", [])
            if candidates:
                cand0 = candidates[0]
                try:
                    content = getattr(cand0, "content", None)
                    if content:
                        parts = getattr(content, "parts", [])
                        if parts and len(parts) > 0:
                            part0 = parts[0]
                            if hasattr(part0, "text") and part0.text.strip():
                                return part0.text
                        out_text = getattr(content, "text", None) or getattr(cand0, "output_text", None)
                        if isinstance(out_text, str) and out_text.strip():
                            return out_text
                except Exception:
                    pass
        except Exception:
            pass
        try:
            s = str(resp)
            if s and s.strip():
                return s
        except Exception:
            pass
        return ""

    # Gen AI: Agentic Motivation + Quote
    def generate_motivation(feeling: str) -> dict:
        feeling = (feeling or "").strip()
        if not feeling:
            return {"message": "Please share how you're feeling so I can encourage you! 🌟", "quote": ""}
        agentic_prompt = f"""
The user said: "{feeling}"
Act as an empathic Motivational Healthcare Buddy.
Follow these steps internally:
1. Identify the user’s emotional state in 2–3 words.
2. Choose one motivational theme (comfort, resilience, hope, energy).
3. Generate a short motivational healthcare message (4–6 sentences) tailored to the theme.
4. Select or create a motivational quote that aligns with the theme.
Respond in JSON format only:
{{
  "feeling": "...",
  "theme": "...",
  "message": "...",
  "quote": "..."
}}
"""
        try:
            resp = primary_model.generate_content(agentic_prompt, generation_config=GENERATION_CONFIG)
            text = _extract_text(resp).strip()
            if not text:
                resp = primary_model.generate_content(agentic_prompt, generation_config=GENERATION_CONFIG)
                text = _extract_text(resp).strip()
            if text:
                parsed = _parse_model_json(text)
                if parsed:
                    return {"message": parsed.get("message", ""), "quote": parsed.get("quote", "")}
                return {"message": re.sub(r"```(?:json)?|```", "", text).strip(), "quote": ""}
            resp2 = fallback_model.generate_content(agentic_prompt, generation_config=GENERATION_CONFIG)
            text2 = _extract_text(resp2).strip()
            if text2:
                parsed2 = _parse_model_json(text2)
                if parsed2:
                    return {"message": parsed2.get("message", ""), "quote": parsed2.get("quote", "")}
                return {"message": re.sub(r"```(?:json)?|```", "", text2).strip(), "quote": ""}
            return {"message": "I’m here for you. Even small steps—like a short walk or a glass of water—are wins. 🌿💪", "quote": ""}
        except Exception as e:
            st.error(f"🚫 Error generating motivation: {e}")
            return {"message": "Sorry, I couldn't generate motivation due to an error.", "quote": ""}

    # UI
    st.markdown('<div class="header">Motivational Healthcare Buddy</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">💡 Share how you’re feeling, and I’ll send you health motivation and encouragement 🌿💪</p>', unsafe_allow_html=True)
    user_input = st.text_input("✨ How are you feeling about your health today?")
    if st.button("Get Motivation"):
        if not user_input.strip():
            st.warning("⚠️ Please enter your feelings or thoughts to get motivation.")
        else:
            with st.spinner("✨ Crafting your motivation..."):
                response = generate_motivation(user_input)
            raw_msg = (response.get("message") or "").strip()
            raw_quote = (response.get("quote") or "").strip()
            final_msg = raw_msg
            final_quote = raw_quote
            try:
                if (raw_msg.startswith("{") and raw_msg.endswith("}")) or ('"message"' in raw_msg):
                    parsed_inner = _parse_model_json(raw_msg)
                    if parsed_inner:
                        final_msg = (parsed_inner.get("message") or "").strip() or final_msg
                        final_quote = (parsed_inner.get("quote") or "").strip() or final_quote
            except Exception:
                pass
            st.markdown(f'<div class="motivation-box">💬 {final_msg}</div>', unsafe_allow_html=True)
            if final_quote:
                st.markdown(f'<div class="quote-box">🌟 “{final_quote}”</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="main-text">👉 Share your current health mood or challenge to receive encouragement.</p>', unsafe_allow_html=True)
