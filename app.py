import streamlit as st
import joblib
import os
import pandas as pd

# ================= TRAIN MODEL IF MISSING =================

if not os.path.exists("spam_model.pkl"):

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    df = pd.read_csv(
        "spam.csv",
        encoding='latin-1'
    )

    df = df[['Category','Message']]

    df['Category'] = df['Category'].map({
        'ham':0,
        'spam':1
    })

    model = Pipeline([

        (
            'tfidf',
            TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                ngram_range=(1,2)
            )
        ),

        (
            'clf',
            MultinomialNB(alpha=0.1)
        )

    ])

    model.fit(
        df['Message'],
        df['Category']
    )

    joblib.dump(
        model,
        "spam_model.pkl"
    )

# ================= LOAD MODEL =================

model = joblib.load(
    "spam_model.pkl"
)

# ================= APP =================

st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📩",
    layout="wide"
)

st.title(
    "📩 AI Live Email/SMS Spam Detector"
)

message = st.text_area(
    "Type Message",
    height=250,
    placeholder="Congratulations! You've won a FREE iPhone!"
)

# ================= PREDICTION =================

if message:

    # ML prediction
    ml_probability = model.predict_proba(
        [message]
    )[0][1]

    # Rule-based phishing boost
    message_lower = message.lower()

    phishing_terms = [

        "bank",
        "verify",
        "account",
        "password",
        "login",
        "payment",
        "security alert",
        "click here",
        "urgent",
        "suspended",
        "free",
        "claim",
        "winner"

    ]

    bonus = 0

    for word in phishing_terms:

        if word in message_lower:
            bonus += 0.08

    probability = min(
        1.0,
        ml_probability + bonus
    )

    prediction = int(
        probability > 0.50
    )

    st.subheader("Prediction")

    st.progress(
        float(probability)
    )

    if probability >= 0.70:

        st.error(
            f"🚨 SPAM / PHISHING ({probability*100:.2f}%)"
        )

    elif probability >= 0.40:

        st.warning(
            f"⚠ Suspicious ({probability*100:.2f}%)"
        )

    else:

        st.success(
            f"✅ SAFE ({(1-probability)*100:.2f}%)"
        )

    with st.expander(
        "Detection Details"
    ):

        st.write(
            f"ML Probability: {ml_probability:.2f}"
        )

        st.write(
            f"Rule Boost: +{bonus:.2f}"
        )

        st.write(
            f"Final Probability: {probability:.2f}"
        )