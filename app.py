import streamlit as st
import joblib

# LOAD MODEL
model = joblib.load(
    "spam_model.pkl"
)

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

if message:

    probability = model.predict_proba(
        [message]
    )[0][1]

    prediction = model.predict(
        [message]
    )[0]

    st.subheader(
        "Prediction"
    )

    st.progress(
        float(probability)
    )

    if prediction == 1:

        st.error(
            f"🚨 SPAM ({probability*100:.2f}%)"
        )

    elif probability > 0.40:

        st.warning(
            f"⚠ Suspicious ({probability*100:.2f}%)"
        )

    else:

        st.success(
            f"✅ SAFE ({(1-probability)*100:.2f}%)"
        )