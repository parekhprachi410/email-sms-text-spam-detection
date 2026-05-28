import streamlit as st
import joblib
import os
import pandas as pd

if not os.path.exists("spam_model.pkl"):

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    df = pd.read_csv("spam.csv",encoding='latin-1')

    df=df[['Category','Message']]

    df['Category']=df['Category'].map({
        'ham':0,
        'spam':1
    })

    model=Pipeline([

        ('tfidf',
         TfidfVectorizer(
             stop_words='english'
         )),

        ('clf',
         MultinomialNB())

    ])

    model.fit(
        df['Message'],
        df['Category']
    )

    joblib.dump(
        model,
        "spam_model.pkl"
    )

model=joblib.load(
    "spam_model.pkl"
)

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