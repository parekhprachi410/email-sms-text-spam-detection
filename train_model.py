import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report

# LOAD DATASET
df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

# KEEP NEEDED COLUMNS
df = df[['Category','Message']]

# LABEL ENCODING
df['Category'] = df['Category'].map({
    'ham':0,
    'spam':1
})

X = df['Message']
y = df['Category']

# SPLIT
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# NLP PIPELINE
model = Pipeline([

    (
        'tfidf',
        TfidfVectorizer(
            stop_words='english'
        )
    ),

    (
        'classifier',
        MultinomialNB()
    )

])

# TRAIN
model.fit(X_train,y_train)

# TEST
pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test,pred)
)

print(
    classification_report(
        y_test,
        pred
    )
)

# SAVE MODEL
joblib.dump(
    model,
    "spam_model.pkl"
)

print("Model saved.")