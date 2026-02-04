import pandas as pd
import pickle
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load data
df = pd.read_csv("data/data.csv")

print("Columns found:", df.columns)

# ✅ CORRECT COLUMN NAMES (FROM YOUR DATASET)
RATING_COL = "Ratings"
REVIEW_COL = "Review text"

# Select required columns
df = df[[RATING_COL, REVIEW_COL]].dropna()

# Convert rating to numeric (safety)
df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
df = df.dropna()

# Sentiment labeling
def sentiment_label(rating):
    if rating >= 4:
        return 1   # Positive
    elif rating <= 2:
        return 0   # Negative
    else:
        return None  # Neutral (drop)

df["sentiment"] = df[RATING_COL].apply(sentiment_label)
df = df.dropna()

# Text cleaning
df["clean_review"] = df[REVIEW_COL].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("✅ F1 Score:", f1_score(y_test, y_pred))

# Save artifacts
pickle.dump(model, open("models/sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer/tfidf_vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully")
