from flask import Flask, render_template, request
import pickle
from preprocess import clean_text

app = Flask(__name__)

model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer/tfidf_vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        review = request.form["review"]
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
