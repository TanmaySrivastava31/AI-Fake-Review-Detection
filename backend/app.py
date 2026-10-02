from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "ml", "fake_review_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "ml", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Fake Review Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "review" not in data:
        return jsonify({
            "error": "Review text is required"
        }), 400

    review = data["review"]

    if not review.strip():
        return jsonify({
            "error": "Review cannot be empty"
        }), 400

    review_tfidf = vectorizer.transform([review])

    prediction = model.predict(review_tfidf)[0]
    probabilities = model.predict_proba(review_tfidf)[0]

    confidence = max(probabilities) * 100

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)