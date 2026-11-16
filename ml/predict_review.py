
import joblib

model = joblib.load("ml/fake_review_model.pkl")
vectorizer = joblib.load("ml/tfidf_vectorizer.pkl")

print("\n================================")
print("   AI Fake Review Detection System")
print("================================")

review = input("\nEnter a review: ")

review_tfidf = vectorizer.transform([review])

prediction = model.predict(review_tfidf)[0]

probability = model.predict_proba(review_tfidf)[0]
confidence = max(probability) * 100
authenticity_score = probability[list(model.classes_).index("real")] * 100

print("\nPrediction:", str(prediction).upper())
print("Confidence:", round(confidence, 2), "%")
print("Authenticity Score:", round(authenticity_score, 2), "%")