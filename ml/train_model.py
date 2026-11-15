
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("AI Fake Review Detection System")

hotel_data = pd.read_csv("dataset/deceptive-opinion.csv")

hotel_data = hotel_data[["text", "deceptive"]].dropna()
hotel_data["deceptive"] = hotel_data["deceptive"].str.lower().map({
    "deceptive": "fake",
    "truthful": "real"
})

hotel_data = hotel_data.dropna()
hotel_data.columns = ["review", "label"]
hotel_data["category"] = "Hotel"

amazon_data = pd.read_csv(
    "additional_datasets/Deception-Detection-on-Amazon-reviews-dataset-master/amazon_reviews.txt",
    sep="\t"
)

print("\nAmazon Product Categories:")
print(amazon_data["PRODUCT_CATEGORY"].value_counts())

amazon_data = amazon_data[["LABEL", "REVIEW_TEXT", "PRODUCT_CATEGORY"]].dropna()
amazon_data["LABEL"] = amazon_data["LABEL"].map({
    "__label1__": "fake",
    "__label2__": "real"
})

amazon_data.columns = ["label", "review", "category"]
print("\nAmazon total reviews:", len(amazon_data))
print("Amazon duplicate reviews:", amazon_data["review"].duplicated().sum())

conflicting_reviews = amazon_data.groupby("review")["label"].nunique()
print("Amazon reviews with conflicting labels:", (conflicting_reviews > 1).sum())
amazon_data["word_count"] = amazon_data["review"].astype(str).str.split().str.len()

print("\nAmazon Review Length Analysis:")
print(amazon_data.groupby("label")["word_count"].describe())

print("\nReviews with fewer than 10 words:")
print(amazon_data[amazon_data["word_count"] < 10]["label"].value_counts())
data = pd.concat([hotel_data, amazon_data], ignore_index=True)

data = data.dropna(subset=["review", "label"])
data["review"] = data["review"].astype(str)

print("\nTotal reviews:", len(data))
print("\nClass distribution:")
print(data["label"].value_counts())

categories = data["category"]
X = data["review"]
y = data["label"]

X_train, X_test, y_train, y_test, category_train, category_test = train_test_split(
    X,
    y,
    categories,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF feature extraction completed!")

model = LogisticRegression(max_iter=1000)
svm_model = LinearSVC(random_state=42)

svm_model.fit(X_train_tfidf, y_train)

svm_predictions = svm_model.predict(X_test_tfidf)

print("\nLinearSVC Accuracy:", round(accuracy_score(y_test, svm_predictions) * 100, 2), "%")

print("\nLinearSVC Classification Report:")
print(classification_report(y_test, svm_predictions))
model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
category_results = pd.DataFrame({
    "category": category_test,
    "actual": y_test,
    "predicted": y_pred
})

category_results["correct"] = (
    category_results["actual"] == category_results["predicted"]
)

category_accuracy = category_results.groupby("category").agg(
    total_reviews=("correct", "count"),
    accuracy=("correct", "mean")
)

category_accuracy["accuracy"] = (
    category_accuracy["accuracy"] * 100
).round(2)

category_accuracy = category_accuracy.sort_values(
    by="accuracy",
    ascending=False
)

print("\nCategory-Wise Model Accuracy:")
print(category_accuracy.to_string())
hotel_indices = y_test.index[y_test.index < len(hotel_data)]
amazon_indices = y_test.index[y_test.index >= len(hotel_data)]

print("\nHotel Test Results:")
print(classification_report(
    y_test.loc[hotel_indices],
    pd.Series(y_pred, index=y_test.index).loc[hotel_indices]
))

print("\nAmazon Test Results:")
print(classification_report(
    y_test.loc[amazon_indices],
    pd.Series(y_pred, index=y_test.index).loc[amazon_indices]
))

joblib.dump(model, "ml/fake_review_model.pkl")
joblib.dump(vectorizer, "ml/tfidf_vectorizer.pkl")


print("\nModel saved successfully!")

