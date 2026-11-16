import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

amazon_data = pd.read_csv(
    "additional_datasets/Deception-Detection-on-Amazon-reviews-dataset-master/amazon_reviews.txt",
    sep="\t"
)

amazon_data = amazon_data[["LABEL", "REVIEW_TEXT", "PRODUCT_CATEGORY"]].dropna()

amazon_data["LABEL"] = amazon_data["LABEL"].map({
    "__label1__": "fake",
    "__label2__": "real"
})

amazon_data = amazon_data.dropna()

amazon_data.columns = ["label", "review", "category"]

print("Total Amazon reviews:", len(amazon_data))
print("\nAvailable categories:")
print(amazon_data["category"].value_counts())

categories = sorted(amazon_data["category"].unique())

group_kfold = GroupKFold(n_splits=6)
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(
    group_kfold.split(
        amazon_data["review"],
        amazon_data["label"],
        groups=amazon_data["category"]
    ),
    start=1
):
    print(f"\nFold {fold}")
    print("Training reviews:", len(train_idx))
    print("Testing reviews:", len(test_idx))
    print(
        "Unseen categories:",
        amazon_data.iloc[test_idx]["category"].unique()
    )

    train_data = amazon_data.iloc[train_idx]
    test_data = amazon_data.iloc[test_idx]

    X_train = train_data["review"]
    y_train = train_data["label"]

    X_test = test_data["review"]
    y_test = test_data["label"]

    print("\nTraining reviews:", len(train_data))
    print("Unseen-category testing reviews:", len(test_data))

    print("\nUnseen categories:")
    print(test_data["category"].value_counts())

    vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2)
            )),
            ("char", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=30000,
                sublinear_tf=True
            ))
        ])

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
            C=1.0,
            max_iter=1000
        )

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)

    print("\nUnseen-Category Model Accuracy:", round(accuracy * 100, 2), "%")

    print("\nUnseen-Category Classification Report:")
    print(classification_report(y_test, y_pred))

    category_results = test_data[["category", "label"]].copy()
    category_results["predicted"] = y_pred

    category_results["correct"] = (
            category_results["label"] == category_results["predicted"]
        )

    category_accuracy = (
            category_results.groupby("category")["correct"]
            .agg(["count", "mean"])
        )

    category_accuracy["mean"] = (
            category_accuracy["mean"] * 100
        ).round(2)

    category_accuracy.columns = ["total_reviews", "accuracy"]

    print("\nUnseen Category-Wise Accuracy:")
    print(category_accuracy.sort_values("accuracy", ascending=False))

    print("\nActual Label Distribution:")
    print(y_test.value_counts())

    print("\nPredicted Label Distribution:")
    print(pd.Series(y_pred).value_counts())
    print("\nSix-Fold Unseen-Category Evaluation Summary")
    print("Average Accuracy:", round(sum(fold_accuracies) / len(fold_accuracies) * 100, 2), "%")