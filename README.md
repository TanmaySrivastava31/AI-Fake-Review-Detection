# AI-Based Fake Review Detection and Authenticity Scoring System

An AI-powered web application that analyzes online review text, predicts whether a review is potentially **Fake** or **Real**, and displays a model-estimated **Authenticity Score**.

The project combines Natural Language Processing (NLP), Machine Learning, Flask, and React to provide an interactive review analysis system.

## Project Overview

Online reviews influence purchasing decisions, hotel bookings, and customer trust. However, deceptive reviews can mislead consumers.

This project uses machine learning to identify patterns in review text that may indicate deception.

Users can enter a product, hotel, or service review and receive:

- A **Fake or Real** prediction
- A model confidence percentage
- An authenticity score between 0% and 100%
- A visual authenticity progress bar

**Important:** The application estimates authenticity using patterns learned from training data. It does not independently verify whether a reviewer purchased a product or genuinely experienced a service.

## Objectives

- Detect potentially deceptive online reviews using NLP
- Train a text classification model using labeled review datasets
- Extract textual features using TF-IDF
- Combine word-level and character-level text features
- Classify reviews as Fake or Real
- Generate model confidence and authenticity scores
- Evaluate performance on held-out reviews
- Test generalization across previously unseen product categories
- Develop a Flask REST API for predictions
- Build an interactive React frontend
- Handle invalid review inputs safely
- Maintain project versions using Git and GitHub

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Machine learning and backend development |
| Scikit-learn | TF-IDF feature extraction, classification, and evaluation |
| Logistic Regression | Fake/Real review classification |
| Flask | Backend REST API |
| Flask-CORS | Frontend-backend communication |
| React.js | Interactive frontend |
| JavaScript | Frontend application logic |
| HTML and CSS | Website structure and styling |
| Vite | React development server and build tool |
| Joblib | Model and vectorizer serialization |
| Git and GitHub | Version control |

## Dataset

The current model was trained using **22,600 labeled reviews** collected from two review domains.

| Dataset | Number of reviews |
|---|---:|
| Amazon product reviews | 21,000 |
| Hotel reviews | 1,600 |
| **Total** | **22,600** |

The combined dataset contains balanced Fake and Real labels.

The Amazon reviews cover **30 product categories**, providing a broader training set than the original hotel-only model.

Some additional dataset files are excluded from Git tracking, so they may need to be obtained separately when reproducing the training process.

## Machine Learning Methodology

### 1. Text Feature Extraction

The system uses a combination of two TF-IDF representations:

**Word-level TF-IDF**

- Maximum features: 20,000
- Unigrams and bigrams
- English stop-word removal

**Character-level TF-IDF**

- Maximum features: 30,000
- Character n-grams of lengths 3–5
- Sublinear term-frequency scaling

These feature representations are combined using Scikit-learn's `FeatureUnion`.

### 2. Classification Algorithm

The current classifier uses **Logistic Regression** with a maximum of 1,000 iterations.

A Linear Support Vector Classifier was also evaluated during model development.

### 3. Prediction Process

The prediction workflow is:

1. The user enters a review in the React interface.
2. React sends the review to the Flask API.
3. The backend validates the input.
4. The trained vectorizer transforms the text into numerical features.
5. The Logistic Regression model predicts Fake or Real.
6. The model calculates classification scores.
7. Flask returns the prediction results as JSON.
8. React displays the results and authenticity progress bar.

## Model Performance

The current model was evaluated on held-out review data.

| Evaluation metric | Result |
|---|---:|
| Total reviews in combined dataset | 22,600 |
| Overall held-out test accuracy | **66.37%** |
| Hotel-review test accuracy | **85.67%** |
| Amazon-review test accuracy | Approximately **65%** |
| Unseen-category evaluation average | **65.65%** |
| LinearSVC comparison accuracy | **64.25%** |

### Unseen-Category Evaluation

To evaluate generalization, the project uses **six-fold GroupKFold** testing across Amazon product categories.

In each fold:

- The model trains on reviews from 25 product categories.
- The model is evaluated on five previously unseen categories.

The average accuracy across these evaluations is **65.65%**.

This evaluation helps measure how well the classifier performs when it encounters product categories that were not represented in its training fold.

### Performance Limitations

Although the model demonstrates the complete machine learning workflow, its accuracy is moderate.

The model can make incorrect predictions, including classifying genuine reviews as fake and deceptive reviews as real.

Performance may vary across products, services, writing styles, languages, and review domains.

The system should therefore be used as an **experimental review-analysis tool**, not as definitive evidence of fraud or deception.

## Authenticity Scoring

The application calculates an authenticity score using the Logistic Regression model's estimated probability for the `real` class.

**Authenticity Score = Model score for the Real class × 100**

The application also displays the confidence associated with the predicted class.

For example:

| Output | Example |
|---|---|
| Prediction | FAKE |
| Confidence | 88.90% |
| Authenticity Score | 11.10% |

The frontend displays the authenticity score using a purple-to-green progress bar.

**Note:** These scores are not independently calibrated or verified probabilities of authenticity. A score of 90% does not establish that a review has a 90% chance of being genuine.

## Application Features

- Fake/Real review prediction
- Model confidence percentage
- Authenticity score percentage
- Animated authenticity progress bar
- Dark-themed React interface
- Flask REST API integration
- Input validation for missing, empty, and invalid review values
- Responsive interface styling
- Error messages for invalid requests

## Project Structure

```text
AI-Fake-Review-Detection/
├── backend/
│   └── app.py
├── dataset/
│   └── deceptive-opinion.csv
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   └── package.json
├── ml/
│   ├── fake_review_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── predict_review.py
│   ├── test_unseen_categories.py
│   └── notebooks/
│       └── train_model.py
├── .gitignore
└── README.md
```

Additional dataset and evaluation files may be present locally depending on the development environment.

## Installation and Setup

### Prerequisites

Install the following tools:

- Python
- Node.js and npm
- Git
- Visual Studio Code or another code editor

### 1. Clone the Repository

```bash
git clone https://github.com/TanmaySrivastava31/AI-Fake-Review-Detection.git
cd AI-Fake-Review-Detection
```

### 2. Create a Python Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Backend Dependencies

```bash
pip install flask flask-cors joblib scikit-learn
```

Ensure the trained model and vectorizer files are available in the `ml` directory.

### 4. Start the Flask Backend

From the project root:

```bash
python backend/app.py
```

The backend runs locally at:

`http://127.0.0.1:5000`

Keep this terminal running.

### 5. Start the React Frontend

Open a second terminal.

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, typically:

`http://localhost:5173`

Keep this terminal running while using the application.

## How to Use the Application

1. Open the React application in your browser.
2. Enter a product, hotel, or service review in the text box.
3. Click **Analyze Review**.
4. Wait for the model to process the review.
5. View the Fake/Real prediction.
6. Check the model confidence percentage.
7. Examine the authenticity score and visual progress bar.

The score represents the model's assessment of the review text, not a verified judgment about the reviewer.

## Flask API

### Health Check

**Endpoint:** `GET /`

Example response:

```json
{
  "message": "AI Fake Review Detection API is running"
}
```

### Predict Review

**Endpoint:** `POST /predict`

**Content-Type:** `application/json`

Example request:

```json
{
  "review": "This product works well and arrived on time. The quality is good."
}
```

Example response from the current trained model:

```json
{
  "prediction": "fake",
  "confidence": 82.08,
  "authenticity_score": 17.92
}
```

The exact results depend on the loaded model.

### Input Validation

The API validates submitted review text.

If a review is missing:

```json
{
  "error": "Review text is required"
}
```

If a review is not a string:

```json
{
  "error": "Review must be a text string"
}
```

If a review is empty or contains only whitespace:

```json
{
  "error": "Review cannot be empty"
}
```

These validation errors return HTTP status code `400`.

## Testing

The application has been tested through:

- Machine learning evaluation on held-out reviews
- Six-fold unseen-category testing
- Flask API requests
- Invalid-input validation
- Valid-review prediction requests
- React and Flask integration
- Browser testing of the authenticity score and progress bar

A sample valid review returned:

- Prediction: `fake`
- Confidence: `82.08%`
- Authenticity score: `17.92%`

These results demonstrate that the API is functioning, but they do not establish whether that particular review is actually fake.

## Current Limitations

- Overall model accuracy is approximately 66%.
- The model may produce false positives and false negatives.
- Results can vary across review categories and domains.
- Authenticity scores are not calibrated probabilities.
- The system analyzes review text rather than reviewer identity, transaction history, or verified purchases.
- The current implementation primarily targets English-language reviews.
- The application currently runs in a local development environment.

## Future Improvements

Potential future enhancements include:

- Improving model accuracy using additional high-quality labeled data
- Evaluating transformer-based NLP models
- Calibrating prediction probabilities
- Adding more review categories and service domains
- Introducing review history and metadata features where available
- Improving error handling and automated testing
- Deploying the application online
- Adding batch review analysis
- Building a dashboard for analyzing multiple reviews

## Conclusion

The AI-Based Fake Review Detection and Authenticity Scoring System demonstrates how Natural Language Processing, Machine Learning, Flask, and React can be integrated into a functional review-analysis application.

The project provides Fake/Real predictions, model confidence estimates, and visual authenticity scores through an interactive web interface.

The current model achieves **66.37% overall held-out accuracy** and **65.65% average accuracy in unseen-category evaluation**.

While the system is functional as a prototype, further model improvements and validation are necessary before using its predictions for high-stakes or real-world moderation decisions.

## GitHub Repository

[AI-Fake-Review-Detection](https://github.com/TanmaySrivastava31/AI-Fake-Review-Detection)