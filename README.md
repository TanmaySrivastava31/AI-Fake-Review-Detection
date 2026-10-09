# AI-Based Fake Review Detection and Authenticity Scoring System

## Project Overview

This project is an AI-based web application designed to identify potentially deceptive online reviews.

The system analyzes review text using Natural Language Processing (NLP) and Machine Learning techniques. It classifies reviews as Truthful or Deceptive and displays the prediction along with a model confidence score.

The application is designed for reviews from different domains. However, the current model has been trained and evaluated on a publicly available hotel-review dataset.

## Objectives


- Detect potentially deceptive online reviews
- Analyze review text using Natural Language Processing (NLP)
- Classify reviews as Truthful or Deceptive
- Apply TF-IDF for text feature extraction
- Use Logistic Regression for classification
- Display predictions with a model confidence score
- Evaluate the model using accuracy, precision, recall, and F1-score
- Provide a user-friendly web interface using React
- Integrate the machine learning model through a Flask API


## Technology Stack

- Python — Machine Learning and backend development
- Scikit-learn — TF-IDF and Logistic Regression
- Flask — Backend REST API
- React.js — Frontend user interface
- JavaScript — Frontend functionality
- HTML and CSS — Webpage structure and styling
- Vite — Frontend development server and build tool
- Joblib — Saving and loading the trained model
- Git and GitHub — Version control and project hosting

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
│   └── notebooks/
│       └── train_model.py
├── .gitignore
└── README.md
```


## Model Performance

The model was trained and evaluated using the Deceptive Opinion Spam Corpus.

- Total reviews: 1,600
- Truthful reviews: 800
- Deceptive reviews: 800
- Training samples: 1,280 (80%)
- Testing samples: 320 (20%)
- Algorithm: Logistic Regression
- Feature extraction: TF-IDF
- Test accuracy: 88.44%

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Deceptive | 0.89 | 0.88 | 0.88 |
| Truthful | 0.88 | 0.89 | 0.89 |

**Note:** These results are based on the hotel-review dataset. Performance on other review domains has not yet been evaluated.


## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/TanmaySrivastava31/AI-Fake-Review-Detection.git
cd AI-Fake-Review-Detection
```

### 2. Set Up the Python Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```bash
pip install flask flask-cors joblib scikit-learn
```

### 3. Start the Backend

```bash
python backend/app.py
```

The Flask API runs at `http://127.0.0.1:5000`.

### 4. Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually `http://localhost:5173`.

### 5. Use the Application

1. Enter a review in the text box.
2. Click **Analyze Review**.
3. View the predicted class and model confidence score.
