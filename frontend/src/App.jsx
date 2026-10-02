import { useState } from "react";
import "./App.css";

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeReview = async () => {
    if (!review.trim()) {
      alert("Please enter a review.");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          review: review,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResult(data);
    } catch (error) {
      setResult({
        error: "Could not connect to the AI backend.",
      });
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <nav className="navbar">
        <div className="logo">
          <span>AI</span> Fake Review Detector
        </div>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#about">About</a>
        </div>
      </nav>

      <main id="home">
        <section className="hero-section">
          <div className="hero-content">
            <p className="tagline">AI-POWERED REVIEW ANALYSIS</p>

            <h1>
              Detect <span>Fake Reviews</span>
              <br />
              with AI
            </h1>

            <p className="description">
              Enter a hotel review and let our Machine Learning model analyze
              whether it appears truthful or deceptive.
            </p>

            <div className="review-box">
              <textarea
                value={review}
                onChange={(e) => setReview(e.target.value)}
                placeholder="Enter a hotel review here..."
                rows="7"
              />

              <button onClick={analyzeReview} disabled={loading}>
                {loading ? "Analyzing..." : "Analyze Review"}
              </button>
            </div>

            {result && (
              <div className="result-box">
                {result.error ? (
                  <p className="error">{result.error}</p>
                ) : (
                  <>
                    <h2>Analysis Result</h2>

                    <div
                      className={
                        result.prediction?.toLowerCase() === "deceptive"
                          ? "prediction fake"
                          : "prediction truthful"
                      }
                    >
                      {result.prediction?.toUpperCase()}
                    </div>

                    {result.confidence !== undefined && (
                      <p>
                        Confidence:{" "}
                        <strong>{Number(result.confidence).toFixed(2)}%</strong>
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </section>

        <section id="about" className="about-section">
          <h2>How It Works</h2>

          <div className="features">
            <div className="feature-card">
              <h3>01</h3>
              <h4>Enter Review</h4>
              <p>
                Provide the hotel review that you want to analyze.
              </p>
            </div>

            <div className="feature-card">
              <h3>02</h3>
              <h4>AI Analysis</h4>
              <p>
                The review is converted into TF-IDF features and processed by
                our trained Machine Learning model.
              </p>
            </div>

            <div className="feature-card">
              <h3>03</h3>
              <h4>Get Result</h4>
              <p>
                The system predicts whether the review is truthful or
                deceptive and provides a confidence value.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <p>AI Fake Review Detection System</p>
        <p>Machine Learning + NLP</p>
      </footer>
    </div>
  );
}

export default App;