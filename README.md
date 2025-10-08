
# 🧠 Fake News Detection

A machine learning project that detects whether a news headline or article is **Fake** or **Real** using **TF-IDF vectorization** and **Logistic Regression**. This project also includes a **Gradio web app** for easy interactive predictions.

---

## 🚀 Project Overview

Fake news spreads misinformation and can have serious societal impacts. This project uses natural language processing (NLP) and machine learning to classify news as Fake or Real based on its text content.  

Key features:

- Preprocessing pipeline: text cleaning, stopword removal, and lemmatization.
- TF-IDF vectorization (unigrams + bigrams).
- Logistic Regression classifier.
- Gradio-based web interface for real-time predictions.
- GitHub-ready structure for reproducibility.

---

## 📁 Dataset

The dataset contains news articles with the following columns:

| Column | Description |
|--------|-------------|
| `title` | Headline of the news article |
| `text`  | Full news content |
| `label` | 0 = Fake, 1 = Real |

**Preprocessing Steps Applied:**

1. Combined `title` + `text`.
2. Lowercased all text.
3. Removed punctuation, numbers, URLs, and extra whitespace.
4. Removed English stopwords.
5. Applied lemmatization.

**Processed dataset:** `processed_news.csv`  
**Raw cleaned dataset:** `cleaned_news.csv`

---

## 📊 Model

- **Vectorizer:** TF-IDF (max 5000 features, unigrams + bigrams)
- **Classifier:** Logistic Regression (`max_iter=1000`)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

Example performance on test data:

```

Accuracy: 0.9847
Classification Report:
precision    recall  f1-score   support
Fake       0.98      0.99      0.98      4696
Real       0.99      0.98      0.98      4284

````

**Saved Artifacts:**

- `fake_news_model.pkl` — trained Logistic Regression model  
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer  

---

## 🌐 Web Interface (Gradio)

Launch a simple interactive web app for testing news articles:

**Features:**

- Paste or type any news headline/article.
- Click Submit → Get prediction:
  - ✅ Real News
  - ❌ Fake News

**Example UI:**

![Gradio UI Screenshot](./screenshot_placeholder.png)  

**Example Predictions:**

| Input | Prediction |
|-------|------------|
| Breaking: NASA confirms discovery of alien life on Mars! | 🚨 Fake News ❌ |
| The Federal Reserve raised interest rates again this quarter. | 📰 Real News ✅ |

---

## 💻 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run preprocessing (optional)**

```python
# Stage 4: Text Preprocessing
python preprocessing.py
```

4. **Vectorize text (optional)**

```python
# Stage 5: TF-IDF Vectorization
python vectorization.py
```

5. **Train the model (optional)**

```python
# Stage 6: Model Training
python train_model.py
```

6. **Launch Gradio app**

```python
python app.py
```

---

## 📂 Repository Structure

```
fake-news-detector/
│
├─ cleaned_news.csv          # Raw cleaned dataset
├─ processed_news.csv        # Preprocessed dataset
├─ fake_news_model.pkl       # Trained model
├─ tfidf_vectorizer.pkl      # TF-IDF vectorizer
├─ app.py                    # Gradio web interface
├─ preprocessing.py          # Stage 4 code
├─ vectorization.py          # Stage 5 code
├─ train_model.py            # Stage 6 code
├─ requirements.txt          # Required Python packages
└─ README.md                 # Project documentation
```

---

## 📌 Notes

* Ensure all NLTK resources are downloaded:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

* TF-IDF vectorizer ignores empty strings, so missing text entries are handled automatically.

* You can expand this project by:

  * Using more advanced models (e.g., Naive Bayes, LSTM, BERT)
  * Adding a confidence score in predictions
  * Deploying the Gradio app on Hugging Face Spaces or Streamlit Cloud

---

## 📖 References

* [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
* [NLTK Documentation](https://www.nltk.org/)
* [Gradio Documentation](https://gradio.app/)

---

## ✨ Author

**Uswah Akhund** – [GitHub](https://github.com/UswahAA)


