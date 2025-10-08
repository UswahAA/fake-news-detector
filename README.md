# 📰 Fake News Detector

A **machine learning project** that classifies news articles as **Fake** or **Real** using **TF-IDF vectorization** and **Logistic Regression**.  
This project demonstrates an **end-to-end NLP workflow**, including text preprocessing, model training, evaluation, and deployment via a **Gradio web app**.

---

## 🧠 Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, Gradio  
- **Model:** Logistic Regression  
- **Feature Extraction:** TF-IDF Vectorization  
- **UI / Deployment:** Gradio

---

## 📊 Dataset

- **Source:** [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
- Contains labeled news articles:  
  - `Fake` = 0  
  - `Real` = 1  
- Cleaned and preprocessed for this project (~20k–30k rows used)  
- Split into **train/test sets** (80/20 split)

---

## 🏗️ Project Pipeline

1. **Data Setup & Exploration**  
   - Load Fake and Real news datasets  
   - Merge and shuffle data  
   - Add labels and save cleaned dataset  

2. **Text Preprocessing**  
   - Lowercase text  
   - Remove punctuation, numbers, URLs  
   - Remove stopwords and lemmatize words  

3. **Vectorization**  
   - Convert text into numerical features using **TF-IDF**  
   - Consider unigrams and bigrams  

4. **Model Training & Evaluation**  
   - Train **Logistic Regression** on TF-IDF features  
   - Evaluate using **accuracy score**, **classification report**, and **confusion matrix**  

5. **Deployment (Gradio App)**  
   - Build a simple UI to input news headline or paragraph  
   - Output prediction: **Fake News** 🚨 / **Real News** ✅  

---

## 📅 Project Progress

| Stage | Tasks | Status |
|-------|-------|--------|
| 1 | Data setup and exploration | ✅ Completed |
| 2 | Text cleaning & preprocessing | ✅ Completed |
| 3 | TF-IDF Vectorization | ✅ Completed |
| 4 | Model training & evaluation | ✅ Completed |
| 5 | Gradio web app deployment | ✅ Completed |

---

## 💾 Saved Files

- `fake_news_model.pkl` → Trained Logistic Regression model  
- `tfidf_vectorizer.pkl` → Saved TF-IDF vectorizer  
- `processed_news.csv` → Preprocessed text dataset  

---

## 📈 Model Performance

- **Accuracy:** ~96% on test dataset  
- **Classification Report:** Included in notebook  
- **Confusion Matrix:** Included in notebook  

> ⚠️ Note: This model is trained on a **static Kaggle dataset**. Predictions for live news may **not reflect real-world accuracy**.

---

## 🚀 Usage Instructions

1. **Clone the repository:**  
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
