
# Basic setup
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report pd.read_csv("/content/Fake.csv", on_bad_lines='skip', nrows=2).head()   import pandas as pd

# --- Load datasets with proper quoting ---
fake = pd.read_csv("/content/Fake.csv", encoding='utf-8', quotechar='"', on_bad_lines='skip')
true = pd.read_csv("/content/True.csv", encoding='utf-8', quotechar='"', on_bad_lines='skip')

print("Fake columns:", fake.columns)
print("True columns:", true.columns)

# --- Keep only title and text (if available) ---
if 'title' in fake.columns and 'text' in fake.columns:
    fake = fake[['title', 'text']]
    true = true[['title', 'text']]
else:
    print("⚠️ Columns not found. Check printed column names above!")

# --- Add labels ---
fake["label"] = 0
true["label"] = 1

# --- Combine and shuffle ---
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Check ---
print("\n✅ Cleaned data sample:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

# --- Save cleaned version ---
data.to_csv("/content/cleaned_news.csv", index=False)
print("\n✅ Clean dataset saved successfully as 'cleaned_news.csv'") # --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load cleaned data ---
data = pd.read_csv("/content/cleaned_news.csv")
print("Data shape:", data.shape)
print(data.head())

# --- Split data ---
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nVectorization complete.")

# --- Model Training ---
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# --- Predictions ---
y_pred = model.predict(X_test_tfidf)

# --- Evaluation ---
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
 import pickle

# Save model
with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and vectorizer saved!")  # Load saved items
with open("fake_news_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

# Test example
sample_text = ["Breaking: Scientists Discover Aliens Living Under the Ocean"]
sample_features = loaded_vectorizer.transform(sample_text)
prediction = loaded_model.predict(sample_features)[0]

print("📰 Prediction:", "Fake News" if prediction == 0 else "Real News")  

import gradio as gr

def predict_news(text):
    features = loaded_vectorizer.transform([text])
    pred = loaded_model.predict(features)[0]
    return "🧾 Real News ✅" if pred == 1 else "🚨 Fake News ❌"

demo = gr.Interface(
    fn=predict_news,
    inputs="text",
    outputs="text",
    title="Fake News Detector",
    description="Enter any news headline or paragraph to check if it's real or fake."
)  

demo.launch()  # ===============================================
# 🧠 Stage 4 — Model Training and Evaluation
# ===============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Load cleaned dataset ---
df = pd.read_csv("/content/cleaned_news.csv")

print("Data shape:", df.shape)
print(df.head())

# --- Split the dataset ---
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# --- Vectorize text data using TF-IDF ---
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nVectorization complete.")

# --- Train a Logistic Regression model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --- Evaluate the model ---
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# --- Save model and vectorizer ---
import pickle

with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("\n✅ Model and vectorizer saved successfully as 'fake_news_model.pkl' and 'vectorizer.pkl'") # ===============================================
# 🧹 Stage 4 — Text Preprocessing
# ===============================================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# --- Load your dataset ---
df = pd.read_csv("/content/cleaned_news.csv")

# --- Initialize preprocessing tools ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Define text cleaning function ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                              # lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove links
    text = re.sub(r'[^a-z\s]', '', text)             # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()         # remove extra spaces
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- Apply cleaning to all text ---
df['clean_text'] = df['text'].apply(clean_text)

# --- Save the cleaned version ---
df[['clean_text', 'label']].to_csv("processed_news.csv", index=False)

print("✅ Text preprocessing complete!")
print(df[['clean_text', 'label']].head()) # ===============================================
# ⚙️ Stage 5 — Text Vectorization (TF-IDF)
# ===============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# --- Load preprocessed dataset ---
df = pd.read_csv("/content/processed_news.csv")

# --- Handle missing values ---
df['clean_text'] = df['clean_text'].fillna('')

# --- Split data ---
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Save everything for next stage ---
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(X_train_tfidf, open("X_train.pkl", "wb"))
pickle.dump(X_test_tfidf, open("X_test.pkl", "wb"))
pickle.dump(y_train, open("y_train.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))

print("✅ TF-IDF vectorization complete!")
print(f"Training features: {X_train_tfidf.shape}")
print(f"Test features: {X_test_tfidf.shape}") # ===============================================
# 🧠 Stage 6 — Model Training & Evaluation
# ===============================================

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Load all data ---
X_train = pickle.load(open("X_train.pkl", "rb"))
X_test = pickle.load(open("X_test.pkl", "rb"))
y_train = pickle.load(open("y_train.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

# --- Train the Logistic Regression model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Make predictions ---
y_pred = model.predict(X_test)

# --- Evaluate performance ---
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model training complete!")
print(f"🎯 Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# --- Save trained model ---
pickle.dump(model, open("fake_news_model.pkl", "wb"))
print("\n💾 Model saved as 'fake_news_model.pkl'") # ===============================================
# 🌐 Stage 7 — Gradio Web App for Fake News Detection
# ===============================================

import gradio as gr
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load model and vectorizer ---
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# --- Define prediction function ---
def predict_news(text):
    if not text or text.strip() == "":
        return "Please enter some text."
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "📰 Real News ✅" if prediction == 1 else "🚨 Fake News ❌"

# --- Create Gradio Interface ---
demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Paste a news headline or paragraph here...",
        label="News Article / Headline"
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="🧠 Fake News Detector",
    description="Detect whether a news headline or article is real or fake using Machine Learning (TF-IDF + Logistic Regression)."
)

# --- Launch the app ---
demo.launch()
