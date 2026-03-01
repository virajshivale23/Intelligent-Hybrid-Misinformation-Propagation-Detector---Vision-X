import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources (only first time)
nltk.download("stopwords")
nltk.download("wordnet")

print("Loading dataset...")

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Take balanced subset
fake = fake.sample(5000, random_state=42)
true = true.sample(5000, random_state=42)

fake["label"] = 1   # Fake
true["label"] = 0   # Real

df = pd.concat([fake, true]).sample(frac=1, random_state=42)

# ----------------------------
# TEXT PREPROCESSING PIPELINE
# ----------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

print("Preprocessing text...")

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

# ----------------------------
# TF-IDF VECTORIZATION
# ----------------------------

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_df=0.7,
    min_df=5,
    ngram_range=(1, 2)  # Unigram + Bigram
)

X_vec = vectorizer.fit_transform(X)

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ----------------------------
# MODEL TRAINING
# ----------------------------

print("Training model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# SAVE MODEL
# ----------------------------

print("Saving model...")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel trained and saved successfully.")