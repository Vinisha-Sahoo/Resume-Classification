import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load your CSV
df = pd.read_csv("resume_dataset_labeled.csv")
df = df[df['content'] != "Invalid Resume Content"]
df.dropna(subset=['content', 'role'], inplace=True)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['cleaned_text'] = df['content'].apply(clean_text)

# Encode labels
le = LabelEncoder()
df['role_encoded'] = le.fit_transform(df['role'])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['role_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and export models
models = {
    "logistic_model.pkl": LogisticRegression(max_iter=1000),
    "naivebayes_model.pkl": MultinomialNB(),
    "randomforest_model.pkl": RandomForestClassifier(),
    "svm_model.pkl": SVC(probability=True),
    "knn_model.pkl": KNeighborsClassifier()
}

for filename, clf in models.items():
    clf.fit(X_train, y_train)
    joblib.dump(clf, filename)

# Save vectorizer and label encoder
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ All models trained and saved.")
