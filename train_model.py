import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load datasets
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

# Add labels
fake['label'] = 0
true['label'] = 1

# Combine datasets
data = pd.concat([fake, true])

# Features and labels
X = data['text']
y = data['label']

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model trained and saved successfully!")