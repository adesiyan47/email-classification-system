import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
dataset = pd.read_csv('emails.csv')

# Create vectorizer and transform text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text'])

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, dataset['spam'], test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save vectorizer and model
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')

print("✅ Model and vectorizer saved!")
