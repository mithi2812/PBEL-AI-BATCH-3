import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(words)

# Load data

df = pd.read_csv('reviews.csv')
print(df)
df['review'] = df['review'].apply(preprocess)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict function
def predict_sentiment(text):
    text = preprocess(text)
    vect = vectorizer.transform([text])
    return model.predict(vect)[0]

# Example
print("Sample prediction:", predict_sentiment("This product is fantastic!"))
