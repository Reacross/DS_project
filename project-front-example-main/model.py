import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Sample data
data = {
    'comment': [
        'You are amazing!',
        'I hate you!',
        'You are the worst.',
        'I love this!',
        'This is bad.',
        'I feel great today!',
        'You are so dumb!'
    ],
    'toxic': [0, 1, 1, 0, 1, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a pipeline with TfidfVectorizer and Logistic Regression
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Fit the model
model.fit(df['comment'], df['toxic'])

# Save the model
joblib.dump(model, 'example.pkl')
