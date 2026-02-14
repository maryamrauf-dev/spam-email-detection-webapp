import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

def train_spam_model():
    """
    Loads the downloaded dataset from data/dataset/dataset.csv and trains a binary classifier.
    """
    dataset_path = os.path.join("data", "dataset", "dataset.csv")
    
    if not os.path.exists(dataset_path):
        return None, None, f"Dataset not found at {dataset_path}"

    try:
        # Load the local CSV
        df = pd.read_csv(dataset_path)
        
        # Check if 'text' and 'category' columns exist
        if 'text' not in df.columns or 'category' not in df.columns:
            # Try to handle different column names if necessary
            return None, None, "Required columns 'text' or 'category' not found in CSV."

        # Map to binary (Spam vs. Ham)
        # category == 'spam' -> 0 (Spam)
        # all others ('social_media', 'promotions', 'updates', etc.) -> 1 (Ham)
        df['label'] = df['category'].apply(lambda x: 0 if str(x).lower() == 'spam' else 1)
        
        # Clean data
        data = df[['text', 'label']].copy()
        data = data.dropna()
        
        X = data['text']
        Y = data['label'].astype('int')
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Feature Extraction
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, max_features=5000)
        X_train_features = vectorizer.fit_transform(X_train)
        
        # Model Training with balanced class weights
        model = LogisticRegression(class_weight='balanced', solver='liblinear')
        model.fit(X_train_features, Y_train)
        
        return vectorizer, model, None
        
    except Exception as e:
        return None, None, str(e)

def predict_message(message, vectorizer, model):
    """
    Predicts if a message is spam or ham given the trained vectorizer and model.
    """
    input_features = vectorizer.transform([message])
    prediction = model.predict(input_features)[0]
    
    # Get probability score for confidence
    probabilities = model.predict_proba(input_features)[0]
    confidence = probabilities[prediction] * 100
    
    label = "Ham" if prediction == 1 else "Spam"
    return label, confidence
