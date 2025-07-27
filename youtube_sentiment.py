#!/usr/bin/env python3
"""
YouTube Comments Sentiment Analysis
A simple, beginner-friendly sentiment analysis project for YouTube comments.

Goal: Classify comments as either positive or negative using basic NLP and machine learning tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import spacy
import sys
import subprocess
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("NLTK data downloaded successfully!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

def download_spacy_model():
    """Download required SpaCy model"""
    try:
        # Check if model is already installed
        nlp = spacy.load('en_core_web_sm')
        print("SpaCy model already installed!")
        return nlp
    except OSError:
        print("Downloading SpaCy model...")
        try:
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], 
                         check=True, capture_output=True)
            nlp = spacy.load('en_core_web_sm')
            print("SpaCy model downloaded successfully!")
            return nlp
        except Exception as e:
            print(f"Error downloading SpaCy model: {e}")
            return None

def preprocess_text(text, nlp, stop_words):
    """
    Preprocess text by:
    1. Lowercasing
    2. Removing punctuation
    3. Removing stopwords
    4. Lemmatization
    """
    # Handle NaN/None values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    doc = nlp(' '.join(tokens))
    lemmas = [token.lemma_ for token in doc]
    
    return ' '.join(lemmas)

def plot_top_features(classifier, vectorizer, class_labels, n=15):
    """
    Plot top features for each class based on logistic regression coefficients
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    for i, class_label in enumerate(class_labels):
        if class_label == 'positive':
            # Top positive features (highest coefficients)
            top_indices = np.argsort(classifier.coef_[0])[-n:]
            title = f'Top {n} Positive Features'
            color = 'lightgreen'
        else:
            # Top negative features (lowest coefficients)
            top_indices = np.argsort(classifier.coef_[0])[:n]
            title = f'Top {n} Negative Features'
            color = 'lightcoral'
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names[top_indices], classifier.coef_[0][top_indices], color=color)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

def main():
    print("=" * 60)
    print("YouTube Comments Sentiment Analysis")
    print("=" * 60)
    
    # Download required data
    print("\n1. Downloading required data...")
    download_nltk_data()
    nlp = download_spacy_model()
    if nlp is None:
        print("Failed to load SpaCy model. Exiting...")
        return
    
    stop_words = set(stopwords.words('english'))
    
    # Load and inspect dataset
    print("\n2. Loading and inspecting dataset...")
    try:
        # Use the user's original dataset
        df = pd.read_csv('YoutubeCommentsDataSet.csv')
        print(f"Dataset Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check for null values
        print(f"\nNull values:")
        print(df.isnull().sum())
        
        # Show class distribution
        print(f"\nClass distribution:")
        sentiment_counts = df['Sentiment'].value_counts()
        print(sentiment_counts)
        
        # Filter to only positive and negative (remove neutral for binary classification)
        print(f"\nFiltering to binary classification (positive/negative only)...")
        df_binary = df[df['Sentiment'].isin(['positive', 'negative'])].copy()
        print(f"After filtering: {df_binary.shape[0]} samples")
        
        # Remove rows with missing comments
        print(f"\nRemoving rows with missing comments...")
        df_binary = df_binary.dropna(subset=['Comment']).copy()
        print(f"After removing missing comments: {df_binary.shape[0]} samples")
        
        # Remove empty comments
        df_binary = df_binary[df_binary['Comment'].str.strip() != ''].copy()
        print(f"After removing empty comments: {df_binary.shape[0]} samples")
        
        sentiment_counts_binary = df_binary['Sentiment'].value_counts()
        print(f"Binary class distribution:")
        print(sentiment_counts_binary)
        
        # Check if we have enough samples for stratification
        min_samples_per_class = sentiment_counts_binary.min()
        if min_samples_per_class < 2:
            print(f"\nWarning: Class with only {min_samples_per_class} samples. Using non-stratified split.")
            use_stratify = False
        else:
            use_stratify = True
        
        # Visualize class distribution
        plt.figure(figsize=(10, 6))
        # Plot original distribution
        plt.subplot(1, 2, 1)
        sns.countplot(data=df, x='Sentiment', hue='Sentiment', legend=False, 
                     palette=['lightcoral', 'lightblue', 'lightgray'])
        plt.title('Original Sentiment Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        for i, v in enumerate(sentiment_counts.values):
            plt.text(i, v + max(sentiment_counts.values)*0.01, str(v), ha='center', fontweight='bold')
        
        # Plot binary distribution
        plt.subplot(1, 2, 2)
        sns.countplot(data=df_binary, x='Sentiment', hue='Sentiment', legend=False, 
                     palette=['lightcoral', 'lightblue'])
        plt.title('Binary Sentiment Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        for i, v in enumerate(sentiment_counts_binary.values):
            plt.text(i, v + max(sentiment_counts_binary.values)*0.01, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Use the binary dataset for the rest of the analysis
        df = df_binary
        
        # Check if we have enough data
        if df.shape[0] < 10:
            print("Error: Not enough data after cleaning. Need at least 10 samples.")
            return
        
    except FileNotFoundError:
        print("Error: YoutubeCommentsDataSet.csv not found!")
        return
    
    # Preprocess text
    print("\n3. Preprocessing text...")
    print("Sample of original vs cleaned comments:")
    print("=" * 80)
    
    sample_df = df.sample(5, random_state=42).copy()
    sample_df['cleaned'] = sample_df['Comment'].apply(lambda x: preprocess_text(x, nlp, stop_words))
    
    for idx, row in sample_df.iterrows():
        print(f"\nOriginal: {row['Comment']}")
        print(f"Cleaned:  {row['cleaned']}")
        print('-' * 80)
    
    # Apply preprocessing to all comments
    print("\nApplying preprocessing to all comments...")
    df['cleaned_comment'] = df['Comment'].apply(lambda x: preprocess_text(x, nlp, stop_words))
    print("Preprocessing complete!")
    
    # Vectorize text
    print("\n4. Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_comment'])
    y = df['Sentiment']
    
    print(f"TF-IDF matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Train/test split
    print("\n5. Splitting data into train/test sets...")
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Using stratified split to maintain class balance.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Using non-stratified split due to class imbalance.")
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"\nTraining set class distribution:")
    print(y_train.value_counts())
    print(f"\nTest set class distribution:")
    print(y_test.value_counts())
    
    # Train model
    print("\n6. Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    
    # Evaluate model
    print("\n7. Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['positive', 'negative'], 
                yticklabels=['positive', 'negative'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize features
    print("\n8. Visualizing top features...")
    plot_top_features(model, vectorizer, ['positive', 'negative'], n=15)
    
    # Test on new comments
    print("\n9. Testing on new comments...")
    test_comments = [
        "This video is absolutely amazing! I love it!",
        "Great content, very informative and well explained.",
        "This is terrible, I hate this video.",
        "The quality is poor and the information is wrong.",
        "Awesome tutorial, thanks for sharing!"
    ]
    
    # Preprocess the test comments
    cleaned_test = [preprocess_text(comment, nlp, stop_words) for comment in test_comments]
    
    # Vectorize
    X_test_new = vectorizer.transform(cleaned_test)
    
    # Predict
    predictions = model.predict(X_test_new)
    probabilities = model.predict_proba(X_test_new)
    
    print('Test Results:')
    print('=' * 60)
    for i, (comment, pred, prob) in enumerate(zip(test_comments, predictions, probabilities)):
        confidence = max(prob) * 100
        print(f"\nComment {i+1}: {comment}")
        print(f"Prediction: {pred} (confidence: {confidence:.1f}%)")
        print('-' * 60)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 