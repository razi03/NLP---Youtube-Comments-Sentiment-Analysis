# YouTube Comments Sentiment Analysis

A simple, beginner-friendly sentiment analysis project for YouTube comments that classifies comments as either **positive** or **negative** using basic NLP and machine learning tools.

## 🎯 Project Goal

Classify YouTube comments as either `positive` or `negative` using simple and lightweight tools, similar to an IMDB movie reviews sentiment analysis project.

## 🛠️ Technologies Used

- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **NLP**: `nltk`, `spacy`
- **Machine Learning**: `scikit-learn`

## 📁 Project Structure

```
youtube-comments-sentiment-analysis/
├── youtube_sentiment.py          # Main Python script
├── YoutubeCommentsDataSet.csv    # Original dataset with comments and labels
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.7+ installed.

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   ```bash
   python youtube_sentiment.py
   ```

## 📊 What the Script Does

The script performs a complete sentiment analysis pipeline:

1. **Data Loading**: Loads the YouTube comments dataset
2. **Data Inspection**: Shows basic statistics and class distribution
3. **Text Preprocessing**: 
   - Lowercasing
   - Removing punctuation
   - Removing stopwords
   - Lemmatization using SpaCy
4. **Feature Extraction**: TF-IDF vectorization
5. **Model Training**: Logistic Regression classifier
6. **Evaluation**: Accuracy, confusion matrix, classification report
7. **Feature Analysis**: Visualizes important words for each sentiment
8. **Testing**: Tests the model on new sample comments

## 📦 Dataset

The `YoutubeCommentsDataSet.csv` file contains:
- **Comment**: The YouTube comment text
- **Sentiment**: Label (`positive`, `negative`, or `neutral`)

The script automatically filters to binary classification (positive/negative only) for sentiment analysis.

## 📈 Expected Output

The script will show:
- Dataset statistics and visualizations
- Text preprocessing examples
- Model performance metrics
- Confusion matrix plot
- Feature importance plots
- Predictions on test comments

## 🎓 Educational Value

This project demonstrates:
- Basic NLP preprocessing techniques
- Text vectorization with TF-IDF
- Simple machine learning classification
- Model evaluation and interpretation
- Feature importance analysis

## 🔧 Customization

You can easily modify the script to:
- Use your own dataset
- Try different preprocessing steps
- Experiment with other ML algorithms
- Adjust model parameters

## 📝 Notes

- The script automatically downloads required NLTK and SpaCy data
- All plots are displayed using matplotlib
- The model achieves good accuracy on the balanced dataset
- The code is well-commented for educational purposes

## 🤝 Contributing

Feel free to improve the code, add new features, or suggest enhancements!
