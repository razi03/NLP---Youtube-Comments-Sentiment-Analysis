# YouTube Comments Sentiment Analysis

A simple, beginner-friendly sentiment analysis project for YouTube comments that classifies comments as either **positive** or **negative** using basic NLP and machine learning tools.

## 🎯 Project Goal

Classify YouTube comments as either `positive` or `negative` using simple and lightweight tools, similar to an IMDB movie reviews sentiment analysis project.

## 🛠️ Technologies Used

- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **NLP**: `nltk`, `spacy`
- **Machine Learning**: `scikit-learn`
- **Environment**: Jupyter Notebook

## 📁 Project Structure

```
youtube-comments-sentiment-analysis/
├── youtube_sentiment_from_script.ipynb  # Main Jupyter notebook
├── YoutubeCommentsDataSet.csv           # Original dataset with comments and labels
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.7+ and Jupyter installed.

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**:
   - Navigate to `youtube_sentiment_from_script.ipynb`
   - Run all cells sequentially

## 📊 What the Notebook Does

The notebook performs a complete sentiment analysis pipeline:

1. **Setup & Installation**: Installs and downloads required libraries and models
2. **Data Loading**: Loads the YouTube comments dataset
3. **Data Inspection**: Shows basic statistics and class distribution
4. **Text Preprocessing**: 
   - Lowercasing
   - Removing punctuation
   - Removing stopwords
   - Lemmatization using SpaCy
5. **Feature Extraction**: TF-IDF vectorization
6. **Model Training**: Logistic Regression classifier
7. **Evaluation**: Accuracy, confusion matrix, classification report
8. **Testing**: Tests the model on new sample comments

## 📦 Dataset

The `YoutubeCommentsDataSet.csv` file contains:
- **Comment**: The YouTube comment text
- **Sentiment**: Label (`positive`, `negative`, or `neutral`)

The notebook automatically filters to binary classification (positive/negative only) for sentiment analysis.

## 📈 Expected Output

The notebook will show:
- Dataset statistics and visualizations
- Text preprocessing examples
- Model performance metrics (88.68% accuracy achieved)
- Confusion matrix plot
- Predictions on test comments

## 🎓 Educational Value

This project demonstrates:
- Basic NLP preprocessing techniques
- Text vectorization with TF-IDF
- Simple machine learning classification
- Model evaluation and interpretation
- Interactive Jupyter notebook workflow

## 🔧 Customization

You can easily modify the notebook to:
- Use your own dataset
- Try different preprocessing steps
- Experiment with other ML algorithms
- Adjust model parameters
- Add more visualizations

## 📝 Notes

- The notebook automatically downloads required NLTK and SpaCy data
- All plots are displayed inline in the notebook
- The model achieves 88.68% accuracy on the real YouTube comments dataset
- The code is well-commented for educational purposes
- Each cell can be run independently for experimentation

## 🚀 Performance Results

- **Dataset Size**: 18,408 total comments (filtered to 13,739 for binary classification)
- **Model Accuracy**: 88.68%
- **Training Set**: 10,991 samples
- **Test Set**: 2,748 samples
- **Class Distribution**: 11,402 positive, 2,337 negative

## 🤝 Contributing

Feel free to improve the code, add new features, or suggest enhancements!
