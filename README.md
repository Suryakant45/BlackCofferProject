# BlackCoffer Text Analysis Project

A comprehensive data extraction and text analysis solution that scrapes web content and performs detailed sentiment analysis, readability assessment, and text statistics.

## 🚀 Project Overview

This project consists of two main phases:
1. **Data Extraction**: Scraping text content from URLs and saving to individual files
2. **Text Analysis**: Comprehensive analysis including sentiment analysis, readability metrics, and text statistics

## 📁 Project Structure

```
BlackCoffer Project/
├── data_extraction&text_analysis.ipynb  # Main Jupyter notebook
├── text_analysis.py                     # Standalone Python script
├── requirements.txt                     # Project dependencies
├── Input.xlsx                          # Input URLs (if available)
├── combined_data.csv                   # Combined extracted text data
├── text_analysis_results.csv          # Analysis results
├── 1.txt to 147.txt                   # Individual extracted text files
└── README.md                          # This file
```

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download NLTK data** (if not automatically downloaded):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   ```

## 🎯 Features

### Data Extraction
- Web scraping from BlackCoffer insights URLs
- HTML content parsing using BeautifulSoup
- Text cleaning and preprocessing
- Individual file storage and combined CSV creation

### Text Analysis
- **Sentiment Analysis**:
  - Positive/Negative word counting
  - Polarity score calculation
  - Subjectivity analysis using TextBlob

- **Readability Metrics**:
  - Average sentence length
  - Percentage of complex words
  - Fog Index (readability score)
  - Syllable counting

- **Text Statistics**:
  - Word count (excluding stop words)
  - Sentence count
  - Complex word count
  - Personal pronoun count
  - Average word length

## 🚀 Usage

### Option 1: Using Jupyter Notebook
1. Open `data_extraction&text_analysis.ipynb`
2. Run all cells sequentially
3. The notebook will perform both data extraction and analysis

### Option 2: Using Python Script
```bash
python text_analysis.py
```

**Note**: The Python script requires `combined_data.csv` to exist (created by the data extraction phase).

## 📊 Output Files

1. **`combined_data.csv`**: All extracted text content in a single file
2. **`text_analysis_results.csv`**: Comprehensive analysis results with metrics:
   - Document ID
   - Sentiment scores (positive, negative, polarity, subjectivity)
   - Readability metrics (word count, sentence count, fog index, etc.)
   - Text statistics (complex words, syllables, pronouns, etc.)

## 📈 Analysis Metrics Explained

### Sentiment Analysis
- **Positive Score**: Count of positive words in the text
- **Negative Score**: Count of negative words in the text
- **Polarity Score**: Ranges from -1 (negative) to +1 (positive)
- **Subjectivity Score**: Ranges from 0 (objective) to 1 (subjective)

### Readability Metrics
- **Fog Index**: Readability score (lower = easier to read)
  - < 12: Easy to read
  - 12-16: Medium difficulty
  - > 16: Hard to read
- **Average Sentence Length**: Average words per sentence
- **Percentage Complex Words**: Words with more than 2 syllables

### Text Statistics
- **Word Count**: Total words excluding stop words
- **Complex Word Count**: Words with more than 2 syllables
- **Personal Pronoun Count**: Count of personal pronouns (I, we, you, etc.)

## 🔧 Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **requests**: HTTP library for web scraping
- **beautifulsoup4**: HTML parsing
- **nltk**: Natural language processing
- **textblob**: Sentiment analysis
- **matplotlib/seaborn**: Data visualization

### Text Processing Pipeline
1. **Text Cleaning**: Remove special characters, convert to lowercase
2. **Tokenization**: Split into words and sentences using NLTK
3. **Stop Word Removal**: Filter out common English stop words
4. **Syllable Counting**: Custom algorithm for syllable estimation
5. **Sentiment Analysis**: Dictionary-based approach with TextBlob integration

## 📋 Sample Results

The analysis provides insights such as:
- Overall sentiment distribution across documents
- Readability assessment of content
- Text complexity analysis
- Statistical summaries and visualizations

## 🤝 Contributing

This is a BlackCoffer project. For any modifications or improvements:
1. Follow the existing code structure
2. Maintain comprehensive documentation
3. Test thoroughly before deployment

## 📞 Contact

**BlackCoffer Team**
- Website: www.blackcoffer.com
- Email: ajay@blackcoffer.com
- Address: 4/2, E-Extension, Shaym Vihar Phase 1, New Delhi 110043

## 📄 License

This project is developed by BlackCoffer Team for text analysis and research purposes.