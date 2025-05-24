#!/usr/bin/env python3
"""
BlackCoffer Text Analysis Script
Comprehensive text analysis including sentiment analysis, readability metrics, and text statistics
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("Warning: Could not download NLTK data. Some features may not work properly.")

class TextAnalyzer:
    def __init__(self):
        """Initialize the TextAnalyzer with required dictionaries and settings"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            print("Warning: Could not load stopwords. Using empty set.")
        
        self.personal_pronouns = {
            'i', 'we', 'my', 'ours', 'us', 'me', 'our', 'you', 'your', 'yours'
        }
        
        # Load sentiment dictionaries
        self.positive_words = {
            'good', 'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'awesome', 
            'outstanding', 'brilliant', 'superb', 'magnificent', 'marvelous', 'terrific',
            'perfect', 'beautiful', 'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied',
            'delighted', 'thrilled', 'excited', 'positive', 'successful', 'effective',
            'efficient', 'innovative', 'creative', 'impressive', 'remarkable', 'exceptional',
            'superior', 'splendid', 'fabulous', 'incredible', 'phenomenal', 'spectacular'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
            'poor', 'worst', 'disappointing', 'frustrating', 'annoying', 'boring',
            'sad', 'angry', 'upset', 'worried', 'concerned', 'problem', 'issue',
            'difficulty', 'challenge', 'failure', 'error', 'mistake', 'wrong',
            'negative', 'ineffective', 'inefficient', 'useless', 'worthless',
            'inferior', 'dreadful', 'appalling', 'atrocious', 'deplorable'
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove special characters and digits, keep only alphabets and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespaces
        text = ' '.join(text.split())
        return text
    
    def count_syllables(self, word):
        """Count syllables in a word using a simple heuristic"""
        if not word:
            return 0
        
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Every word has at least one syllable
        return max(1, syllable_count)
    
    def is_complex_word(self, word):
        """Check if word is complex (more than 2 syllables)"""
        return self.count_syllables(word) > 2
    
    def sentiment_analysis(self, text):
        """Perform sentiment analysis"""
        if not text:
            return {
                'positive_score': 0,
                'negative_score': 0,
                'polarity_score': 0,
                'subjectivity_score': 0
            }
        
        # Clean text for analysis
        cleaned_text = self.clean_text(text)
        words = word_tokenize(cleaned_text)
        
        # Count positive and negative words
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        
        # Calculate polarity score
        total_sentiment_words = positive_score + negative_score
        if total_sentiment_words > 0:
            polarity_score = (positive_score - negative_score) / total_sentiment_words
        else:
            polarity_score = 0
        
        # Calculate subjectivity score using TextBlob
        try:
            blob = TextBlob(text)
            subjectivity_score = blob.sentiment.subjectivity
        except:
            subjectivity_score = 0
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'polarity_score': polarity_score,
            'subjectivity_score': subjectivity_score
        }
    
    def readability_analysis(self, text):
        """Calculate readability metrics"""
        if not text:
            return self._empty_readability_results()
        
        # Tokenize sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(self.clean_text(text))
        
        # Filter out stop words and empty strings
        words = [word for word in words if word and word not in self.stop_words]
        
        # Basic counts
        word_count = len(words)
        sentence_count = len(sentences)
        
        if word_count == 0 or sentence_count == 0:
            return self._empty_readability_results()
        
        # Calculate syllables and complex words
        total_syllables = sum(self.count_syllables(word) for word in words)
        complex_words = [word for word in words if self.is_complex_word(word)]
        complex_word_count = len(complex_words)
        
        # Calculate metrics
        avg_sentence_length = word_count / sentence_count
        percentage_complex_words = (complex_word_count / word_count * 100)
        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
        
        # Count personal pronouns
        personal_pronoun_count = sum(1 for word in word_tokenize(text.lower()) 
                                   if word in self.personal_pronouns)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'percentage_complex_words': percentage_complex_words,
            'fog_index': fog_index,
            'avg_words_per_sentence': avg_sentence_length,
            'complex_word_count': complex_word_count,
            'syllable_count': total_syllables,
            'personal_pronoun_count': personal_pronoun_count,
            'avg_word_length': avg_word_length
        }
    
    def _empty_readability_results(self):
        """Return empty readability results for edge cases"""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'percentage_complex_words': 0,
            'fog_index': 0,
            'avg_words_per_sentence': 0,
            'complex_word_count': 0,
            'syllable_count': 0,
            'personal_pronoun_count': 0,
            'avg_word_length': 0
        }
    
    def analyze_text(self, text):
        """Perform comprehensive text analysis"""
        sentiment_results = self.sentiment_analysis(text)
        readability_results = self.readability_analysis(text)
        
        # Combine all results
        results = {**sentiment_results, **readability_results}
        return results

def main():
    """Main function to run the text analysis"""
    print("🚀 BlackCoffer Text Analysis Tool")
    print("=" * 50)
    
    # Check if combined_data.csv exists
    if not os.path.exists('combined_data.csv'):
        print("❌ Error: combined_data.csv not found!")
        print("Please run the data extraction part first.")
        return
    
    # Initialize analyzer
    print("📊 Initializing Text Analyzer...")
    analyzer = TextAnalyzer()
    
    # Load data
    print("📁 Loading combined data...")
    try:
        df_combined = pd.read_csv('combined_data.csv')
        print(f"✅ Loaded {len(df_combined)} text documents for analysis")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Analyze texts
    print("🔍 Analyzing texts...")
    analysis_results = []
    
    for idx, row in df_combined.iterrows():
        text = row['content']
        results = analyzer.analyze_text(text)
        results['document_id'] = idx + 1
        analysis_results.append(results)
        
        if (idx + 1) % 10 == 0:
            print(f"   Analyzed {idx + 1}/{len(df_combined)} documents...")
    
    # Create results DataFrame
    print("📋 Creating results DataFrame...")
    results_df = pd.DataFrame(analysis_results)
    
    # Reorder columns
    column_order = [
        'document_id', 'positive_score', 'negative_score', 'polarity_score',
        'subjectivity_score', 'word_count', 'sentence_count', 'avg_sentence_length',
        'percentage_complex_words', 'fog_index', 'avg_words_per_sentence',
        'complex_word_count', 'syllable_count', 'personal_pronoun_count', 'avg_word_length'
    ]
    results_df = results_df[column_order]
    
    # Save results
    print("💾 Saving results...")
    results_df.to_csv('text_analysis_results.csv', index=False)
    print("✅ Results saved to 'text_analysis_results.csv'")
    
    # Display summary
    print("\n📈 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total documents analyzed: {len(results_df)}")
    print(f"Average polarity score: {results_df['polarity_score'].mean():.3f}")
    print(f"Average subjectivity score: {results_df['subjectivity_score'].mean():.3f}")
    print(f"Average word count: {results_df['word_count'].mean():.1f}")
    print(f"Average Fog Index: {results_df['fog_index'].mean():.1f}")
    
    # Sentiment distribution
    positive_docs = len(results_df[results_df['polarity_score'] > 0])
    negative_docs = len(results_df[results_df['polarity_score'] < 0])
    neutral_docs = len(results_df[results_df['polarity_score'] == 0])
    
    print(f"\n📊 Sentiment Distribution:")
    print(f"   Positive: {positive_docs} ({positive_docs/len(results_df)*100:.1f}%)")
    print(f"   Negative: {negative_docs} ({negative_docs/len(results_df)*100:.1f}%)")
    print(f"   Neutral: {neutral_docs} ({neutral_docs/len(results_df)*100:.1f}%)")
    
    print("\n✅ Text analysis completed successfully!")
    print("📁 Check 'text_analysis_results.csv' for detailed results.")

if __name__ == "__main__":
    main()
