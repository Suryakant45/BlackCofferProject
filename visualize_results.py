#!/usr/bin/env python3
"""
Visualization script for BlackCoffer Text Analysis Results
Creates comprehensive visualizations and insights from the analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_and_validate_data():
    """Load and validate the analysis results"""
    if not os.path.exists('text_analysis_results.csv'):
        print("❌ Error: text_analysis_results.csv not found!")
        print("Please run the text analysis first.")
        return None
    
    try:
        df = pd.read_csv('text_analysis_results.csv')
        print(f"✅ Loaded analysis results for {len(df)} documents")
        return df
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def create_comprehensive_visualization(df):
    """Create comprehensive visualization of text analysis results"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Sentiment Distribution (Polarity)
    plt.subplot(3, 4, 1)
    plt.hist(df['polarity_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Polarity Scores', fontweight='bold')
    plt.xlabel('Polarity Score')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
    plt.legend()
    
    # 2. Subjectivity Distribution
    plt.subplot(3, 4, 2)
    plt.hist(df['subjectivity_score'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Subjectivity Scores', fontweight='bold')
    plt.xlabel('Subjectivity Score')
    plt.ylabel('Frequency')
    
    # 3. Word Count Distribution
    plt.subplot(3, 4, 3)
    plt.hist(df['word_count'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribution of Word Counts', fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    # 4. Fog Index Distribution
    plt.subplot(3, 4, 4)
    plt.hist(df['fog_index'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Distribution of Fog Index (Readability)', fontweight='bold')
    plt.xlabel('Fog Index')
    plt.ylabel('Frequency')
    
    # 5. Sentiment Categories Pie Chart
    plt.subplot(3, 4, 5)
    positive_docs = len(df[df['polarity_score'] > 0.1])
    negative_docs = len(df[df['polarity_score'] < -0.1])
    neutral_docs = len(df) - positive_docs - negative_docs
    
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive_docs, negative_docs, neutral_docs]
    colors = ['lightgreen', 'lightcoral', 'lightgray']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Distribution', fontweight='bold')
    
    # 6. Polarity vs Subjectivity Scatter
    plt.subplot(3, 4, 6)
    plt.scatter(df['polarity_score'], df['subjectivity_score'], alpha=0.6, color='red', s=30)
    plt.title('Polarity vs Subjectivity', fontweight='bold')
    plt.xlabel('Polarity Score')
    plt.ylabel('Subjectivity Score')
    plt.grid(True, alpha=0.3)
    
    # 7. Complex Words Percentage
    plt.subplot(3, 4, 7)
    plt.hist(df['percentage_complex_words'], bins=30, alpha=0.7, color='brown', edgecolor='black')
    plt.title('Distribution of Complex Words %', fontweight='bold')
    plt.xlabel('Percentage of Complex Words')
    plt.ylabel('Frequency')
    
    # 8. Readability Categories
    plt.subplot(3, 4, 8)
    easy_docs = len(df[df['fog_index'] < 12])
    medium_docs = len(df[(df['fog_index'] >= 12) & (df['fog_index'] < 16)])
    hard_docs = len(df[df['fog_index'] >= 16])
    
    categories = ['Easy\n(<12)', 'Medium\n(12-16)', 'Hard\n(≥16)']
    counts = [easy_docs, medium_docs, hard_docs]
    colors = ['lightgreen', 'yellow', 'lightcoral']
    
    plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Document Readability Categories', fontweight='bold')
    plt.ylabel('Number of Documents')
    
    # 9. Average Sentence Length vs Fog Index
    plt.subplot(3, 4, 9)
    plt.scatter(df['avg_sentence_length'], df['fog_index'], alpha=0.6, color='blue', s=30)
    plt.title('Sentence Length vs Readability', fontweight='bold')
    plt.xlabel('Average Sentence Length')
    plt.ylabel('Fog Index')
    plt.grid(True, alpha=0.3)
    
    # 10. Personal Pronouns Distribution
    plt.subplot(3, 4, 10)
    plt.hist(df['personal_pronoun_count'], bins=20, alpha=0.7, color='pink', edgecolor='black')
    plt.title('Distribution of Personal Pronouns', fontweight='bold')
    plt.xlabel('Personal Pronoun Count')
    plt.ylabel('Frequency')
    
    # 11. Positive vs Negative Words
    plt.subplot(3, 4, 11)
    plt.scatter(df['positive_score'], df['negative_score'], alpha=0.6, color='green', s=30)
    plt.title('Positive vs Negative Words', fontweight='bold')
    plt.xlabel('Positive Word Count')
    plt.ylabel('Negative Word Count')
    plt.grid(True, alpha=0.3)
    
    # 12. Word Count vs Complexity
    plt.subplot(3, 4, 12)
    plt.scatter(df['word_count'], df['percentage_complex_words'], alpha=0.6, color='purple', s=30)
    plt.title('Document Length vs Complexity', fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Percentage Complex Words')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('BlackCoffer Text Analysis - Comprehensive Results', fontsize=20, fontweight='bold', y=0.98)
    
    # Save the plot
    plt.savefig('comprehensive_text_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comprehensive visualization saved as 'comprehensive_text_analysis.png'")

def generate_detailed_insights(df):
    """Generate detailed insights from the analysis"""
    
    print("\n" + "="*80)
    print("🔍 DETAILED TEXT ANALYSIS INSIGHTS")
    print("="*80)
    
    # Basic Statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Total documents analyzed: {len(df)}")
    print(f"   • Average word count: {df['word_count'].mean():.1f} words")
    print(f"   • Average sentence count: {df['sentence_count'].mean():.1f} sentences")
    print(f"   • Average document length: {df['avg_word_length'].mean():.1f} characters per word")
    
    # Sentiment Analysis
    positive_docs = len(df[df['polarity_score'] > 0.1])
    negative_docs = len(df[df['polarity_score'] < -0.1])
    neutral_docs = len(df) - positive_docs - negative_docs
    
    print(f"\n😊 SENTIMENT ANALYSIS:")
    print(f"   • Positive documents: {positive_docs} ({positive_docs/len(df)*100:.1f}%)")
    print(f"   • Negative documents: {negative_docs} ({negative_docs/len(df)*100:.1f}%)")
    print(f"   • Neutral documents: {neutral_docs} ({neutral_docs/len(df)*100:.1f}%)")
    print(f"   • Average polarity: {df['polarity_score'].mean():.3f}")
    print(f"   • Average subjectivity: {df['subjectivity_score'].mean():.3f}")
    
    # Readability Analysis
    easy_docs = len(df[df['fog_index'] < 12])
    medium_docs = len(df[(df['fog_index'] >= 12) & (df['fog_index'] < 16)])
    hard_docs = len(df[df['fog_index'] >= 16])
    
    print(f"\n📖 READABILITY ANALYSIS:")
    print(f"   • Easy to read (Fog < 12): {easy_docs} ({easy_docs/len(df)*100:.1f}%)")
    print(f"   • Medium difficulty (12 ≤ Fog < 16): {medium_docs} ({medium_docs/len(df)*100:.1f}%)")
    print(f"   • Hard to read (Fog ≥ 16): {hard_docs} ({hard_docs/len(df)*100:.1f}%)")
    print(f"   • Average Fog Index: {df['fog_index'].mean():.1f}")
    print(f"   • Average sentence length: {df['avg_sentence_length'].mean():.1f} words")
    print(f"   • Average complex words: {df['percentage_complex_words'].mean():.1f}%")
    
    # Extreme Cases
    most_positive = df.loc[df['polarity_score'].idxmax()]
    most_negative = df.loc[df['polarity_score'].idxmin()]
    most_complex = df.loc[df['fog_index'].idxmax()]
    longest_doc = df.loc[df['word_count'].idxmax()]
    most_subjective = df.loc[df['subjectivity_score'].idxmax()]
    
    print(f"\n🏆 EXTREME CASES:")
    print(f"   • Most positive document: #{most_positive['document_id']} (polarity: {most_positive['polarity_score']:.3f})")
    print(f"   • Most negative document: #{most_negative['document_id']} (polarity: {most_negative['polarity_score']:.3f})")
    print(f"   • Most complex document: #{most_complex['document_id']} (Fog Index: {most_complex['fog_index']:.1f})")
    print(f"   • Longest document: #{longest_doc['document_id']} ({longest_doc['word_count']} words)")
    print(f"   • Most subjective document: #{most_subjective['document_id']} (subjectivity: {most_subjective['subjectivity_score']:.3f})")
    
    # Correlations
    print(f"\n🔗 INTERESTING CORRELATIONS:")
    polarity_subjectivity_corr = df['polarity_score'].corr(df['subjectivity_score'])
    length_complexity_corr = df['word_count'].corr(df['percentage_complex_words'])
    fog_sentence_corr = df['fog_index'].corr(df['avg_sentence_length'])
    
    print(f"   • Polarity vs Subjectivity: {polarity_subjectivity_corr:.3f}")
    print(f"   • Document length vs Complexity: {length_complexity_corr:.3f}")
    print(f"   • Fog Index vs Sentence length: {fog_sentence_corr:.3f}")
    
    # Summary Statistics Table
    print(f"\n📋 SUMMARY STATISTICS:")
    print("="*60)
    summary_stats = df[['polarity_score', 'subjectivity_score', 'word_count', 
                       'fog_index', 'percentage_complex_words']].describe()
    print(summary_stats.round(3))

def main():
    """Main function"""
    print("📊 BlackCoffer Text Analysis - Results Visualization")
    print("="*60)
    
    # Load data
    df = load_and_validate_data()
    if df is None:
        return
    
    # Create visualizations
    print("🎨 Creating comprehensive visualizations...")
    create_comprehensive_visualization(df)
    
    # Generate insights
    generate_detailed_insights(df)
    
    print(f"\n✅ Analysis complete!")
    print("📁 Files generated:")
    print("   • comprehensive_text_analysis.png")
    print("   • text_analysis_results.csv")

if __name__ == "__main__":
    main()
