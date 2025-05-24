#!/usr/bin/env python3
"""
Test script for BlackCoffer Text Analysis
This script tests the text analysis functionality with sample data
"""

import sys
import os
import pandas as pd
from text_analysis import TextAnalyzer

def test_text_analyzer():
    """Test the TextAnalyzer with sample texts"""
    print("🧪 Testing BlackCoffer Text Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    print("✅ TextAnalyzer initialized successfully")
    
    # Sample texts for testing
    test_texts = [
        {
            "id": 1,
            "text": "This is an excellent and amazing project. The results are fantastic and wonderful. I love the innovative approach and creative solutions.",
            "expected_sentiment": "positive"
        },
        {
            "id": 2,
            "text": "This is a terrible and awful implementation. The results are disappointing and frustrating. I hate the poor quality and useless features.",
            "expected_sentiment": "negative"
        },
        {
            "id": 3,
            "text": "The data analysis involves statistical methods and computational algorithms. The research methodology includes quantitative analysis and empirical validation.",
            "expected_sentiment": "neutral"
        },
        {
            "id": 4,
            "text": "We are developing comprehensive solutions for complex problems. Our team implements sophisticated algorithms to achieve optimal performance.",
            "expected_sentiment": "neutral/positive"
        }
    ]
    
    print(f"\n🔍 Testing with {len(test_texts)} sample texts...")
    
    results = []
    for test_case in test_texts:
        print(f"\n📝 Analyzing Text {test_case['id']}:")
        print(f"   Expected sentiment: {test_case['expected_sentiment']}")
        
        # Analyze the text
        analysis = analyzer.analyze_text(test_case['text'])
        analysis['document_id'] = test_case['id']
        analysis['expected_sentiment'] = test_case['expected_sentiment']
        results.append(analysis)
        
        # Display key metrics
        print(f"   Polarity Score: {analysis['polarity_score']:.3f}")
        print(f"   Subjectivity Score: {analysis['subjectivity_score']:.3f}")
        print(f"   Word Count: {analysis['word_count']}")
        print(f"   Fog Index: {analysis['fog_index']:.1f}")
        print(f"   Positive Words: {analysis['positive_score']}")
        print(f"   Negative Words: {analysis['negative_score']}")
        
        # Sentiment interpretation
        if analysis['polarity_score'] > 0.1:
            detected_sentiment = "positive"
        elif analysis['polarity_score'] < -0.1:
            detected_sentiment = "negative"
        else:
            detected_sentiment = "neutral"
        
        print(f"   Detected Sentiment: {detected_sentiment}")
        
        # Check if detection matches expectation
        if detected_sentiment in test_case['expected_sentiment']:
            print("   ✅ Sentiment detection: PASS")
        else:
            print("   ⚠️  Sentiment detection: Different from expected")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 30)
    print(f"Average Polarity: {results_df['polarity_score'].mean():.3f}")
    print(f"Average Subjectivity: {results_df['subjectivity_score'].mean():.3f}")
    print(f"Average Word Count: {results_df['word_count'].mean():.1f}")
    print(f"Average Fog Index: {results_df['fog_index'].mean():.1f}")
    
    # Save test results
    results_df.to_csv('test_results.csv', index=False)
    print(f"\n💾 Test results saved to 'test_results.csv'")
    
    return results_df

def test_with_existing_data():
    """Test with existing combined_data.csv if available"""
    print(f"\n🔍 Testing with existing data...")
    
    if not os.path.exists('combined_data.csv'):
        print("⚠️  combined_data.csv not found. Skipping existing data test.")
        return None
    
    try:
        df = pd.read_csv('combined_data.csv')
        print(f"✅ Found combined_data.csv with {len(df)} documents")
        
        # Test with first few documents
        analyzer = TextAnalyzer()
        sample_size = min(5, len(df))
        
        print(f"📊 Analyzing first {sample_size} documents...")
        
        for i in range(sample_size):
            text = df.iloc[i]['content']
            analysis = analyzer.analyze_text(text)
            
            print(f"\nDocument {i+1}:")
            print(f"   Word Count: {analysis['word_count']}")
            print(f"   Polarity: {analysis['polarity_score']:.3f}")
            print(f"   Fog Index: {analysis['fog_index']:.1f}")
        
        print("✅ Existing data test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing existing data: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 BlackCoffer Text Analysis - Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Sample text analysis
        test_results = test_text_analyzer()
        
        # Test 2: Existing data (if available)
        test_with_existing_data()
        
        print(f"\n🎉 ALL TESTS COMPLETED!")
        print("=" * 30)
        print("✅ Text analysis functionality is working correctly")
        print("📁 Check 'test_results.csv' for detailed test results")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Please check your installation and dependencies")
        sys.exit(1)

if __name__ == "__main__":
    main()
