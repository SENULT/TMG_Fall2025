"""
Test Preprocessing Module
Quick test to verify preprocessing works correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocess import VietnameseTextPreprocessor, load_config


def test_preprocessing():
    """Test preprocessing on sample texts"""
    
    print("=" * 80)
    print("TESTING VIETNAMESE TEXT PREPROCESSOR")
    print("=" * 80)
    
    # Load config
    config = load_config('configs/config.yaml')
    
    # Initialize preprocessor
    preprocessor = VietnameseTextPreprocessor(config)
    
    # Test cases
    test_cases = [
        {
            'name': 'URL, Number, Emoji',
            'text': 'Xin chào! Tôi đang học tại https://fpt.edu.vn. Điểm: 9.5 😊'
        },
        {
            'name': 'Teencode',
            'text': 'K biết ntn nhỉ??? Em vs bạn đc học chung ko???'
        },
        {
            'name': 'Multiple sentences',
            'text': 'Câu thứ nhất. Câu thứ hai! Câu thứ ba?'
        },
        {
            'name': 'Punctuation normalization',
            'text': 'Thật vậy!!!!! Không thể tin được...... Sao lại thế??????'
        },
        {
            'name': 'Mixed case with Vietnamese tones',
            'text': 'Chào BẠN! Tôi LÀ SINH VIÊN FPT. Rất VUI được GẶP bạn.'
        },
        {
            'name': 'Complex example',
            'text': 'Hôm nay t đi học ở http://university.edu.vn, đc gặp 50 bạn mới 🎉! K biết ntn nx...'
        }
    ]
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'=' * 80}")
        
        original = test_case['text']
        processed, sentences = preprocessor.preprocess_text(original, keep_sentences=True)
        
        print(f"\n📝 Original:")
        print(f"   {original}")
        print(f"\n✅ Processed:")
        print(f"   {processed}")
        
        if sentences and len(sentences) > 1:
            print(f"\n📄 Sentences ({len(sentences)}):")
            for j, sent in enumerate(sentences, 1):
                print(f"   {j}. {sent}")
    
    # Test similarity calculation
    print(f"\n{'=' * 80}")
    print("TESTING SIMILARITY CALCULATION")
    print(f"{'=' * 80}\n")
    
    text1 = "Xin chào mọi người"
    text2 = "Xin chào mọi người"
    text3 = "Chào các bạn"
    
    sim12 = preprocessor.calculate_similarity(text1, text2)
    sim13 = preprocessor.calculate_similarity(text1, text3)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {sim12:.2%}\n")
    
    print(f"Text 1: {text1}")
    print(f"Text 3: {text3}")
    print(f"Similarity: {sim13:.2%}\n")
    
    # Test statistics
    print(f"{'=' * 80}")
    print("PREPROCESSING STATISTICS")
    print(f"{'=' * 80}\n")
    
    sample_text = "Đây là một đoạn văn mẫu để test. Nó có nhiều câu. Và nhiều từ khác nhau!"
    processed, sentences = preprocessor.preprocess_text(sample_text, keep_sentences=True)
    
    print(f"Original text: {sample_text}")
    print(f"Processed text: {processed}")
    print(f"\nStatistics:")
    print(f"  - Text length: {len(processed)} characters")
    print(f"  - Word count: {len(processed.split())} words")
    print(f"  - Sentence count: {len(sentences) if sentences else 0} sentences")
    
    print(f"\n{'=' * 80}")
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    try:
        test_preprocessing()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
