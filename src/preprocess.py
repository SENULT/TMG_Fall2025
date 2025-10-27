"""
Vietnamese Text Preprocessing Module
TMG Fall 2025 - Text Mining & Generation

Features:
- Lowercase normalization
- Keep Vietnamese tones
- URL/Number/Emoji mapping
- Punctuation normalization
- Teencode handling
- Sentence splitting
- Deduplication (~95% threshold)
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VietnameseTextPreprocessor:
    """Preprocessing pipeline for Vietnamese text"""
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Dictionary containing preprocessing settings
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        
        # Load teencode dictionary
        teencode_path = self.preprocessing_config.get('teencode_dict_path', 'configs/teencode_dict.json')
        self.teencode_dict = self._load_teencode_dict(teencode_path)
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info("VietnameseTextPreprocessor initialized")
    
    def _load_teencode_dict(self, path: str) -> Dict[str, str]:
        """Load teencode dictionary from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Teencode dictionary not found at {path}. Using empty dict.")
            return {}
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|(?:[a-zA-Z0-9-]+\.)+(?:com|vn|net|org|edu|gov|mil|int|info|biz|name|museum|coop|aero|asia|jobs|mobi|travel|xxx)'
        )
        
        # Number pattern
        self.number_pattern = re.compile(r'\b\d+(?:[.,]\d+)*\b')
        
        # Emoji pattern (Unicode emoji ranges)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FA6F"  # extended symbols
            "]+", 
            flags=re.UNICODE
        )
        
        # Punctuation pattern for normalization
        self.multiple_punct_pattern = re.compile(r'([!?.,;:])(\1{2,})')
        
        # Multiple spaces pattern
        self.multiple_space_pattern = re.compile(r'\s+')
        
        # Vietnamese sentence splitting pattern
        self.sentence_split_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ])')
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation marks
        - Limit consecutive punctuation to max 3
        - Normalize common Vietnamese punctuation
        """
        if not self.preprocessing_config.get('normalize_punctuation', True):
            return text
        
        max_consecutive = self.preprocessing_config.get('max_consecutive_punct', 3)
        
        # Replace multiple consecutive punctuation
        text = self.multiple_punct_pattern.sub(lambda m: m.group(1) * min(len(m.group(2)) + 1, max_consecutive), text)
        
        # Normalize Vietnamese punctuation
        text = text.replace('…', '...')
        text = text.replace('–', '-')
        text = text.replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def map_special_tokens(self, text: str) -> str:
        """
        Map URLs, numbers, and emojis to special tokens
        """
        # Map URLs
        if self.preprocessing_config.get('map_urls', True):
            url_token = self.preprocessing_config.get('url_token', '<URL>')
            text = self.url_pattern.sub(url_token, text)
        
        # Map numbers
        if self.preprocessing_config.get('map_numbers', True):
            number_token = self.preprocessing_config.get('number_token', '<NUM>')
            text = self.number_pattern.sub(number_token, text)
        
        # Map emojis
        if self.preprocessing_config.get('map_emojis', True):
            emoji_token = self.preprocessing_config.get('emoji_token', '<EMOJI>')
            text = self.emoji_pattern.sub(emoji_token, text)
        
        return text
    
    def handle_teencode(self, text: str) -> str:
        """
        Replace teencode words with standard Vietnamese
        Uses word boundary matching to avoid partial replacements
        """
        if not self.preprocessing_config.get('handle_teencode', True):
            return text
        
        if not self.teencode_dict:
            return text
        
        # Split into words
        words = text.split()
        
        # Replace teencode words
        processed_words = []
        for word in words:
            # Check if word (lowercase) is in teencode dict
            lower_word = word.lower()
            if lower_word in self.teencode_dict:
                processed_words.append(self.teencode_dict[lower_word])
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences (for error analysis)
        Uses Vietnamese sentence boundaries
        """
        if not self.preprocessing_config.get('split_sentences', True):
            return [text]
        
        sentences = self.sentence_split_pattern.split(text)
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def normalize_text(self, text: str) -> str:
        """
        Main text normalization function
        - Lowercase (optional)
        - Keep Vietnamese tones
        - Normalize whitespace
        """
        # Lowercase
        if self.preprocessing_config.get('lowercase', True):
            text = text.lower()
        
        # Normalize whitespace
        text = self.multiple_space_pattern.sub(' ', text)
        text = text.strip()
        
        return text
    
    def preprocess_text(self, text: str, keep_sentences: bool = False) -> Tuple[str, Optional[List[str]]]:
        """
        Full preprocessing pipeline for a single text
        
        Args:
            text: Input text
            keep_sentences: Whether to return sentence splits
            
        Returns:
            Tuple of (processed_text, sentences_list or None)
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Step 1: Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Step 2: Map special tokens (URL, NUM, EMOJI)
        text = self.map_special_tokens(text)
        
        # Step 3: Handle teencode
        text = self.handle_teencode(text)
        
        # Step 4: Normalize text (lowercase, whitespace)
        text = self.normalize_text(text)
        
        # Step 5: Split sentences if needed
        sentences = None
        if keep_sentences:
            sentences = self.split_sentences(text)
            # Rejoin sentences with separator
            sep_token = self.preprocessing_config.get('sentence_sep_token', ' [SEP] ')
            text = sep_token.join(sentences)
        
        return text, sentences
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using SequenceMatcher
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def deduplicate(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Deduplicate dataframe based on text similarity
        Uses fuzzy matching with ~95% threshold
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            
        Returns:
            Deduplicated dataframe
        """
        if not self.preprocessing_config.get('dedup', True):
            return df
        
        threshold = self.preprocessing_config.get('dedup_threshold', 0.95)
        method = self.preprocessing_config.get('dedup_method', 'fuzzy')
        
        logger.info(f"Starting deduplication with method={method}, threshold={threshold}")
        
        if method == 'exact':
            # Simple exact match deduplication
            df_dedup = df.drop_duplicates(subset=[text_column])
        
        elif method == 'fuzzy':
            # Fuzzy deduplication
            texts = df[text_column].tolist()
            keep_indices = []
            seen_texts = []
            
            for idx, text in enumerate(texts):
                # Check similarity with all seen texts
                is_duplicate = False
                for seen_text in seen_texts:
                    if self.calculate_similarity(text, seen_text) >= threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    keep_indices.append(idx)
                    seen_texts.append(text)
                
                # Log progress
                if (idx + 1) % 1000 == 0:
                    logger.info(f"Processed {idx + 1}/{len(texts)} texts")
            
            df_dedup = df.iloc[keep_indices].reset_index(drop=True)
        
        else:
            logger.warning(f"Unknown dedup method: {method}. Skipping deduplication.")
            df_dedup = df
        
        logger.info(f"Deduplication: {len(df)} -> {len(df_dedup)} ({len(df) - len(df_dedup)} removed)")
        
        return df_dedup
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocess entire dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column to preprocess
            
        Returns:
            Preprocessed dataframe
        """
        logger.info(f"Preprocessing {len(df)} texts...")
        
        # Make a copy
        df = df.copy()
        
        # Preprocess texts
        processed_texts = []
        all_sentences = []
        
        for idx, text in enumerate(df[text_column]):
            processed_text, sentences = self.preprocess_text(text, keep_sentences=True)
            processed_texts.append(processed_text)
            all_sentences.append(sentences)
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Preprocessed {idx + 1}/{len(df)} texts")
        
        # Update dataframe
        df[f'{text_column}_processed'] = processed_texts
        df['sentences'] = all_sentences
        df['num_sentences'] = [len(s) if s else 0 for s in all_sentences]
        
        # Calculate text statistics
        df['text_length'] = df[f'{text_column}_processed'].str.len()
        df['num_words'] = df[f'{text_column}_processed'].str.split().str.len()
        
        # Filter by length
        min_length = self.preprocessing_config.get('min_length', 10)
        max_length = self.preprocessing_config.get('max_length', 5000)
        
        before_filter = len(df)
        df = df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]
        logger.info(f"Length filter: {before_filter} -> {len(df)} ({before_filter - len(df)} removed)")
        
        # Deduplicate
        df = self.deduplicate(df, f'{text_column}_processed')
        
        logger.info(f"Preprocessing complete. Final: {len(df)} texts")
        
        return df


class RobustnessChecker:
    """Check model robustness through augmentation tests"""
    
    def __init__(self, config: Dict):
        self.config = config.get('robustness', {})
    
    def replace_special_tokens(self, text: str, token_type: str = 'url') -> str:
        """Replace special tokens with placeholders"""
        if token_type == 'url':
            return re.sub(r'<URL>', 'example.com', text)
        elif token_type == 'number':
            return re.sub(r'<NUM>', '123', text)
        elif token_type == 'emoji':
            return re.sub(r'<EMOJI>', '😊', text)
        return text
    
    def permute_punctuation(self, text: str) -> List[str]:
        """Generate variations with different punctuation"""
        variations = [
            text,
            re.sub(r'[!?.,;:]', '', text),  # Remove all punctuation
            re.sub(r'\.', '!', text),  # Replace periods with exclamation
            re.sub(r'[!?]', '.', text),  # Normalize to periods
        ]
        return list(set(variations))
    
    def check_class_distribution(self, df: pd.DataFrame, label_col: str) -> Dict:
        """Check for class imbalance and small classes"""
        distribution = df[label_col].value_counts(normalize=True)
        
        threshold = self.config.get('min_class_ratio', 0.01)
        small_classes = distribution[distribution < threshold]
        
        return {
            'distribution': distribution.to_dict(),
            'small_classes': small_classes.to_dict(),
            'num_small_classes': len(small_classes)
        }


def load_config(config_path: str = 'configs/config.yaml') -> Dict:
    """Load configuration from YAML file"""
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    # Example usage
    config = load_config()
    preprocessor = VietnameseTextPreprocessor(config)
    
    # Test text
    test_text = "Xin chào mọi ng! Tôi đang học ở https://fpt.edu.vn. Điểm của em là 9.5 😊 K biết ntn nhỉ???"
    
    processed, sentences = preprocessor.preprocess_text(test_text, keep_sentences=True)
    
    print("Original:", test_text)
    print("Processed:", processed)
    print("Sentences:", sentences)
