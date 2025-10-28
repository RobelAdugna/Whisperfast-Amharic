"""Advanced Amharic Tokenizer for Whisper Fine-tuning"""

import re
from typing import List, Dict, Optional, Tuple
import unicodedata

class AmharicTokenizer:
    """Advanced tokenizer specifically designed for Amharic (Ge'ez script)"""
    
    # Comprehensive Ge'ez Unicode blocks
    GEEZ_SCRIPT_RANGES = [
        (0x1200, 0x137F),  # Ethiopic
        (0x1380, 0x139F),  # Ethiopic Supplement
        (0x2D80, 0x2DDF),  # Ethiopic Extended
        (0xAB00, 0xAB2F),  # Ethiopic Extended-A
    ]
    
    # Ge'ez syllabary structure (7 orders)
    GEEZ_ORDERS = ['ə', 'u', 'i', 'a', 'e', 'ɨ', 'o']
    
    # Special Ge'ez characters
    GEEZ_PUNCTUATION = {
        '።': '.',   # Ethiopic full stop
        '፣': ',',   # Ethiopic comma
        '፤': ';',   # Ethiopic semicolon  
        '፥': ':',   # Ethiopic colon
        '፦': '::',  # Ethiopic preface colon
        '፧': '?',   # Ethiopic question mark
        '፨': '¶',   # Ethiopic paragraph separator
    }
    
    # Ethiopic numerals (complete set)
    ETHIOPIC_NUMERALS = {
        '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
        '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10',
        '፳': '20', '፴': '30', '፵': '40', '፶': '50',
        '፷': '60', '፸': '70', '፹': '80', '፺': '90',
        '፻': '100', '፼': '10000'
    }
    
    # Common Amharic abbreviations
    ABBREVIATIONS = {
        'ዓ.ም': 'ዓመተ ምህረት',  # Anno Mundi (year)
        'ዶ/ር': 'ዶክተር',  # Doctor
        'ፕ/ር': 'ፕሮፌሰር',  # Professor
    }
    
    def __init__(self, normalize_punctuation: bool = True,
                 normalize_numerals: bool = True,
                 preserve_diacritics: bool = True):
        """
        Initialize Amharic tokenizer
        
        Args:
            normalize_punctuation: Convert Ethiopic to standard punctuation
            normalize_numerals: Convert Ethiopic to Arabic numerals
            preserve_diacritics: Keep gemination and vowel length markers
        """
        self.normalize_punctuation = normalize_punctuation
        self.normalize_numerals = normalize_numerals
        self.preserve_diacritics = preserve_diacritics
    
    def is_geez_char(self, char: str) -> bool:
        """Check if character is in Ge'ez script"""
        if not char:
            return False
        code = ord(char)
        return any(start <= code <= end for start, end in self.GEEZ_SCRIPT_RANGES)
    
    def normalize_text(self, text: str) -> str:
        """
        Comprehensive Amharic text normalization
        
        Args:
            text: Input Amharic text
        
        Returns:
            Normalized text
        """
        # Unicode normalization (NFC form for Ethiopic)
        text = unicodedata.normalize('NFC', text)
        
        # Expand abbreviations
        for abbr, full in self.ABBREVIATIONS.items():
            text = text.replace(abbr, full)
        
        # Normalize numerals
        if self.normalize_numerals:
            for ethiopic, arabic in self.ETHIOPIC_NUMERALS.items():
                text = text.replace(ethiopic, arabic)
        
        # Normalize punctuation
        if self.normalize_punctuation:
            for ethiopic, standard in self.GEEZ_PUNCTUATION.items():
                text = text.replace(ethiopic, standard)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle gemination marker (if preserving)
        if not self.preserve_diacritics:
            text = text.replace('ː', '')  # Remove gemination
        
        return text.strip()
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize Amharic text into words
        
        Note: Amharic doesn't use spaces consistently, especially
        in older texts. This handles both spaced and non-spaced text.
        """
        # First normalize
        text = self.normalize_text(text)
        
        # Split on whitespace and punctuation
        # Keep Ge'ez punctuation as separate tokens
        pattern = r'([።፣፤፥፦፧፨\s]+|[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F\s]+)'
        tokens = re.split(pattern, text)
        
        # Filter empty tokens
        tokens = [t.strip() for t in tokens if t.strip()]
        
        return tokens
    
    def get_syllable_breakdown(self, word: str) -> List[str]:
        """
        Break down Amharic word into syllables (Ge'ez characters)
        Useful for phonetic analysis and alignment
        """
        syllables = []
        for char in word:
            if self.is_geez_char(char):
                syllables.append(char)
            elif char.strip():  # Non-Geez, non-whitespace
                syllables.append(char)
        return syllables
    
    def detect_dialect_markers(self, text: str) -> Dict[str, bool]:
        """
        Detect potential dialect variations in Amharic text
        
        Returns:
            Dictionary of dialect markers found
        """
        markers = {
            'gonder': False,      # Northern dialects
            'shewa': False,       # Central dialects  
            'gojjam': False,      # Western dialects
            'wollo': False,       # Eastern dialects
            'harari': False,      # Eastern city dialect
        }
        
        # Simplified dialect detection based on character usage patterns
        # Note: This is approximate - actual dialect detection requires
        # phonetic and morphological analysis
        
        # Gonder: More use of ቀ series
        if 'ቀ' in text or 'ቁ' in text or 'ቂ' in text:
            markers['gonder'] = True
        
        # Shewa: Standard Amharic (baseline)
        markers['shewa'] = True  # Default
        
        return markers
    
    def detect_code_switching(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect code-switching between Amharic and other scripts
        (commonly English in modern Amharic text)
        
        Returns:
            (has_code_switching, list of non-Amharic segments)
        """
        non_amharic_segments = []
        current_segment = []
        in_non_amharic = False
        
        for char in text:
            if self.is_geez_char(char):
                if in_non_amharic and current_segment:
                    non_amharic_segments.append(''.join(current_segment))
                    current_segment = []
                in_non_amharic = False
            else:
                if char.isalpha():  # Latin or other script
                    in_non_amharic = True
                    current_segment.append(char)
        
        if current_segment:
            non_amharic_segments.append(''.join(current_segment))
        
        has_code_switching = len(non_amharic_segments) > 0
        return has_code_switching, non_amharic_segments
    
    def get_character_frequency(self, text: str) -> Dict[str, int]:
        """
        Get frequency distribution of Ge'ez characters
        Useful for data quality analysis
        """
        freq = {}
        for char in text:
            if self.is_geez_char(char):
                freq[char] = freq.get(char, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    
    def validate_text_quality(self, text: str) -> Dict[str, any]:
        """
        Validate Amharic text quality for training
        
        Returns:
            Dictionary with quality metrics
        """
        total_chars = len(text)
        geez_chars = sum(1 for c in text if self.is_geez_char(c))
        
        # Check for code-switching
        has_cs, cs_segments = self.detect_code_switching(text)
        
        # Calculate metrics
        amharic_ratio = geez_chars / total_chars if total_chars > 0 else 0
        
        quality = {
            'total_characters': total_chars,
            'geez_characters': geez_chars,
            'amharic_ratio': amharic_ratio,
            'has_code_switching': has_cs,
            'code_switched_segments': len(cs_segments),
            'is_pure_amharic': amharic_ratio > 0.9,
            'quality_score': amharic_ratio if not has_cs else amharic_ratio * 0.8
        }
        
        return quality
    
    def prepare_for_whisper(self, text: str) -> str:
        """
        Prepare Amharic text specifically for Whisper training/inference
        
        This applies all normalizations optimized for ASR
        """
        # Normalize
        text = self.normalize_text(text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper spacing around punctuation
        for punct in '.،؛:؟!':  # Common punctuation
            text = text.replace(punct, f' {punct} ')
        
        # Clean up multiple spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
