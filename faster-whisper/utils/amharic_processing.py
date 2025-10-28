"""Amharic text processing utilities"""

import re
from typing import Dict, List, Optional

# Ge'ez script Unicode range
GEEZ_RANGE = (0x1200, 0x137F)
GEEZ_SUPPLEMENT_RANGE = (0x1380, 0x139F)
ETHIOPIC_EXTENDED_RANGE = (0x2D80, 0x2DDF)

# Ethiopic numerals to Arabic mapping
ETHIOPIC_NUMERALS = {
    '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
    '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10',
    '፳': '20', '፴': '30', '፵': '40', '፶': '50',
    '፷': '60', '፸': '70', '፹': '80', '፺': '90', '፻': '100'
}


class AmharicTextProcessor:
    """Amharic text normalization and processing"""
    
    def __init__(self):
        self.ethiopic_to_arabic = ETHIOPIC_NUMERALS
    
    def normalize_numerals(self, text: str) -> str:
        """
        Convert Ethiopic numerals to Arabic numerals
        
        Args:
            text: Input text with Ethiopic numerals
        
        Returns:
            Text with Arabic numerals
        """
        for ethiopic, arabic in self.ethiopic_to_arabic.items():
            text = text.replace(ethiopic, arabic)
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize Amharic punctuation
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        # Ethiopic punctuation marks
        replacements = {
            '።': '.',  # Ethiopic full stop
            '፣': ',',  # Ethiopic comma
            '፤': ';',  # Ethiopic semicolon
            '፥': ':',  # Ethiopic colon
            '፦': '::',  # Ethiopic preface colon
            '፧': '?',  # Ethiopic question mark
        }
        
        for ethiopic, standard in replacements.items():
            text = text.replace(ethiopic, standard)
        
        return text
    
    def is_geez_script(self, char: str) -> bool:
        """
        Check if character is in Ge'ez script range
        
        Args:
            char: Single character
        
        Returns:
            True if character is Ge'ez script
        """
        if not char:
            return False
        
        code = ord(char)
        return (
            (GEEZ_RANGE[0] <= code <= GEEZ_RANGE[1]) or
            (GEEZ_SUPPLEMENT_RANGE[0] <= code <= GEEZ_SUPPLEMENT_RANGE[1]) or
            (ETHIOPIC_EXTENDED_RANGE[0] <= code <= ETHIOPIC_EXTENDED_RANGE[1])
        )
    
    def normalize_text(self, text: str, lowercase: bool = False) -> str:
        """
        Full text normalization pipeline
        
        Args:
            text: Input text
            lowercase: Whether to lowercase (not recommended for Amharic)
        
        Returns:
            Normalized text
        """
        # Normalize numerals
        text = self.normalize_numerals(text)
        
        # Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase if requested (not typical for Amharic)
        if lowercase:
            text = text.lower()
        
        return text
    
    def detect_language(self, text: str) -> Dict[str, float]:
        """
        Detect if text contains Amharic/Ge'ez script
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with language detection scores
        """
        if not text:
            return {'amharic': 0.0, 'other': 0.0}
        
        geez_count = sum(1 for char in text if self.is_geez_script(char))
        total_chars = len([c for c in text if c.strip()])
        
        if total_chars == 0:
            return {'amharic': 0.0, 'other': 0.0}
        
        amharic_ratio = geez_count / total_chars
        
        return {
            'amharic': amharic_ratio,
            'other': 1.0 - amharic_ratio
        }
    
    def transliterate_to_latin(self, text: str) -> str:
        """
        Simple transliteration from Ge'ez to Latin script
        (Basic implementation - can be extended)
        
        Args:
            text: Amharic text in Ge'ez script
        
        Returns:
            Transliterated text in Latin script
        """
        # Basic transliteration map (simplified)
        transliteration_map = {
            'ሀ': 'ha', 'ሁ': 'hu', 'ሂ': 'hi', 'ሃ': 'ha', 'ሄ': 'he', 'ህ': 'h', 'ሆ': 'ho',
            'ለ': 'le', 'ሉ': 'lu', 'ሊ': 'li', 'ላ': 'la', 'ሌ': 'le', 'ል': 'l', 'ሎ': 'lo',
            'መ': 'me', 'ሙ': 'mu', 'ሚ': 'mi', 'ማ': 'ma', 'ሜ': 'me', 'ም': 'm', 'ሞ': 'mo',
            'ረ': 're', 'ሩ': 'ru', 'ሪ': 'ri', 'ራ': 'ra', 'ሬ': 're', 'ር': 'r', 'ሮ': 'ro',
            'ሰ': 'se', 'ሱ': 'su', 'ሲ': 'si', 'ሳ': 'sa', 'ሴ': 'se', 'ስ': 's', 'ሶ': 'so',
            'ሸ': 'she', 'ሹ': 'shu', 'ሺ': 'shi', 'ሻ': 'sha', 'ሼ': 'she', 'ሽ': 'sh', 'ሾ': 'sho',
            'በ': 'be', 'ቡ': 'bu', 'ቢ': 'bi', 'ባ': 'ba', 'ቤ': 'be', 'ብ': 'b', 'ቦ': 'bo',
            'ተ': 'te', 'ቱ': 'tu', 'ቲ': 'ti', 'ታ': 'ta', 'ቴ': 'te', 'ት': 't', 'ቶ': 'to',
            'ነ': 'ne', 'ኑ': 'nu', 'ኒ': 'ni', 'ና': 'na', 'ኔ': 'ne', 'ን': 'n', 'ኖ': 'no',
            'አ': 'a', 'ኡ': 'u', 'ኢ': 'i', 'ኣ': 'a', 'ኤ': 'e', 'እ': 'e', 'ኦ': 'o',
            'ከ': 'ke', 'ኩ': 'ku', 'ኪ': 'ki', 'ካ': 'ka', 'ኬ': 'ke', 'ክ': 'k', 'ኮ': 'ko',
            'ወ': 'we', 'ዉ': 'wu', 'ዊ': 'wi', 'ዋ': 'wa', 'ዌ': 'we', 'ው': 'w', 'ዎ': 'wo',
            'የ': 'ye', 'ዩ': 'yu', 'ዪ': 'yi', 'ያ': 'ya', 'ዬ': 'ye', 'ይ': 'y', 'ዮ': 'yo',
            'ደ': 'de', 'ዱ': 'du', 'ዲ': 'di', 'ዳ': 'da', 'ዴ': 'de', 'ድ': 'd', 'ዶ': 'do',
            'ገ': 'ge', 'ጉ': 'gu', 'ጊ': 'gi', 'ጋ': 'ga', 'ጌ': 'ge', 'ግ': 'g', 'ጎ': 'go',
            'ጠ': 'Te', 'ጡ': 'Tu', 'ጢ': 'Ti', 'ጣ': 'Ta', 'ጤ': 'Te', 'ጥ': 'T', 'ጦ': 'To',
            'ጨ': 'che', 'ጩ': 'chu', 'ጪ': 'chi', 'ጫ': 'cha', 'ጬ': 'che', 'ጭ': 'ch', 'ጮ': 'cho',
            'ፈ': 'fe', 'ፉ': 'fu', 'ፊ': 'fi', 'ፋ': 'fa', 'ፌ': 'fe', 'ፍ': 'f', 'ፎ': 'fo',
            'ፐ': 'pe', 'ፑ': 'pu', 'ፒ': 'pi', 'ፓ': 'pa', 'ፔ': 'pe', 'ፕ': 'p', 'ፖ': 'po',
        }
        
        result = []
        for char in text:
            if char in transliteration_map:
                result.append(transliteration_map[char])
            else:
                result.append(char)
        
        return ''.join(result)
