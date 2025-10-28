"""Amharic-specific evaluation metrics for ASR"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

try:
    from utils.amharic_tokenizer import AmharicTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


class AmharicASRMetrics:
    """Evaluation metrics specifically designed for Amharic ASR"""
    
    def __init__(self):
        if TOKENIZER_AVAILABLE:
            self.tokenizer = AmharicTokenizer(
                normalize_punctuation=True,
                normalize_numerals=True
            )
        else:
            self.tokenizer = None
    
    def normalize_for_eval(self, text: str) -> str:
        """
        Normalize text for fair evaluation
        
        Args:
            text: Input text (hypothesis or reference)
        
        Returns:
            Normalized text
        """
        if self.tokenizer:
            text = self.tokenizer.normalize_text(text)
        
        # Additional normalization for evaluation
        text = text.lower() if text.isascii() else text  # Only lowercase non-Amharic
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def calculate_wer(self, 
                      references: List[str],
                      hypotheses: List[str]) -> Dict[str, float]:
        """
        Calculate Word Error Rate with Amharic-specific normalization
        
        Args:
            references: List of reference transcripts
            hypotheses: List of hypothesis transcripts
        
        Returns:
            Dictionary with WER and related metrics
        """
        if not JIWER_AVAILABLE:
            raise ImportError("jiwer not available. Install with: pip install jiwer")
        
        # Normalize all texts
        norm_refs = [self.normalize_for_eval(r) for r in references]
        norm_hyps = [self.normalize_for_eval(h) for h in hypotheses]
        
        # Calculate WER
        wer = jiwer.wer(norm_refs, norm_hyps)
        
        # Calculate detailed measures
        measures = jiwer.compute_measures(norm_refs, norm_hyps)
        
        return {
            'wer': wer * 100,  # Convert to percentage
            'mer': measures['mer'] * 100,  # Match error rate
            'wil': measures['wil'] * 100,  # Word information lost
            'wip': (1 - measures['wil']) * 100,  # Word information preserved
            'hits': measures['hits'],
            'substitutions': measures['substitutions'],
            'deletions': measures['deletions'],
            'insertions': measures['insertions']
        }
    
    def calculate_cer(self,
                      references: List[str],
                      hypotheses: List[str]) -> Dict[str, float]:
        """
        Calculate Character Error Rate (important for Amharic syllabary)
        
        Args:
            references: List of reference transcripts
            hypotheses: List of hypothesis transcripts
        
        Returns:
            Dictionary with CER and related metrics
        """
        if not JIWER_AVAILABLE:
            raise ImportError("jiwer not available")
        
        # Normalize
        norm_refs = [self.normalize_for_eval(r) for r in references]
        norm_hyps = [self.normalize_for_eval(h) for h in hypotheses]
        
        # Calculate CER
        cer = jiwer.cer(norm_refs, norm_hyps)
        
        # Detailed character-level measures
        measures = jiwer.compute_measures(
            norm_refs,
            norm_hyps,
            truth_transform=jiwer.Compose([jiwer.RemoveMultipleSpaces()]),
            hypothesis_transform=jiwer.Compose([jiwer.RemoveMultipleSpaces()])
        )
        
        return {
            'cer': cer * 100,
            'character_hits': measures['hits'],
            'character_substitutions': measures['substitutions'],
            'character_deletions': measures['deletions'],
            'character_insertions': measures['insertions']
        }
    
    def calculate_geez_syllable_error_rate(self,
                                           references: List[str],
                                           hypotheses: List[str]) -> float:
        """
        Calculate error rate at Ge'ez syllable level
        This is more meaningful for Amharic than character-level
        
        Args:
            references: List of reference transcripts
            hypotheses: List of hypothesis transcripts
        
        Returns:
            Syllable error rate (percentage)
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not available")
        
        total_syllables = 0
        total_errors = 0
        
        for ref, hyp in zip(references, hypotheses):
            # Get syllable breakdowns
            ref_syllables = self.tokenizer.get_syllable_breakdown(
                self.normalize_for_eval(ref)
            )
            hyp_syllables = self.tokenizer.get_syllable_breakdown(
                self.normalize_for_eval(hyp)
            )
            
            # Calculate Levenshtein distance at syllable level
            errors = self._levenshtein_distance(ref_syllables, hyp_syllables)
            
            total_syllables += len(ref_syllables)
            total_errors += errors
        
        ser = (total_errors / total_syllables * 100) if total_syllables > 0 else 0
        return ser
    
    def _levenshtein_distance(self, s1: List[str], s2: List[str]) -> int:
        """
        Calculate Levenshtein distance between two sequences
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def analyze_error_patterns(self,
                              references: List[str],
                              hypotheses: List[str],
                              top_n: int = 20) -> Dict:
        """
        Analyze common error patterns in Amharic ASR
        
        Returns:
            Dictionary with error analysis
        """
        if not self.tokenizer:
            return {}
        
        substitution_errors = Counter()
        deletion_errors = Counter()
        insertion_errors = Counter()
        
        for ref, hyp in zip(references, hypotheses):
            ref_words = self.tokenizer.tokenize_words(self.normalize_for_eval(ref))
            hyp_words = self.tokenizer.tokenize_words(self.normalize_for_eval(hyp))
            
            # Align words (simple approach)
            for i, (r_word, h_word) in enumerate(zip(ref_words, hyp_words)):
                if r_word != h_word:
                    substitution_errors[(r_word, h_word)] += 1
            
            # Deletions
            if len(ref_words) > len(hyp_words):
                for word in ref_words[len(hyp_words):]:
                    deletion_errors[word] += 1
            
            # Insertions
            if len(hyp_words) > len(ref_words):
                for word in hyp_words[len(ref_words):]:
                    insertion_errors[word] += 1
        
        return {
            'top_substitutions': substitution_errors.most_common(top_n),
            'top_deletions': deletion_errors.most_common(top_n),
            'top_insertions': insertion_errors.most_common(top_n),
            'total_substitutions': sum(substitution_errors.values()),
            'total_deletions': sum(deletion_errors.values()),
            'total_insertions': sum(insertion_errors.values())
        }
    
    def calculate_comprehensive_metrics(self,
                                       references: List[str],
                                       hypotheses: List[str]) -> Dict:
        """
        Calculate all Amharic-specific metrics at once
        
        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {}
        
        # WER
        try:
            wer_metrics = self.calculate_wer(references, hypotheses)
            metrics.update(wer_metrics)
        except Exception as e:
            print(f"WER calculation failed: {e}")
        
        # CER
        try:
            cer_metrics = self.calculate_cer(references, hypotheses)
            metrics.update(cer_metrics)
        except Exception as e:
            print(f"CER calculation failed: {e}")
        
        # Syllable Error Rate
        try:
            ser = self.calculate_geez_syllable_error_rate(references, hypotheses)
            metrics['syllable_error_rate'] = ser
        except Exception as e:
            print(f"SER calculation failed: {e}")
        
        # Error analysis
        try:
            error_analysis = self.analyze_error_patterns(references, hypotheses)
            metrics['error_patterns'] = error_analysis
        except Exception as e:
            print(f"Error analysis failed: {e}")
        
        return metrics
