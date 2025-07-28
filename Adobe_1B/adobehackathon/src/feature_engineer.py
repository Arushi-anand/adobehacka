"""
Feature engineering module for heading detection.
Extracts comprehensive features from text blocks to train the ML model.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import Counter
import re

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extracts features from text blocks for heading classification."""
    
    def __init__(self):
        # Academic terms and their types
        self.academic_terms = {
            'general': ['abstract', 'keywords', 'introduction', 'conclusion', 'references', 'bibliography'],
            'math': ['definition', 'theorem', 'corollary', 'proposition', 'lemma', 'proof', 'example'],
            'sections': ['chapter', 'section', 'subsection', 'appendix', 'part'],
            'multilingual': {
                'hindi': ['परिभाषा', 'प्रमेय', 'उदाहरण', 'अध्याय'],
                'chinese': ['定义', '定理', '例子', '章'],
                'arabic': ['تعريف', 'نظرية', 'مثال', 'فصل'],
                'spanish': ['definición', 'teorema', 'ejemplo', 'capítulo'],
                'french': ['définition', 'théorème', 'exemple', 'chapitre'],
                'german': ['definition', 'theorem', 'beispiel', 'kapitel']
            }
        }
        
        # Numbering patterns for different languages
        self.numbering_patterns = {
            'arabic': [r'^\d+\.?\s+', r'^\d+\.\d+\.?\s+'],
            'roman': [r'^[IVX]+\.?\s+', r'^[ivx]+\.?\s+'],
            'letter': [r'^[A-Z]\.?\s+', r'^[a-z]\.?\s+'],
            'multilingual': {
                'chinese': [r'^第\d+章', r'^第\d+节'],
                'hindi': [r'^अध्याय\s*\d+', r'^भाग\s*\d+'],
                'arabic': [r'^الفصل\s*\d+', r'^الباب\s*\d+']
            }
        }
        
        # Import semantic filter
        try:
            from semantic_filter import SemanticFilter
            self.semantic_filter = SemanticFilter()
        except ImportError:
            logger.warning("SemanticFilter not available, skipping semantic features")
            self.semantic_filter = None
    
    def extract_features(self, text_blocks: List[Dict[str, Any]], 
                        excluded_info: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Extract comprehensive features from text blocks.
        
        Args:
            text_blocks: List of text blocks with metadata
            excluded_info: Information about excluded regions (tables, headers, etc.)
            
        Returns:
            DataFrame with features for each text block
        """
        logger.info(f"Extracting features from {len(text_blocks)} text blocks")
        
        if not text_blocks:
            return pd.DataFrame()
        
        # Calculate document-level statistics first
        doc_stats = self._calculate_document_stats(text_blocks)
        
        features_list = []
        
        for i, block in enumerate(text_blocks):
            features = self._extract_block_features(block, i, text_blocks, doc_stats)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # Add relative rankings
        df = self._add_relative_features(df)
        
        # Add semantic filtering features
        if self.semantic_filter:
            df = self._add_semantic_features(df, text_blocks)
        
        logger.info(f"Extracted {len(df.columns)} features per text block")
        return df
    
    def _add_semantic_features(self, df: pd.DataFrame, text_blocks: List[Dict[str, Any]]) -> pd.DataFrame:
        """Add semantic filtering features to detect pseudo-headings."""
        logger.info("Adding semantic filtering features")
        
        # Get semantic analysis
        semantic_analysis = self.semantic_filter.analyze_pseudo_headings(text_blocks)
        
        # Apply adjustments
        df = self.semantic_filter.apply_semantic_adjustments(df, semantic_analysis)
        
        # Add individual semantic features for each block
        for i, (idx, row) in enumerate(df.iterrows()):
            if i < len(text_blocks):
                text = text_blocks[i]['text']
                language = text_blocks[i].get('language', 'en')
                
                semantic_features = self.semantic_filter.get_filtering_features(text, language)
                
                for feature_name, value in semantic_features.items():
                    df.loc[idx, f'semantic_{feature_name}'] = value
        
        return df
    
    def _calculate_document_stats(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate document-level statistics for relative features."""
        font_sizes = [block['font_size'] for block in text_blocks]
        text_lengths = [len(block['text']) for block in text_blocks]
        
        stats = {
            'max_font_size': max(font_sizes),
            'min_font_size': min(font_sizes),
            'median_font_size': np.median(font_sizes),
            'mean_font_size': np.mean(font_sizes),
            'font_size_std': np.std(font_sizes),
            'unique_font_sizes': sorted(set(font_sizes), reverse=True),
            'max_text_length': max(text_lengths),
            'median_text_length': np.median(text_lengths),
            'total_blocks': len(text_blocks),
            'total_pages': max(block['page_number'] for block in text_blocks),
        }
        
        # Font size rankings
        size_ranks = {}
        for i, size in enumerate(stats['unique_font_sizes']):
            size_ranks[size] = i + 1
        stats['font_size_ranks'] = size_ranks
        
        return stats
    
    def _extract_block_features(self, block: Dict[str, Any], block_index: int,
                               all_blocks: List[Dict[str, Any]], 
                               doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for a single text block."""
        text = block['text'].strip()
        features = {}
        
        # Basic text properties (from original block)
        features.update({
            'text': text,
            'font_size': block['font_size'],
            'font_name': block.get('font_name', ''),
            'normalized_font_name': block.get('normalized_font_name', ''),
            'is_bold': block.get('is_bold', False),
            'is_italic': block.get('is_italic', False),
            'is_underlined': block.get('is_underlined', False),
            'font_color': block.get('font_color', '#000000'),
            'text_height': block.get('text_height', 0),
            'text_width': block.get('text_width', 0),
            'x0': block.get('x0', 0),
            'y0': block.get('y0', 0),
            'x1': block.get('x1', 0),
            'y1': block.get('y1', 0),
            'horizontal_alignment': block.get('horizontal_alignment', 'left'),
            'is_centered': block.get('is_centered', False),
            'is_top_of_page': block.get('is_top_of_page', False),
            'page_number': block.get('page_number', 1),
            'num_words': block.get('num_words', len(text.split())),
            'text_length': block.get('text_length', len(text)),
            'all_uppercase': block.get('all_uppercase', text.isupper()),
            'starts_with_numbering': block.get('starts_with_numbering', False),
            'contains_colon': block.get('contains_colon', ':' in text),
            'contains_special_symbols': any(c in text for c in '!@#$%^&*()[]{}'),
            'language': block.get('language', 'unknown'),
            'page_section_ratio': block.get('page_section_ratio', 0),
            'punctuation_ratio': block.get('punctuation_ratio', 0),
            'digit_ratio': block.get('digit_ratio', 0),
        })
        
        # Font size ranking and relative features
        features['font_size_rank'] = doc_stats['font_size_ranks'].get(block['font_size'], 999)
        features['font_size_percentile'] = (doc_stats['max_font_size'] - block['font_size']) / (doc_stats['max_font_size'] - doc_stats['min_font_size']) if doc_stats['max_font_size'] != doc_stats['min_font_size'] else 0
        features['is_largest_font'] = block['font_size'] == doc_stats['max_font_size']
        features['is_above_median_font'] = block['font_size'] > doc_stats['median_font_size']
        
        # Enhanced case analysis features
        case_features = self._analyze_case_patterns(text)
        features.update(case_features)
        
        # Spacing analysis
        spacing_features = self._calculate_spacing_features(block, block_index, all_blocks)
        features.update(spacing_features)
        
        # Academic term analysis
        academic_features = self._analyze_academic_terms(text, block.get('language', 'en'))
        features.update(academic_features)
        
        # Numbering pattern analysis
        numbering_features = self._analyze_numbering_patterns(text, block.get('language', 'en'))
        features.update(numbering_features)
        
        # Position and layout features
        layout_features = self._analyze_layout_features(block, doc_stats)
        features.update(layout_features)
        
        # Multi-line detection features
        multiline_features = self._analyze_multiline_potential(block, block_index, all_blocks)
        features.update(multiline_features)
        
        # Language-specific features
        lang_features = self._extract_language_features(text, block.get('language', 'en'))
        features.update(lang_features)
        
        return features
    
    def _analyze_case_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text case patterns for heading detection."""
        features = {}
        
        if len(text) == 0:
            return features
        
        # Basic case analysis
        features['is_all_uppercase'] = text.isupper()
        features['is_all_lowercase'] = text.islower()
        features['starts_with_capital'] = text[0].isupper() if text else False
        features['is_title_case'] = text.istitle()
        
        # Word-level case analysis
        words = text.split()
        if words:
            features['all_words_capitalized'] = all(word[0].isupper() for word in words if word and word[0].isalpha())
            features['mixed_case_words'] = any(any(c.isupper() for c in word[1:]) and any(c.islower() for c in word[1:]) for word in words)
            features['first_word_all_caps'] = words[0].isupper() if words[0] else False
            features['last_word_all_caps'] = words[-1].isupper() if words[-1] else False
        else:
            features['all_words_capitalized'] = False
            features['mixed_case_words'] = False
            features['first_word_all_caps'] = False
            features['last_word_all_caps'] = False
        
        # Character distribution
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0:
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / total_alpha
            features['lowercase_ratio'] = sum(1 for c in text if c.islower()) / total_alpha
        else:
            features['uppercase_ratio'] = 0
            features['lowercase_ratio'] = 0
        
        # Heading-like case patterns
        features['likely_heading_case'] = (
            features['is_all_uppercase'] or 
            features['is_title_case'] or 
            (features['starts_with_capital'] and len(words) <= 6) or
            features['all_words_capitalized']
        )
        
        # Paragraph-like case patterns
        features['likely_paragraph_case'] = (
            features['is_all_lowercase'] or 
            (not features['starts_with_capital'] and len(words) > 3) or
            (features['lowercase_ratio'] > 0.8 and len(text) > 30)
        )
        
        return features

    def _calculate_spacing_features(self, block: Dict[str, Any], block_index: int,
                                  all_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate spacing features relative to surrounding blocks."""
        features = {}
        
        # Find spacing before and after
        space_before = 0
        space_after = 0
        
        same_page_blocks = [b for b in all_blocks if b['page_number'] == block['page_number']]
        same_page_blocks.sort(key=lambda x: x['y0'])
        
        current_index = next((i for i, b in enumerate(same_page_blocks) if b == block), -1)
        
        if current_index > 0:
            prev_block = same_page_blocks[current_index - 1]
            space_before = block['y0'] - prev_block['y1']
        
        if current_index < len(same_page_blocks) - 1:
            next_block = same_page_blocks[current_index + 1]
            space_after = next_block['y0'] - block['y1']
        
        features.update({
            'space_before': max(0, space_before),
            'space_after': max(0, space_after),
            'space_ratio': space_before / space_after if space_after > 0 else 0,
            'has_significant_space_before': space_before > 10,
            'has_significant_space_after': space_after > 10,
            'isolated_text': space_before > 15 and space_after > 15,
        })
        
        # Line gap ratio (relative to text height)
        text_height = block.get('text_height', 12)
        features['line_gap_ratio'] = space_before / text_height if text_height > 0 else 0
        
        return features
    
    def _analyze_academic_terms(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze academic terms in the text."""
        text_lower = text.lower().strip()
        features = {}
        
        # Check for academic terms
        found_terms = []
        
        # General academic terms
        for term in self.academic_terms['general']:
            if text_lower.startswith(term):
                found_terms.append(('general', term))
        
        # Math terms
        for term in self.academic_terms['math']:
            if text_lower.startswith(term):
                found_terms.append(('math', term))
        
        # Section terms
        for term in self.academic_terms['sections']:
            if text_lower.startswith(term):
                found_terms.append(('section', term))
        
        # Language-specific terms
        if language in self.academic_terms['multilingual']:
            for term in self.academic_terms['multilingual'][language]:
                if text.startswith(term):
                    found_terms.append(('multilingual', term))
        
        features.update({
            'contains_academic_term': len(found_terms) > 0,
            'academic_term_count': len(found_terms),
            'academic_term_type': found_terms[0][0] if found_terms else 'none',
            'is_math_term': any(t[0] == 'math' for t in found_terms),
            'is_section_term': any(t[0] == 'section' for t in found_terms),
            'is_numbered_academic': self._is_numbered_academic_term(text),
        })
        
        return features
    
    def _is_numbered_academic_term(self, text: str) -> bool:
        """Check if text is a numbered academic term like 'Definition 1.2'."""
        patterns = [
            r'^(definition|theorem|corollary|proposition|lemma|example)\s+\d+',
            r'^(定义|定理|例子)\s*\d+',  # Chinese
            r'^(تعريف|نظرية|مثال)\s*\d+',  # Arabic
        ]
        
        for pattern in patterns:
            if re.match(pattern, text.lower()):
                return True
        return False
    
    def _analyze_numbering_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze numbering patterns in the text."""
        features = {}
        
        # Check different numbering patterns
        has_arabic_num = bool(re.match(r'^\d+\.?\s+', text))
        has_roman_num = bool(re.match(r'^[IVX]+\.?\s+', text.upper()))
        has_letter_num = bool(re.match(r'^[A-Za-z]\.?\s+', text))
        
        # Multi-level numbering
        has_multilevel = bool(re.match(r'^\d+\.\d+', text))
        has_deep_multilevel = bool(re.match(r'^\d+\.\d+\.\d+', text))
        
        # Language-specific numbering
        has_lang_specific = False
        if language in self.numbering_patterns['multilingual']:
            patterns = self.numbering_patterns['multilingual'][language]
            has_lang_specific = any(re.match(pattern, text) for pattern in patterns)
        
        features.update({
            'has_arabic_numbering': has_arabic_num,
            'has_roman_numbering': has_roman_num,
            'has_letter_numbering': has_letter_num,
            'has_multilevel_numbering': has_multilevel,
            'has_deep_multilevel_numbering': has_deep_multilevel,
            'has_language_specific_numbering': has_lang_specific,
            'numbering_level': self._determine_numbering_level(text),
        })
        
        return features
    
    def _determine_numbering_level(self, text: str) -> int:
        """Determine the hierarchical level from numbering pattern."""
        # Level 1: 1., I., Chapter 1, etc.
        if re.match(r'^(\d+\.|\w+\s+\d+|[IVX]+\.)', text):
            return 1
        # Level 2: 1.1., 1.1, etc.
        if re.match(r'^\d+\.\d+\.?', text):
            return 2
        # Level 3: 1.1.1., etc.
        if re.match(r'^\d+\.\d+\.\d+\.?', text):
            return 3
        # Level 4: 1.1.1.1., etc.
        if re.match(r'^\d+\.\d+\.\d+\.\d+\.?', text):
            return 4
        
        return 0
    
    def _analyze_layout_features(self, block: Dict[str, Any], doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze layout and positioning features."""
        features = {}
        
        page_width = block.get('page_width', 600)
        page_height = block.get('page_height', 800)
        
        # Position ratios
        features.update({
            'x_position_ratio': block['x0'] / page_width if page_width > 0 else 0,
            'y_position_ratio': block['y0'] / page_height if page_height > 0 else 0,
            'width_ratio': block.get('text_width', 0) / page_width if page_width > 0 else 0,
            'height_ratio': block.get('text_height', 0) / page_height if page_height > 0 else 0,
        })
        
        # Text length relative to others
        features.update({
            'is_short_text': len(block['text']) < 50,
            'is_medium_text': 50 <= len(block['text']) <= 150,
            'is_long_text': len(block['text']) > 150,
            'text_length_percentile': min(len(block['text']) / doc_stats['max_text_length'], 1.0),
        })
        
        return features
    
    def _analyze_multiline_potential(self, block: Dict[str, Any], block_index: int,
                                   all_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential for multi-line title merging."""
        features = {}
        
        # Check if this could be part of a multi-line title
        same_page_blocks = [b for b in all_blocks if b['page_number'] == block['page_number']]
        same_page_blocks.sort(key=lambda x: x['y0'])
        
        current_index = next((i for i, b in enumerate(same_page_blocks) if b == block), -1)
        
        # Check next block for continuation potential
        could_continue = False
        if current_index < len(same_page_blocks) - 1:
            next_block = same_page_blocks[current_index + 1]
            
            # Similar font size and style
            font_similar = abs(block['font_size'] - next_block['font_size']) <= 2
            style_similar = (block.get('is_bold') == next_block.get('is_bold') and
                           block.get('font_name') == next_block.get('font_name'))
            close_proximity = (next_block['y0'] - block['y1']) < (block.get('text_height', 12) * 1.5)
            
            could_continue = font_similar and style_similar and close_proximity
        
        features.update({
            'could_be_multiline_start': could_continue and len(block['text']) < 100,
            'could_be_multiline_continuation': self._could_be_continuation(block, same_page_blocks, current_index),
            'line_position_in_group': current_index + 1,
        })
        
        return features
    
    def _could_be_continuation(self, block: Dict[str, Any], same_page_blocks: List[Dict],
                             current_index: int) -> bool:
        """Check if this block could be a continuation of previous line."""
        if current_index <= 0:
            return False
        
        prev_block = same_page_blocks[current_index - 1]
        
        # Similar formatting and close proximity
        font_similar = abs(block['font_size'] - prev_block['font_size']) <= 2
        style_similar = (block.get('is_bold') == prev_block.get('is_bold') and
                        block.get('font_name') == prev_block.get('font_name'))
        close_proximity = (block['y0'] - prev_block['y1']) < (block.get('text_height', 12) * 1.5)
        
        return font_similar and style_similar and close_proximity
    
    def _extract_language_features(self, text: str, language: str) -> Dict[str, Any]:
        """Extract language-specific features."""
        features = {}
        
        # Script type detection
        script_types = {
            'latin': bool(re.search(r'[a-zA-Z]', text)),
            'cyrillic': bool(re.search(r'[\u0400-\u04FF]', text)),
            'arabic': bool(re.search(r'[\u0600-\u06FF]', text)),
            'chinese': bool(re.search(r'[\u4e00-\u9fff]', text)),
            'devanagari': bool(re.search(r'[\u0900-\u097F]', text)),
        }
        
        features.update({
            f'script_{script}': present for script, present in script_types.items()
        })
        
        # Mixed script detection
        features['mixed_script'] = sum(script_types.values()) > 1
        
        # Language-specific punctuation
        features.update({
            'has_arabic_punctuation': any(c in text for c in '،؍؎؏ؐؑ'),
            'has_chinese_punctuation': any(c in text for c in '，。！？；：'),
            'has_hindi_punctuation': any(c in text for c in '।॥'),
        })
        
        return features
    
    def _add_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative ranking features across the document."""
        if df.empty:
            return df
        
        # Font size rankings
        df['font_size_rank_percentile'] = df['font_size_rank'].rank(pct=True)
        
        # Space rankings
        if 'space_before' in df.columns:
            df['space_before_rank'] = df['space_before'].rank(ascending=False)
            df['space_after_rank'] = df['space_after'].rank(ascending=False)
        
        # Text length rankings
        df['text_length_rank'] = df['text_length'].rank(ascending=False)
        
        # Page position features
        df['is_first_page'] = df['page_number'] == df['page_number'].min()
        df['is_early_page'] = df['page_number'] <= df['page_number'].quantile(0.3)
        
        return df


def main():
    """Test the feature engineer."""
    print("FeatureEngineer created successfully")
    print("Use with text blocks for feature extraction testing")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 