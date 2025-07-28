"""
Semantic filtering module for heading detection.
Identifies and filters out pseudo-headings that are inline labels rather than structural headings.
"""

import logging
import re
from typing import List, Dict, Any, Set, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class SemanticFilter:
    """Filters out pseudo-headings using semantic analysis."""
    
    def __init__(self):
        # Common inline label patterns that are NOT headings
        self.inline_label_patterns = {
            'descriptors': [
                'competence:', 'criticality:', 'complexity:', 'importance:', 'urgency:',
                'priority:', 'scope:', 'purpose:', 'objective:', 'goal:', 'requirement:',
                'constraint:', 'assumption:', 'limitation:', 'consideration:', 'factor:',
                'aspect:', 'dimension:', 'criteria:', 'parameter:', 'variable:',
                'attribute:', 'characteristic:', 'feature:', 'property:', 'quality:'
            ],
            'metadata': [
                'author:', 'date:', 'version:', 'status:', 'type:', 'category:',
                'source:', 'reference:', 'location:', 'time:', 'duration:',
                'frequency:', 'quantity:', 'amount:', 'size:', 'volume:',
                'weight:', 'length:', 'width:', 'height:', 'depth:'
            ],
            'annotations': [
                'note:', 'warning:', 'caution:', 'tip:', 'hint:', 'example:',
                'illustration:', 'case:', 'scenario:', 'situation:', 'context:',
                'background:', 'overview:', 'summary:', 'conclusion:', 'result:',
                'outcome:', 'finding:', 'observation:', 'insight:', 'lesson:'
            ],
            'technical': [
                'input:', 'output:', 'parameter:', 'argument:', 'return:', 'value:',
                'type:', 'format:', 'structure:', 'syntax:', 'semantics:',
                'algorithm:', 'method:', 'function:', 'procedure:', 'process:',
                'operation:', 'action:', 'step:', 'phase:', 'stage:'
            ]
        }
        
        # Multilingual equivalents
        self.multilingual_patterns = {
            'spanish': ['competencia:', 'criticidad:', 'complejidad:', 'importancia:'],
            'french': ['compétence:', 'criticité:', 'complexité:', 'importance:'],
            'german': ['kompetenz:', 'kritikalität:', 'komplexität:', 'wichtigkeit:'],
            'chinese': ['能力：', '关键性：', '复杂性：', '重要性：'],
            'arabic': ['الكفاءة:', 'الأهمية الحرجة:', 'التعقيد:', 'الأهمية:'],
        }
        
        # Patterns that suggest continuation (inline labels)
        self.continuation_indicators = [
            r':\s*[a-z]',  # Colon followed by lowercase (continuation)
            r':\s*the\s+',  # Colon followed by "the"
            r':\s*this\s+',  # Colon followed by "this"
            r':\s*it\s+',   # Colon followed by "it"
            r':\s*\w+\s+\w+\s+\w+',  # Colon followed by 3+ words (paragraph)
        ]
        
        # True heading patterns (these ARE likely headings)
        self.true_heading_patterns = [
            r'^(chapter|section|part|appendix)\s+\d+',
            r'^\d+\.\s*[A-Z]',  # Numbered headings
            r'^[IVX]+\.\s*[A-Z]',  # Roman numeral headings
            r'^(introduction|conclusion|abstract|references)$',
            r'^(methodology|results|discussion|background)$',
        ]
    
    def analyze_pseudo_headings(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze text blocks to identify pseudo-headings.
        
        Args:
            text_blocks: List of text blocks with features
            
        Returns:
            Analysis results with pseudo-heading indicators
        """
        logger.info(f"Analyzing {len(text_blocks)} text blocks for pseudo-headings")
        
        results = {
            'pseudo_headings': [],
            'inline_labels': [],
            'repeated_patterns': {},
            'continuation_patterns': [],
            'confidence_adjustments': {}
        }
        
        # Group by text content for repetition analysis
        text_counter = Counter()
        for block in text_blocks:
            clean_text = self._normalize_for_comparison(block['text'])
            text_counter[clean_text] += 1
        
        # Analyze each block
        for i, block in enumerate(text_blocks):
            text = block['text'].strip()
            analysis = self._analyze_single_block(text, block, text_blocks, i)
            
            # Check for repetition (weakens heading candidacy)
            clean_text = self._normalize_for_comparison(text)
            analysis['repetition_count'] = text_counter[clean_text]
            analysis['is_repeated'] = text_counter[clean_text] > 1
            
            # Store results
            if analysis['is_pseudo_heading']:
                results['pseudo_headings'].append({
                    'index': i,
                    'text': text,
                    'reason': analysis['pseudo_reason'],
                    'confidence': analysis['pseudo_confidence']
                })
            
            if analysis['is_inline_label']:
                results['inline_labels'].append({
                    'index': i,
                    'text': text,
                    'pattern_type': analysis['label_type']
                })
            
            if analysis['has_continuation']:
                results['continuation_patterns'].append({
                    'index': i,
                    'text': text,
                    'continuation_type': analysis['continuation_type']
                })
            
            # Store confidence adjustment for this block
            results['confidence_adjustments'][i] = analysis['confidence_adjustment']
        
        # Find repeated patterns
        for text, count in text_counter.items():
            if count > 1 and len(text) > 5:
                results['repeated_patterns'][text] = count
        
        logger.info(f"Found {len(results['pseudo_headings'])} pseudo-headings, "
                   f"{len(results['inline_labels'])} inline labels")
        
        return results
    
    def _analyze_single_block(self, text: str, block: Dict[str, Any], 
                             all_blocks: List[Dict[str, Any]], block_index: int) -> Dict[str, Any]:
        """Analyze a single text block for pseudo-heading patterns."""
        analysis = {
            'is_pseudo_heading': False,
            'pseudo_reason': '',
            'pseudo_confidence': 0.0,
            'is_inline_label': False,
            'label_type': '',
            'has_continuation': False,
            'continuation_type': '',
            'confidence_adjustment': 0.0
        }
        
        text_lower = text.lower().strip()
        
        # Check for inline label patterns
        for category, patterns in self.inline_label_patterns.items():
            for pattern in patterns:
                if text_lower == pattern or text_lower.startswith(pattern + ' '):
                    analysis['is_inline_label'] = True
                    analysis['label_type'] = category
                    analysis['is_pseudo_heading'] = True
                    analysis['pseudo_reason'] = f'inline_label_{category}'
                    analysis['pseudo_confidence'] = 0.8
                    analysis['confidence_adjustment'] = -30  # Strong penalty
                    break
            if analysis['is_inline_label']:
                break
        
        # Check for continuation patterns
        for pattern in self.continuation_indicators:
            if re.search(pattern, text):
                analysis['has_continuation'] = True
                analysis['continuation_type'] = 'inline_continuation'
                analysis['is_pseudo_heading'] = True
                analysis['pseudo_reason'] = 'continuation_pattern'
                analysis['pseudo_confidence'] = 0.7
                analysis['confidence_adjustment'] = -20  # Medium penalty
                break
        
        # Check for true heading patterns (positive indicators)
        is_likely_true_heading = False
        for pattern in self.true_heading_patterns:
            if re.match(pattern, text_lower):
                is_likely_true_heading = True
                analysis['confidence_adjustment'] = +15  # Boost confidence
                break
        
        # Additional semantic checks
        if not is_likely_true_heading:
            
            # Single word + colon = likely label
            if re.match(r'^\w+:$', text):
                analysis['is_pseudo_heading'] = True
                analysis['pseudo_reason'] = 'single_word_colon'
                analysis['pseudo_confidence'] = 0.6
                analysis['confidence_adjustment'] = -15
            
            # Very short text with colon but no clear heading structure
            elif len(text) < 20 and ':' in text and not any(c.isdigit() for c in text):
                analysis['is_pseudo_heading'] = True
                analysis['pseudo_reason'] = 'short_colon_no_structure'
                analysis['pseudo_confidence'] = 0.5
                analysis['confidence_adjustment'] = -10
            
            # Check context - if surrounded by long text, likely inline
            context_analysis = self._analyze_context(block, all_blocks, block_index)
            if context_analysis['surrounded_by_long_text']:
                analysis['confidence_adjustment'] -= 10
            
            # Check if text flows grammatically into next line
            if context_analysis['flows_to_next']:
                analysis['is_pseudo_heading'] = True
                analysis['pseudo_reason'] = 'grammatical_flow'
                analysis['pseudo_confidence'] = 0.6
                analysis['confidence_adjustment'] = -25
        
        return analysis
    
    def _analyze_context(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]], 
                        block_index: int) -> Dict[str, Any]:
        """Analyze the context around a text block."""
        context = {
            'surrounded_by_long_text': False,
            'flows_to_next': False,
            'prev_text_length': 0,
            'next_text_length': 0
        }
        
        # Get same-page blocks
        same_page_blocks = [b for b in all_blocks if b['page_number'] == block['page_number']]
        same_page_blocks.sort(key=lambda x: x['y0'])
        
        current_index = next((i for i, b in enumerate(same_page_blocks) if b == block), -1)
        
        # Check previous block
        if current_index > 0:
            prev_block = same_page_blocks[current_index - 1]
            context['prev_text_length'] = len(prev_block['text'])
            
        # Check next block
        if current_index < len(same_page_blocks) - 1:
            next_block = same_page_blocks[current_index + 1]
            context['next_text_length'] = len(next_block['text'])
            
            # Check if current text flows into next (grammatical continuation)
            current_text = block['text'].strip()
            next_text = next_block['text'].strip()
            
            if (current_text.endswith(':') and 
                next_text and next_text[0].islower() and
                len(next_text) > 50):  # Next is a substantial paragraph
                context['flows_to_next'] = True
        
        # Surrounded by long text
        if (context['prev_text_length'] > 100 and 
            context['next_text_length'] > 100):
            context['surrounded_by_long_text'] = True
            
        return context
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for repetition comparison."""
        # Remove page numbers, dates, and other variable elements
        normalized = text.strip().lower()
        
        # Remove common variable patterns
        normalized = re.sub(r'\b\d+\b', '[NUM]', normalized)
        normalized = re.sub(r'\b(page|p\.)\s*\d+', '[PAGE]', normalized)
        normalized = re.sub(r'\b\d{4}\b', '[YEAR]', normalized)
        
        return normalized
    
    def apply_semantic_adjustments(self, features_df, semantic_analysis: Dict[str, Any]):
        """Apply semantic confidence adjustments to features."""
        if 'confidence_adjustments' not in semantic_analysis:
            return features_df
        
        # Add semantic features
        features_df['is_pseudo_heading'] = False
        features_df['is_inline_label'] = False
        features_df['semantic_confidence_adj'] = 0.0
        features_df['repetition_count'] = 1
        
        for i, (idx, row) in enumerate(features_df.iterrows()):
            if i in semantic_analysis['confidence_adjustments']:
                adj = semantic_analysis['confidence_adjustments'][i]
                features_df.loc[idx, 'semantic_confidence_adj'] = adj
                
                # Mark pseudo-headings
                if adj <= -20:
                    features_df.loc[idx, 'is_pseudo_heading'] = True
                    
                # Check if it's an inline label
                for label_info in semantic_analysis['inline_labels']:
                    if label_info['index'] == i:
                        features_df.loc[idx, 'is_inline_label'] = True
                        break
        
        logger.info(f"Applied semantic adjustments to {len(features_df)} blocks")
        return features_df
    
    def get_filtering_features(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Get semantic filtering features for a single text block."""
        features = {}
        text_lower = text.lower().strip()
        
        # Inline label detection
        features['matches_inline_pattern'] = any(
            text_lower.startswith(pattern) 
            for patterns in self.inline_label_patterns.values() 
            for pattern in patterns
        )
        
        # Continuation pattern detection
        features['has_continuation_pattern'] = any(
            re.search(pattern, text) for pattern in self.continuation_indicators
        )
        
        # True heading pattern detection
        features['matches_heading_pattern'] = any(
            re.match(pattern, text_lower) for pattern in self.true_heading_patterns
        )
        
        # Colon analysis
        features['ends_with_colon'] = text.endswith(':')
        features['single_word_colon'] = bool(re.match(r'^\w+:$', text))
        features['colon_continuation'] = bool(re.search(r':\s*[a-z]', text))
        
        # Length and structure
        features['very_short_with_colon'] = len(text) < 20 and ':' in text
        features['word_count'] = len(text.split())
        features['has_structural_numbering'] = bool(
            re.match(r'^(\d+\.|\d+\.\d+\.?|[IVX]+\.)', text)
        )
        
        return features


def main():
    """Test the semantic filter."""
    filter = SemanticFilter()
    
    # Test cases
    test_texts = [
        "Competence: the user must possess domain knowledge",
        "Criticality: high stakes influence design decisions", 
        "Chapter 1: Introduction",
        "1.1 Background",
        "Importance: this factor affects outcomes",
        "Introduction",
        "References"
    ]
    
    print("Testing semantic filtering:")
    for text in test_texts:
        features = filter.get_filtering_features(text)
        print(f"\nText: '{text}'")
        print(f"  Inline pattern: {features['matches_inline_pattern']}")
        print(f"  Continuation: {features['has_continuation_pattern']}")
        print(f"  True heading: {features['matches_heading_pattern']}")
        print(f"  Single word colon: {features['single_word_colon']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 