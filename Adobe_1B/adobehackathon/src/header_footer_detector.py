"""
Header and footer detection module.
Identifies repetitive headers and footers across pages to exclude from heading detection.
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)


class HeaderFooterDetector:
    """Detects headers and footers in PDF documents."""
    
    def __init__(self):
        self.header_threshold_ratio = 0.15  # Top 15% of page
        self.footer_threshold_ratio = 0.85  # Bottom 15% of page
        self.min_repetition_count = 2  # Minimum pages for header/footer
        self.similarity_threshold = 0.8  # Text similarity threshold
        
    def detect_headers_footers(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """
        Detect headers and footers from text blocks.
        
        Args:
            text_blocks: List of text blocks with position information
            
        Returns:
            Dictionary with 'headers' and 'footers' sets containing repetitive text
        """
        logger.info(f"Analyzing {len(text_blocks)} text blocks for headers/footers")
        
        # Group blocks by page and position
        page_blocks = defaultdict(list)
        for block in text_blocks:
            page_blocks[block['page_number']].append(block)
        
        # Identify potential headers and footers by position
        potential_headers = defaultdict(list)
        potential_footers = defaultdict(list)
        
        for page_num, blocks in page_blocks.items():
            if not blocks:
                continue
                
            page_height = blocks[0].get('page_height', 800)  # Default height
            
            for block in blocks:
                position_ratio = block['y0'] / page_height
                
                # Check if in header region
                if position_ratio <= self.header_threshold_ratio:
                    potential_headers[page_num].append(block)
                
                # Check if in footer region
                elif position_ratio >= self.footer_threshold_ratio:
                    potential_footers[page_num].append(block)
        
        # Find repetitive patterns
        headers = self._find_repetitive_text(potential_headers)
        footers = self._find_repetitive_text(potential_footers)
        
        # Add page number patterns
        page_number_patterns = self._detect_page_numbers(page_blocks)
        headers.update(page_number_patterns['headers'])
        footers.update(page_number_patterns['footers'])
        
        logger.info(f"Detected {len(headers)} header patterns and {len(footers)} footer patterns")
        
        return {
            'headers': headers,
            'footers': footers
        }
    
    def _find_repetitive_text(self, position_blocks: Dict[int, List[Dict]]) -> Set[str]:
        """Find text that repeats across multiple pages in the same position."""
        repetitive_texts = set()
        
        # Extract text from each page
        page_texts = {}
        for page_num, blocks in position_blocks.items():
            page_texts[page_num] = [block['text'].strip() for block in blocks if block['text'].strip()]
        
        # Find common text patterns
        if len(page_texts) < self.min_repetition_count:
            return repetitive_texts
        
        # Count text occurrences across pages
        text_counter = Counter()
        for page_num, texts in page_texts.items():
            for text in texts:
                # Normalize text for comparison
                normalized_text = self._normalize_text_for_comparison(text)
                if normalized_text:
                    text_counter[normalized_text] += 1
        
        # Find texts that appear on multiple pages
        total_pages = len(position_blocks)
        for text, count in text_counter.items():
            repetition_ratio = count / total_pages
            
            # Consider as header/footer if appears on significant portion of pages
            if count >= self.min_repetition_count and repetition_ratio >= 0.3:
                repetitive_texts.add(text)
        
        # Also find similar text patterns
        similar_patterns = self._find_similar_patterns(list(text_counter.keys()))
        repetitive_texts.update(similar_patterns)
        
        return repetitive_texts
    
    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normalize text for comparison (remove page numbers, dates, etc.)."""
        if not text or len(text.strip()) < 3:
            return ""
        
        # Remove common variable elements
        normalized = text.strip()
        
        # Remove page numbers (common patterns)
        normalized = re.sub(r'\b\d+\b', '[NUM]', normalized)
        
        # Remove dates
        normalized = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', normalized)
        normalized = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '[DATE]', normalized)
        
        # Remove years
        normalized = re.sub(r'\b(19|20)\d{2}\b', '[YEAR]', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized.lower()
    
    def _find_similar_patterns(self, texts: List[str]) -> Set[str]:
        """Find groups of similar text patterns."""
        similar_patterns = set()
        
        # Group similar texts
        for i, text1 in enumerate(texts):
            similar_group = [text1]
            
            for j, text2 in enumerate(texts):
                if i != j and self._are_similar_texts(text1, text2):
                    similar_group.append(text2)
            
            # If we have a group of similar texts, consider them as pattern
            if len(similar_group) >= self.min_repetition_count:
                # Use the most common pattern as representative
                similar_patterns.add(max(similar_group, key=texts.count))
        
        return similar_patterns
    
    def _are_similar_texts(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar enough to be considered the same pattern."""
        if not text1 or not text2:
            return False
        
        # Simple similarity check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= self.similarity_threshold
    
    def _detect_page_numbers(self, page_blocks: Dict[int, List[Dict]]) -> Dict[str, Set[str]]:
        """Detect page number patterns in headers and footers."""
        page_patterns = {'headers': set(), 'footers': set()}
        
        for page_num, blocks in page_blocks.items():
            if not blocks:
                continue
                
            page_height = blocks[0].get('page_height', 800)
            
            for block in blocks:
                text = block['text'].strip()
                position_ratio = block['y0'] / page_height
                
                # Check if text looks like a page number
                if self._is_page_number_pattern(text, page_num):
                    if position_ratio <= self.header_threshold_ratio:
                        # Normalize the pattern
                        pattern = self._normalize_page_number_pattern(text)
                        page_patterns['headers'].add(pattern)
                    elif position_ratio >= self.footer_threshold_ratio:
                        pattern = self._normalize_page_number_pattern(text)
                        page_patterns['footers'].add(pattern)
        
        return page_patterns
    
    def _is_page_number_pattern(self, text: str, page_num: int) -> bool:
        """Check if text looks like a page number."""
        if not text or len(text) > 50:  # Page numbers are usually short
            return False
        
        # Direct page number match
        if text.strip() == str(page_num):
            return True
        
        # Page number with prefix/suffix
        if re.search(rf'\b{page_num}\b', text):
            # Common page number patterns
            patterns = [
                rf'page\s*{page_num}',
                rf'{page_num}\s*of\s*\d+',
                rf'-\s*{page_num}\s*-',
                rf'\|\s*{page_num}\s*\|',
            ]
            
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    return True
        
        # Roman numerals (for some documents)
        if self._is_roman_numeral(text.strip()):
            return True
        
        return False
    
    def _normalize_page_number_pattern(self, text: str) -> str:
        """Normalize page number pattern for matching."""
        # Replace actual numbers with placeholder
        normalized = re.sub(r'\b\d+\b', '[PAGE]', text)
        return normalized.strip().lower()
    
    def _is_roman_numeral(self, text: str) -> bool:
        """Check if text is a Roman numeral."""
        if not text:
            return False
        
        roman_pattern = r'^[IVXLCDM]+$'
        return bool(re.match(roman_pattern, text.upper()))
    
    def is_header_footer_text(self, text: str, headers_footers: Dict[str, Set[str]]) -> bool:
        """Check if given text matches any detected header/footer pattern."""
        if not text or not text.strip():
            return False
        
        normalized_text = self._normalize_text_for_comparison(text)
        
        # Check against known headers
        for header_pattern in headers_footers['headers']:
            if self._text_matches_pattern(normalized_text, header_pattern):
                return True
        
        # Check against known footers
        for footer_pattern in headers_footers['footers']:
            if self._text_matches_pattern(normalized_text, footer_pattern):
                return True
        
        return False
    
    def _text_matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a header/footer pattern."""
        if not text or not pattern:
            return False
        
        # Exact match after normalization
        if text == pattern:
            return True
        
        # Pattern matching (for page numbers etc.)
        if '[PAGE]' in pattern:
            # Replace [PAGE] with number pattern and check
            regex_pattern = pattern.replace('[PAGE]', r'\d+')
            if re.search(regex_pattern, text):
                return True
        
        # Similarity-based matching
        return self._are_similar_texts(text, pattern)
    
    def get_header_footer_regions(self, text_blocks: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[float]]]:
        """Get header and footer regions for each page."""
        regions = defaultdict(dict)
        
        page_blocks = defaultdict(list)
        for block in text_blocks:
            page_blocks[block['page_number']].append(block)
        
        for page_num, blocks in page_blocks.items():
            if not blocks:
                continue
                
            page_height = blocks[0].get('page_height', 800)
            page_width = blocks[0].get('page_width', 600)
            
            # Define header and footer regions
            header_region = [0, 0, page_width, page_height * self.header_threshold_ratio]
            footer_region = [0, page_height * self.footer_threshold_ratio, page_width, page_height]
            
            regions[page_num] = {
                'header': header_region,
                'footer': footer_region
            }
        
        return dict(regions)


def main():
    """Test the header/footer detector."""
    # This would typically be called with text blocks from TextExtractor
    print("HeaderFooterDetector created successfully")
    print("Use with text blocks from TextExtractor for testing")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 