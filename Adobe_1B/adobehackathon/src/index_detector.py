"""
Index and Table of Contents detection module.
Identifies index sections and TOC to exclude from heading detection.
"""

import logging
from typing import List, Dict, Any, Set, Tuple
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class IndexDetector:
    """Detects index and table of contents sections in PDF documents."""
    
    def __init__(self):
        # Common TOC/Index keywords in multiple languages
        self.toc_keywords = {
            'english': ['contents', 'table of contents', 'index', 'outline'],
            'spanish': ['contenido', 'índice', 'tabla de contenidos'],
            'french': ['sommaire', 'table des matières', 'index'],
            'german': ['inhalt', 'inhaltsverzeichnis', 'index'],
            'italian': ['indice', 'sommario', 'contenuto'],
            'portuguese': ['sumário', 'índice', 'conteúdo'],
            'russian': ['содержание', 'оглавление', 'указатель'],
            'chinese': ['目录', '内容', '索引'],
            'japanese': ['目次', '索引', '内容'],
            'korean': ['목차', '색인', '내용'],
            'hindi': ['सूची', 'अनुक्रमणिका', 'विषय-सूची'],
            'arabic': ['المحتويات', 'الفهرس', 'جدول المحتويات'],
            'turkish': ['içindekiler', 'dizin', 'içerik']
        }
        
        # Patterns that indicate page references (common in TOC/Index)
        self.page_reference_patterns = [
            r'\.\s*\d+$',  # Dots followed by page number
            r'-+\s*\d+$',  # Dashes followed by page number
            r'\s+\d+$',    # Spaces followed by page number
            r'\d+\s*$',    # Just page number at end
            r'\.\.\.\s*\d+',  # Dot leaders with page number
        ]
        
        # Patterns for chapter/section numbering in TOC
        self.toc_numbering_patterns = [
            r'^\d+\.?\s+',     # 1. or 1 
            r'^\d+\.\d+\.?\s+', # 1.1. or 1.1
            r'^\d+\.\d+\.\d+\.?\s+', # 1.1.1. or 1.1.1
            r'^[IVX]+\.?\s+',   # Roman numerals
            r'^[A-Z]\.?\s+',    # Capital letters
            r'^[a-z]\.?\s+',    # Lowercase letters
        ]
    
    def detect_index_sections(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect index and TOC sections in the document.
        
        Args:
            text_blocks: List of text blocks with position information
            
        Returns:
            Dictionary containing detected index regions and patterns
        """
        logger.info(f"Analyzing {len(text_blocks)} text blocks for index sections")
        
        # Group blocks by page
        page_blocks = defaultdict(list)
        for block in text_blocks:
            page_blocks[block['page_number']].append(block)
        
        results = {
            'toc_pages': set(),
            'index_pages': set(),
            'toc_patterns': set(),
            'index_patterns': set(),
            'page_reference_blocks': [],
            'excluded_regions': []
        }
        
        # Detect TOC/Index by keywords
        keyword_pages = self._detect_by_keywords(page_blocks)
        results['toc_pages'].update(keyword_pages['toc'])
        results['index_pages'].update(keyword_pages['index'])
        
        # Detect by page reference patterns
        page_ref_analysis = self._detect_by_page_references(page_blocks)
        results['page_reference_blocks'] = page_ref_analysis['blocks']
        results['toc_pages'].update(page_ref_analysis['suspected_pages'])
        
        # Detect by numbering patterns
        numbering_analysis = self._detect_by_numbering_patterns(page_blocks)
        results['toc_pages'].update(numbering_analysis['toc_pages'])
        results['toc_patterns'].update(numbering_analysis['patterns'])
        
        # Detect sequential page references
        sequential_analysis = self._detect_sequential_references(page_blocks)
        results['toc_pages'].update(sequential_analysis['pages'])
        
        # Create exclusion regions
        results['excluded_regions'] = self._create_exclusion_regions(
            results['toc_pages'], results['index_pages'], page_blocks
        )
        
        logger.info(f"Detected TOC on {len(results['toc_pages'])} pages, "
                   f"Index on {len(results['index_pages'])} pages")
        
        return results
    
    def _detect_by_keywords(self, page_blocks: Dict[int, List[Dict]]) -> Dict[str, Set[int]]:
        """Detect TOC/Index pages by looking for keyword patterns."""
        toc_pages = set()
        index_pages = set()
        
        for page_num, blocks in page_blocks.items():
            page_texts = [block['text'].lower().strip() for block in blocks]
            page_text_combined = ' '.join(page_texts)
            
            # Check for TOC keywords
            for lang, keywords in self.toc_keywords.items():
                for keyword in keywords:
                    if keyword in page_text_combined:
                        # Verify it's likely a section header, not just mention
                        if self._is_likely_section_header(keyword, page_texts, blocks):
                            if keyword in ['index', 'указатель', '索引', '인덱스']:
                                index_pages.add(page_num)
                            else:
                                toc_pages.add(page_num)
        
        return {'toc': toc_pages, 'index': index_pages}
    
    def _is_likely_section_header(self, keyword: str, page_texts: List[str], 
                                blocks: List[Dict]) -> bool:
        """Check if keyword appears as a section header rather than inline text."""
        for i, text in enumerate(page_texts):
            if keyword in text:
                # Check if it's a standalone line or prominently formatted
                if len(text.strip()) < 50:  # Short line - likely header
                    return True
                
                # Check formatting of corresponding block
                if i < len(blocks):
                    block = blocks[i]
                    # Look for header-like formatting
                    if (block.get('is_bold', False) or 
                        block.get('is_centered', False) or
                        block.get('font_size', 0) > 14):
                        return True
        
        return False
    
    def _detect_by_page_references(self, page_blocks: Dict[int, List[Dict]]) -> Dict[str, Any]:
        """Detect TOC/Index by looking for page reference patterns."""
        page_ref_blocks = []
        suspected_pages = set()
        
        for page_num, blocks in page_blocks.items():
            page_ref_count = 0
            
            for block in blocks:
                text = block['text'].strip()
                
                # Check if text contains page reference patterns
                for pattern in self.page_reference_patterns:
                    if re.search(pattern, text):
                        page_ref_blocks.append({
                            'page': page_num,
                            'text': text,
                            'pattern': pattern,
                            'block': block
                        })
                        page_ref_count += 1
                        break
            
            # If many blocks on a page have page references, likely TOC/Index
            if page_ref_count >= 3:  # Threshold for suspicion
                suspected_pages.add(page_num)
        
        return {
            'blocks': page_ref_blocks,
            'suspected_pages': suspected_pages
        }
    
    def _detect_by_numbering_patterns(self, page_blocks: Dict[int, List[Dict]]) -> Dict[str, Any]:
        """Detect TOC by consistent numbering patterns."""
        toc_pages = set()
        detected_patterns = set()
        
        for page_num, blocks in page_blocks.items():
            pattern_counts = defaultdict(int)
            
            for block in blocks:
                text = block['text'].strip()
                
                # Check for TOC numbering patterns
                for pattern in self.toc_numbering_patterns:
                    if re.match(pattern, text):
                        pattern_counts[pattern] += 1
                        
                        # Also check if followed by page reference
                        if any(re.search(p, text) for p in self.page_reference_patterns):
                            pattern_counts[pattern] += 1  # Extra weight
            
            # If we have consistent numbering patterns, likely TOC
            for pattern, count in pattern_counts.items():
                if count >= 3:  # At least 3 occurrences of same pattern
                    toc_pages.add(page_num)
                    detected_patterns.add(pattern)
        
        return {
            'toc_pages': toc_pages,
            'patterns': detected_patterns
        }
    
    def _detect_sequential_references(self, page_blocks: Dict[int, List[Dict]]) -> Dict[str, Set[int]]:
        """Detect TOC by looking for sequential page number references."""
        suspected_pages = set()
        
        for page_num, blocks in page_blocks.items():
            # Extract all numbers that could be page references
            page_refs = []
            
            for block in blocks:
                text = block['text']
                # Find potential page numbers at end of lines
                matches = re.findall(r'\b(\d+)\s*$', text, re.MULTILINE)
                page_refs.extend([int(m) for m in matches if 1 <= int(m) <= 1000])
            
            # Check if we have sequential or increasing page references
            if len(page_refs) >= 4:  # Need several references
                page_refs.sort()
                
                # Check for sequential or mostly increasing pattern
                increasing_count = 0
                for i in range(1, len(page_refs)):
                    if page_refs[i] > page_refs[i-1]:
                        increasing_count += 1
                
                # If mostly increasing, likely TOC
                if increasing_count / (len(page_refs) - 1) >= 0.7:
                    suspected_pages.add(page_num)
        
        return {'pages': suspected_pages}
    
    def _create_exclusion_regions(self, toc_pages: Set[int], index_pages: Set[int],
                                page_blocks: Dict[int, List[Dict]]) -> List[Dict[str, Any]]:
        """Create exclusion regions for detected TOC/Index pages."""
        exclusion_regions = []
        
        all_excluded_pages = toc_pages.union(index_pages)
        
        for page_num in all_excluded_pages:
            if page_num in page_blocks:
                blocks = page_blocks[page_num]
                
                if blocks:
                    # Create region covering most of the page
                    page_width = blocks[0].get('page_width', 600)
                    page_height = blocks[0].get('page_height', 800)
                    
                    exclusion_regions.append({
                        'type': 'toc' if page_num in toc_pages else 'index',
                        'page': page_num,
                        'bbox': [0, 0, page_width, page_height],
                        'confidence': 0.9,
                        'reason': 'keyword_detection' if page_num in toc_pages else 'index_detection'
                    })
        
        return exclusion_regions
    
    def is_text_in_index_region(self, text_block: Dict[str, Any], 
                               index_results: Dict[str, Any]) -> bool:
        """Check if a text block is in a detected index/TOC region."""
        page_num = text_block['page_number']
        
        # Check if page is marked as TOC/Index
        if (page_num in index_results['toc_pages'] or 
            page_num in index_results['index_pages']):
            return True
        
        # Check if text matches TOC patterns
        text = text_block['text'].strip()
        
        # Check for page reference patterns
        for pattern in self.page_reference_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for TOC numbering patterns
        for pattern in index_results['toc_patterns']:
            if re.match(pattern, text):
                return True
        
        return False
    
    def get_toc_structure(self, text_blocks: List[Dict[str, Any]], 
                         index_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract hierarchical structure from detected TOC sections."""
        toc_structure = []
        
        for page_num in index_results['toc_pages']:
            page_blocks = [b for b in text_blocks if b['page_number'] == page_num]
            
            for block in page_blocks:
                text = block['text'].strip()
                
                # Skip if looks like page reference only
                if re.match(r'^\d+$', text):
                    continue
                
                # Determine heading level from numbering pattern
                level = self._determine_heading_level(text)
                
                if level > 0:
                    # Extract title (remove numbering and page refs)
                    clean_title = self._clean_toc_title(text)
                    
                    if clean_title:
                        toc_structure.append({
                            'level': f'H{level}',
                            'title': clean_title,
                            'page': page_num,
                            'original_text': text
                        })
        
        return toc_structure
    
    def _determine_heading_level(self, text: str) -> int:
        """Determine heading level from TOC numbering."""
        # Level 1: 1., Chapter 1, I., etc.
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
        
        return 0  # No recognizable level
    
    def _clean_toc_title(self, text: str) -> str:
        """Clean TOC entry to extract just the title."""
        # Remove leading numbering
        cleaned = re.sub(r'^[\dIVX]+\.?\s*', '', text)
        cleaned = re.sub(r'^[A-Za-z]+\s+\d+\.?\s*', '', cleaned)
        
        # Remove trailing page references
        for pattern in self.page_reference_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        
        # Remove dot leaders
        cleaned = re.sub(r'\.{3,}', '', cleaned)
        
        return cleaned.strip()


def main():
    """Test the index detector."""
    # This would typically be called with text blocks from TextExtractor
    print("IndexDetector created successfully")
    print("Use with text blocks from TextExtractor for testing")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 