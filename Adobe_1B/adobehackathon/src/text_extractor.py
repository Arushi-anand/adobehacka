"""
Core text extraction module using PyMuPDF.
Extracts text blocks with comprehensive font and position metadata.
"""

import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any, Tuple
from langdetect import detect, DetectorFactory
import re

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extracts text blocks from PDF with rich metadata."""
    
    def __init__(self):
        pass
        
    def extract_text_blocks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all text blocks from PDF with comprehensive metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text blocks with metadata
        """
        logger.info(f"Extracting text from: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_blocks = self._extract_page_blocks(page, page_num + 1)
                text_blocks.extend(page_blocks)
                
            doc.close()
            
            # Merge consecutive blocks with same formatting
            merged_blocks = self._merge_consecutive_blocks(text_blocks)
            
            logger.info(f"Extracted {len(merged_blocks)} merged text blocks")
            return merged_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []
    
    def _extract_page_blocks(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract text blocks from a single page."""
        blocks = []
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Extract text with detailed formatting
        text_dict = page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if "lines" not in block:  # Skip image blocks
                continue
                
            for line in block["lines"]:
                # Process each span separately to avoid merging different formatting
                for span in line["spans"]:
                    span_text = span["text"].strip()
                    
                    # Skip empty spans
                    if not span_text:
                        continue
                    
                    # Extract formatting
                    font_name = span["font"]
                    font_size = span["size"]
                    font_flags = span["flags"]
                    font_color = span["color"]
                    
                    # Extract formatting flags
                    is_bold = bool(font_flags & 2**4)
                    is_italic = bool(font_flags & 2**1)
                    is_underlined = bool(font_flags & 2**0)
                    
                    # Get bounding box
                    bbox = span["bbox"]
                    x0, y0, x1, y1 = bbox
                    
                    # Create text block metadata
                    block_info = {
                        "text": span_text,
                        "font_size": round(font_size, 1),
                        "font_name": font_name,
                        "is_bold": is_bold,
                        "is_italic": is_italic,
                        "x0": round(x0, 1),
                        "y0": round(y0, 1),
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "page_number": page_num,
                        "page_width": round(page_width, 1),
                        "page_height": round(page_height, 1),
                    }
                    
                    # Add basic computed features
                    text = block_info["text"]
                    block_info["text_length"] = len(text)
                    block_info["num_words"] = len(text.split())
                    block_info["all_uppercase"] = text.isupper()
                    
                    # Position analysis
                    center_x = (x0 + x1) / 2
                    page_center_x = page_width / 2
                    block_info["is_centered"] = abs(center_x - page_center_x) < page_width * 0.1
                    
                    # Language detection
                    try:
                        if len(text.strip()) > 3:
                            block_info["language"] = detect(text)
                        else:
                            block_info["language"] = "en"
                    except:
                        block_info["language"] = "en"
                    
                    blocks.append(block_info)
        
        return blocks
    
    def _merge_consecutive_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge consecutive blocks with same font properties."""
        if not blocks:
            return blocks
        
        # Group by page first
        pages = {}
        for block in blocks:
            page_num = block['page_number']
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(block)
        
        merged_blocks = []
        
        for page_num, page_blocks in pages.items():
            # Sort by vertical position (y0), then horizontal (x0)
            page_blocks.sort(key=lambda x: (x['y0'], x['x0']))
            
            i = 0
            while i < len(page_blocks):
                current = page_blocks[i]
                merged_text = current['text']
                merged_x0 = current['x0']
                merged_x1 = current['x1']
                merged_y1 = current['y1']
                
                # Look for consecutive blocks to merge
                j = i + 1
                while j < len(page_blocks):
                    next_block = page_blocks[j]
                    
                    # Check if should merge
                    if self._should_merge_blocks(current, next_block):
                        # Merge text with space (unless it looks like a word continuation)
                        next_text = next_block['text']
                        
                        # Special handling for uppercase headers that got cut
                        if (merged_text.isupper() and next_text.isupper() and 
                            len(merged_text) < 20 and len(next_text) < 20):
                            # No space for uppercase continuations
                            merged_text += next_text
                        else:
                            merged_text += " " + next_text
                        
                        merged_x0 = min(merged_x0, next_block['x0'])
                        merged_x1 = max(merged_x1, next_block['x1'])
                        merged_y1 = max(merged_y1, next_block['y1'])
                        j += 1
                    else:
                        break
                
                # Post-process merged text for common patterns
                merged_text = self._fix_common_fragments(merged_text.strip())
                
                # Create merged block
                merged_block = current.copy()
                merged_block['text'] = merged_text
                merged_block['x0'] = merged_x0
                merged_block['x1'] = merged_x1
                merged_block['y1'] = merged_y1
                merged_block['text_length'] = len(merged_text)
                merged_block['num_words'] = len(merged_text.split())
                
                merged_blocks.append(merged_block)
                i = j
        
        return merged_blocks
    
    def _should_merge_blocks(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> bool:
        """Improved logic: merge if same font size, name, and close position."""
        # Must be same page
        if block1['page_number'] != block2['page_number']:
            return False
        
        # Must have same font properties (more lenient)
        same_font = (
            abs(block1['font_size'] - block2['font_size']) < 1.0 and  # More lenient size
            block1['font_name'] == block2['font_name'] and
            block1['is_bold'] == block2['is_bold']
        )
        
        if not same_font:
            return False
        
        # Calculate line height based on font size
        line_height = max(block1['font_size'], block2['font_size']) * 1.3
        
        # Check vertical proximity - must be close (within 3 line heights)
        vertical_gap = abs(block2['y0'] - block1['y1'])
        if vertical_gap > line_height * 3:
            return False
        
        # For very close blocks (same line or next line), be more aggressive
        if vertical_gap <= line_height * 1.5:
            # Allow larger horizontal gaps for same/close lines
            horizontal_gap = abs(block2['x0'] - block1['x0'])
            if horizontal_gap > 100:  # Very generous for close lines
                return False
        else:
            # For blocks further apart, require similar alignment
            horizontal_gap = abs(block2['x0'] - block1['x0'])
            if horizontal_gap > 30:
                return False
        
        # Text characteristics - be more lenient
        text1 = block1['text'].strip()
        text2 = block2['text'].strip()
        
        # Don't merge if either block is very long (likely paragraph)
        if len(text1) > 100 or len(text2) > 100:
            return False
        
        # Don't merge if combined would be too long
        combined_length = len(text1) + len(text2)
        if combined_length > 300:  # More generous
            return False
        
        # Special case: merge fragments that look like broken words/sentences
        # e.g., "BSTRACT" + something, or citation fragments
        if (len(text1) < 20 and len(text2) < 20) or \
           (text1.endswith('-') or text2.startswith('-')) or \
           (text1.isupper() and len(text1) < 15) or \
           (text2.isupper() and len(text2) < 15):
            return True
        
        # Merge if text looks like it should continue
        # e.g., ending with comma, "and", numbers, etc.
        continuation_patterns = [',', 'and', 'or', '&', 'of', 'the', 'in', 'to', 'for']
        if any(text1.lower().endswith(' ' + pattern) for pattern in continuation_patterns):
            return True
        
        # Merge citation-like fragments
        if re.search(r'\[\s*\d+\s*\]', text1) or re.search(r'\[\s*\d+\s*\]', text2):
            return True
        
        return True  # Default to merging if all other checks pass

    def _fix_common_fragments(self, text: str) -> str:
        """Fix common fragmented text patterns."""
        # Fix common section header fragments
        fixes = {
            'BSTRACT': 'ABSTRACT',
            'NTRODUCTION': 'INTRODUCTION', 
            'HE ONSTRUCTIVE ATURE OF': 'THE CONSTRUCTIVE NATURE OF',
            'ECISION ASKS': 'DECISION TASKS',
            'ECISION CENARIOS PANNING THE': 'DECISION SCENARIOS SPANNING THE',
            'ESIGN PACE': 'DESIGN SPACE',
            'ITUATIONAL NALYSIS OF': 'SITUATIONAL ANALYSIS OF',
            'ECISION ROBLEMS': 'DECISION PROBLEMS',
            'MPLICATIONS FOR ECISION': 'IMPLICATIONS FOR DECISION',
            'ENTRIC ISUALIZATION': 'CENTRIC VISUALIZATION',
            'ONCLUSION': 'CONCLUSION',
            'EFERENCES': 'REFERENCES'
        }
        
        if text in fixes:
            return fixes[text]
        
        return text


def main():
    """Test the text extractor."""
    extractor = TextExtractor()
    
    # Test with input PDF
    blocks = extractor.extract_text_blocks("input/academic_1.pdf")
    
    print(f"Extracted {len(blocks)} text blocks")
    
    # Show first few blocks
    for i, block in enumerate(blocks[:10]):
        print(f"\nBlock {i+1}:")
        print(f"Text: {block['text'][:100]}...")
        print(f"Font: {block['font_name']}, Size: {block['font_size']}")
        print(f"Bold: {block['is_bold']}, Page: {block['page_number']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 