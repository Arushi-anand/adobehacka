"""
Inference pipeline for PDF heading extraction.
Implements simplified title detection: most prominent heading = title, rest = outline.
"""

import logging
import json
import os
import re
from typing import Dict, List, Any

from text_extractor import TextExtractor
from feature_engineer import FeatureEngineer
from table_detector import TableDetector
from header_footer_detector import HeaderFooterDetector
from index_detector import IndexDetector

logger = logging.getLogger(__name__)


class PDFHeadingExtractor:
    """Main inference pipeline for PDF heading extraction."""
    
    def __init__(self):
        self.extractor = TextExtractor()
        self.table_detector = TableDetector()
        self.header_detector = HeaderFooterDetector()
        self.index_detector = IndexDetector()
        self.feature_engineer = FeatureEngineer()
        
        # Form detection patterns
        self.form_patterns = [
            r'Name:\s*_+', r'Date:\s*_+', r'Signature:\s*_+',
            r'Application', r'Form\s+\d+', r'Please\s+(fill|complete)',
            r'□', r'☐', r'__+', r'\.{3,}',
            r'First\s+Name', r'Last\s+Name', r'Address',
            r'Phone\s+(Number|No)', r'Email', r'SSN', r'ID\s+Number'
        ]
    
    def extract_headings(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract headings from PDF and return JSON schema.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with title and outline in required schema format
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Step 1: Extract text blocks
            text_blocks = self.extractor.extract_text_blocks(pdf_path)
            if not text_blocks:
                logger.warning("No text blocks extracted")
                return {"title": "", "outline": []}
            
            logger.info(f"Extracted {len(text_blocks)} text blocks")
            
            # Step 2: Detect exclusions
            table_regions = self.table_detector.detect_tables_and_shapes(pdf_path)
            headers_footers = self.header_detector.detect_headers_footers(text_blocks)
            index_results = self.index_detector.detect_index_sections(text_blocks)
            
            # Step 3: Extract features
            features_df = self.feature_engineer.extract_features(text_blocks)
            
            # Step 4: Filter candidates and score
            candidates = self._filter_and_score_candidates(
                text_blocks, features_df, table_regions, headers_footers, index_results
            )
            
            logger.info(f"Found {len(candidates)} heading candidates")
            
            # Step 5: Generate JSON output with simplified title detection
            json_output = self._generate_json_output(candidates, text_blocks)
            
            logger.info(f"Generated output - Title: '{json_output['title']}', Outline: {len(json_output['outline'])} items")
            
            return json_output
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {"title": "", "outline": []}
    
    def _filter_and_score_candidates(self, text_blocks: List[Dict], features_df,
                                   table_regions: Dict, headers_footers: Dict,
                                   index_results: Dict) -> List[Dict[str, Any]]:
        """Filter out exclusions and score heading candidates."""
        candidates = []
        
        for i, (idx, row) in enumerate(features_df.iterrows()):
            if i >= len(text_blocks):
                continue
                
            block = text_blocks[i]
            
            # Apply exclusions
            if self._should_exclude_block(block, row, table_regions, headers_footers, index_results):
                continue
            
            # Calculate heading score
            score = self._calculate_heading_score(block, row)
            
            if score > 0:  # Only include if positive score
                candidates.append({
                    'text': block['text'],
                    'font_size': block['font_size'],
                    'is_bold': block.get('is_bold', False),
                    'is_centered': block.get('is_centered', False),
                    'page': block['page_number'],
                    'score': score,
                    'features': row.to_dict()
                })
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    
    def _should_exclude_block(self, block: Dict, row, table_regions: Dict,
                            headers_footers: Dict, index_results: Dict) -> bool:
        """Check if block should be excluded from heading candidates."""
        
        # Check table regions
        page_regions = table_regions.get(block['page_number'], [])
        if self.table_detector.is_text_in_table_region(
            [block['x0'], block['y0'], block['x1'], block['y1']], page_regions
        ):
            return True
        
        # Check headers/footers
        if self.header_detector.is_header_footer_text(block['text'], headers_footers):
            return True
        
        # Check index/TOC
        if self.index_detector.is_text_in_index_region(block, index_results):
            return True
        
        # Check mathematical content
        if block.get('contains_math_symbols', False) or block.get('is_equation', False):
            return True
        
        # Check semantic filtering (pseudo-headings)
        if row.get('is_pseudo_heading', False):
            return True
        
        return False
    
    def _calculate_heading_score(self, block: Dict, row) -> float:
        """Calculate heading confidence score for a block."""
        score = 0
        
        # Font size (most important factor)
        font_rank = row.get('font_size_rank', 999)
        if font_rank <= 2:
            score += 40
        elif font_rank <= 5:
            score += 20
        elif font_rank <= 8:
            score += 10
        
        # Formatting indicators
        if block.get('is_bold', False):
            score += 25
        if block.get('is_centered', False):
            score += 15
        
        # Spacing indicators
        if row.get('has_significant_space_before', False):
            score += 10
        if row.get('has_significant_space_after', False):
            score += 10
        
        # Academic terms (positive for headings)
        if row.get('contains_academic_term', False):
            score += 15
        
        # Numbering patterns
        if row.get('has_arabic_numbering', False):
            score += 20
        if row.get('has_multilevel_numbering', False):
            score += 15
        
        # Semantic adjustments
        semantic_adj = row.get('semantic_confidence_adj', 0)
        score += semantic_adj
        
        # Position bonuses
        if row.get('is_first_page', False):
            score += 5
        if row.get('is_top_of_page', False):
            score += 5
        
        # Length penalties
        text_length = len(block['text'])
        if text_length > 200:
            score -= 15  # Very long text unlikely to be heading
        elif text_length < 10:
            score -= 10  # Very short text also suspicious
        
        return max(0, score)
    
    def _detect_form_type(self, candidates: List[Dict], text_blocks: List[Dict]) -> bool:
        """Detect if document is form-type with only title, no headings."""
        
        form_indicators = 0
        
        # Count form-like patterns
        for block in text_blocks:
            text = block['text']
            for pattern in self.form_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    form_indicators += 1
                    break  # Only count once per block
        
        # Form detection criteria
        is_form = (
            form_indicators >= 3 and  # Multiple form patterns
            len(candidates) <= 5 and  # Few heading candidates
            any('form' in c['text'].lower() for c in candidates[:3])  # "Form" in top candidates
        )
        
        return is_form
    
    def _classify_heading_level(self, candidate: Dict, all_candidates: List[Dict]) -> str:
        """Classify heading level based on font size rank and score."""
        font_rank = candidate['features'].get('font_size_rank', 999)
        score = candidate['score']
        
        # Simple classification based on font size ranking and score
        if font_rank <= 2 and score >= 60:
            return 'H1'
        elif font_rank <= 4 and score >= 40:
            return 'H2'
        elif font_rank <= 6 and score >= 30:
            return 'H3'
        else:
            return 'H4'
    
    def _generate_json_output(self, candidates: List[Dict], text_blocks: List[Dict]) -> Dict[str, Any]:
        """Generate final JSON schema with NO title detection (title left empty)."""
        
        # Title is always empty for now
        title_text = ""
        
        # Detect if form-type (title only, no structure)
        is_form_type = self._detect_form_type(candidates, text_blocks)
        
        if is_form_type:
            return {
                "title": title_text,
                "outline": []  # Forms have no heading hierarchy
            }
        
        # Regular document: all candidates become outline
        filtered_candidates = [c for c in candidates if c['score'] >= 20]
        filtered_candidates.sort(key=lambda x: (x['page'], -x['score']))
        outline = []
        for candidate in filtered_candidates:
            level = self._classify_heading_level(candidate, candidates)
            outline.append({
                "level": level,
                "text": candidate['text'],
                "page": candidate['page']
            })
        return {
            "title": title_text,
            "outline": outline
        }
    
    def _find_best_title_candidate(self, candidates: List[Dict]) -> Dict[str, Any]:
        """Find the best title candidate: first largest heading on first page."""
        if not candidates:
            return None
        # Filter to first page
        first_page_candidates = [c for c in candidates if c['page'] == 1]
        if not first_page_candidates:
            return candidates[0]
        # Find largest font size among first page candidates
        max_font = max(c['font_size'] for c in first_page_candidates)
        largest_candidates = [c for c in first_page_candidates if c['font_size'] == max_font]
        # Return the first one (top-most on page)
        return largest_candidates[0]
    
    def process_pdf_to_json(self, pdf_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Process PDF and save/return JSON output.
        
        Args:
            pdf_path: Input PDF path
            output_path: Optional output JSON path
            
        Returns:
            JSON schema dictionary
        """
        # Extract headings
        result = self.extract_headings(pdf_path)
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved output to {output_path}")
        
        return result


def main():
    """Test the inference pipeline."""
    extractor = PDFHeadingExtractor()
    
    # Test with round.pdf
    result = extractor.extract_headings("round.pdf")
    
    print("📋 Generated JSON Output:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 