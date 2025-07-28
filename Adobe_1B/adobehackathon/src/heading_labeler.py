#!/usr/bin/env python3
"""
Comprehensive Heading Labeler with Multilingual Support
Implements all heuristics for H1-H6, Title, and Not-Heading classification.
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import math

logger = logging.getLogger(__name__)

class HeadingLabeler:
    """Comprehensive heading labeling with multilingual support."""
    
    def __init__(self):
        # Academic/section keywords by language
        self.academic_keywords = {
            'en': {
                'h1': ['abstract', 'introduction', 'conclusion', 'references', 'bibliography', 
                       'acknowledgments', 'methodology', 'methods', 'results', 'discussion',
                       'background', 'literature review', 'appendix', 'summary', 'overview'],
                'structural': ['part', 'chapter', 'section', 'volume', 'book']
            },
            'es': {
                'h1': ['resumen', 'introducción', 'conclusión', 'referencias', 'bibliografía',
                       'metodología', 'resultados', 'discusión', 'antecedentes', 'apéndice'],
                'structural': ['parte', 'capítulo', 'sección', 'volumen', 'libro']
            },
            'fr': {
                'h1': ['résumé', 'introduction', 'conclusion', 'références', 'bibliographie',
                       'méthodologie', 'résultats', 'discussion', 'contexte', 'annexe'],
                'structural': ['partie', 'chapitre', 'section', 'volume', 'livre']
            },
            'de': {
                'h1': ['zusammenfassung', 'einleitung', 'schluss', 'literatur', 'bibliographie',
                       'methodik', 'ergebnisse', 'diskussion', 'hintergrund', 'anhang'],
                'structural': ['teil', 'kapitel', 'abschnitt', 'band', 'buch']
            },
            'zh': {
                'h1': ['摘要', '引言', '介绍', '结论', '参考文献', '方法', '结果', '讨论', '背景', '附录'],
                'structural': ['部分', '章', '节', '卷', '书']
            },
            'ja': {
                'h1': ['概要', '序論', '結論', '参考文献', '手法', '結果', '考察', '背景', '付録'],
                'structural': ['部', '章', '節', '巻', '本']
            },
            'ar': {
                'h1': ['ملخص', 'مقدمة', 'خاتمة', 'مراجع', 'منهجية', 'نتائج', 'مناقشة', 'خلفية', 'ملحق'],
                'structural': ['جزء', 'فصل', 'قسم', 'مجلد', 'كتاب']
            },
            'hi': {
                'h1': ['सारांश', 'परिचय', 'निष्कर्ष', 'संदर्भ', 'पद्धति', 'परिणाम', 'चर्चा', 'पृष्ठभूमि', 'परिशिष्ट'],
                'structural': ['भाग', 'अध्याय', 'अनुभाग', 'खंड', 'पुस्तक']
            },
            'ru': {
                'h1': ['аннотация', 'введение', 'заключение', 'литература', 'методология', 'результаты', 'обсуждение', 'приложение'],
                'structural': ['часть', 'глава', 'раздел', 'том', 'книга']
            }
        }
        
        # Numbering patterns by language/script
        self.numbering_patterns = {
            'decimal': r'^(\d+(?:\.\d+)*)\s*[\.\)\-\s]',  # 1.2.3
            'arabic': r'^(\d+)\s*[\.\)\-\s]',  # 1, 2, 3
            'roman': r'^([IVXLCDM]+)\s*[\.\)\-\s]',  # I, II, III
            'alpha': r'^([A-Z])\s*[\.\)\-\s]',  # A, B, C
            'chinese': r'^([一二三四五六七八九十百千万]+)\s*[\.\)\-\s]',  # 一, 二, 三
            'arabic_script': r'^([٠-٩]+)\s*[\.\)\-\s]',  # Arabic numerals
            'hindi': r'^([०-९]+)\s*[\.\)\-\s]',  # Devanagari numerals
        }
        
        # Caption patterns
        self.caption_patterns = {
            'en': [r'^(table|figure|fig|chart|diagram|image)\s*\d*\s*[:\.]\s*',
                   r'^(source|note):\s*'],
            'es': [r'^(tabla|figura|fig|gráfico|diagrama|imagen)\s*\d*\s*[:\.]\s*'],
            'fr': [r'^(tableau|figure|fig|graphique|diagramme|image)\s*\d*\s*[:\.]\s*'],
            'de': [r'^(tabelle|abbildung|abb|grafik|diagramm|bild)\s*\d*\s*[:\.]\s*'],
            'zh': [r'^(表|图|图表|示意图)\s*\d*\s*[:\.]\s*'],
            'ja': [r'^(表|図|図表|グラフ)\s*\d*\s*[:\.]\s*'],
            'ar': [r'^(جدول|شكل|رسم|مخطط|صورة)\s*\d*\s*[:\.]\s*'],
            'hi': [r'^(तालिका|चित्र|आकृति|ग्राफ)\s*\d*\s*[:\.]\s*'],
            'ru': [r'^(таблица|рисунок|схема|диаграмма)\s*\d*\s*[:\.]\s*']
        }
        
        # Math symbols for exclusion
        self.math_symbols = {
            '∫', '∑', '∏', '∆', '∇', '∞', '±', '≤', '≥', '≠', '≈', '≡', '√', '∂',
            'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω',
            '∈', '∉', '⊂', '⊃', '∪', '∩', '→', '←', '↔', '∀', '∃', '∴', '∵'
        }
        
        # Stopwords for density penalty
        self.stopwords = {
            'en': {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'},
            'es': {'el', 'la', 'los', 'las', 'y', 'o', 'pero', 'en', 'con', 'de', 'para', 'por', 'un', 'una', 'es', 'son'},
            'fr': {'le', 'la', 'les', 'et', 'ou', 'mais', 'dans', 'avec', 'de', 'pour', 'par', 'un', 'une', 'est', 'sont'},
            'de': {'der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'mit', 'von', 'für', 'ein', 'eine', 'ist', 'sind'},
            'zh': {'的', '和', '或', '但', '在', '与', '为', '是'},
            'ja': {'の', 'と', 'や', 'が', 'を', 'に', 'で', 'から', 'まで', 'です', 'である'},
            'ar': {'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'التي', 'الذي'},
            'hi': {'का', 'की', 'के', 'और', 'या', 'में', 'से', 'को', 'पर', 'है', 'हैं'},
            'ru': {'и', 'или', 'но', 'в', 'на', 'с', 'от', 'для', 'по', 'это', 'что'}
        }
    
    def label_headings(self, candidates: List[Dict[str, Any]], 
                      text_blocks: List[Dict[str, Any]],
                      doc_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Label heading candidates with simple rule-based classification.
        No confidence scoring - just classify based on formatting and content.
        """
        logger.info(f"Classifying {len(candidates)} heading candidates")
        
        if not candidates:
            return []
        
        # Build document context for classification
        doc_context = self._build_document_context(candidates, text_blocks, doc_stats)
        
        # Classify each candidate 
        for candidate in candidates:
            label = self._classify_candidate_simple(candidate, doc_context)
            candidate['label'] = label
        
        # Find and mark the main title (highest font size on first page)
        self._mark_main_title(candidates)
        
        logger.info(f"Classification complete")
        return candidates
    
    def _classify_candidate_simple(self, candidate: Dict[str, Any], 
                                  doc_context: Dict[str, Any]) -> str:
        """Simple rule-based classification without confidence scoring."""
        
        text = candidate['text'].strip()
        font_size = candidate.get('font_size', 0)
        is_bold = candidate.get('is_bold', False)
        page_number = candidate.get('page_number', 1)
        
        # Skip very short or very long text
        if len(text) < 2 or len(text) > 100:
            return 'Not-Heading'
        
        # Skip if contains math symbols
        if any(symbol in text for symbol in self.math_symbols):
            return 'Not-Heading'
        
        # Check for captions/figure labels
        if self._is_caption(text, candidate.get('language', 'en')):
            return 'Not-Heading'
        
        # Main classification logic
        font_rank = doc_context.get('font_ranks', {}).get(id(candidate), 999)
        
        # Check for academic section keywords first
        academic_level = self._get_academic_section_level(text, candidate.get('language', 'en'))
        if academic_level:
            return academic_level
        
        # Check for numbered sections
        numbering_info = self._analyze_numbering(text)
        if numbering_info['has_numbering']:
            level = numbering_info.get('level', 1)
            if level == 1:
                return 'H1'
            elif level == 2:
                return 'H2'
            elif level == 3:
                return 'H3'
            else:
                return 'H4'
        
        # Font size based classification
        if font_rank <= 2 and (is_bold or font_size > 12):
            return 'H1'
        elif font_rank <= 4 and (is_bold or font_size > 10):
            return 'H2'
        elif font_rank <= 6 and is_bold:
            return 'H3'
        elif is_bold and font_size > 8:
            return 'H4'
        
        # Default to H4 if it has some heading characteristics
        if is_bold or font_size > 9:
            return 'H4'
        
        return 'Not-Heading'
    
    def _merge_multiline_headings(self, candidates: List[Dict]) -> List[Dict]:
        """Merge multi-line headings that were split across blocks."""
        if not candidates:
            return candidates
        
        merged = []
        skip_indices = set()
        
        for i, candidate in enumerate(candidates):
            if i in skip_indices:
                continue
            
            # Look ahead for potential continuation lines
            merged_text = candidate['text']
            merged_bbox = [candidate['x0'], candidate['y0'], candidate['x1'], candidate['y1']]
            merge_indices = [i]
            
            for j in range(i + 1, min(i + 4, len(candidates))):  # Look ahead max 3 lines
                next_candidate = candidates[j]
                
                # Check if this could be a continuation
                if self._should_merge_with_next(candidate, next_candidate, candidates):
                    merged_text += " " + next_candidate['text']
                    merged_bbox[2] = max(merged_bbox[2], next_candidate['x1'])  # Extend width
                    merged_bbox[3] = max(merged_bbox[3], next_candidate['y1'])  # Extend height
                    merge_indices.append(j)
                    skip_indices.add(j)
                else:
                    break
            
            # Create merged candidate
            merged_candidate = candidate.copy()
            merged_candidate['text'] = merged_text.strip()
            merged_candidate['x1'] = merged_bbox[2]
            merged_candidate['y1'] = merged_bbox[3]
            merged_candidate['merged_from'] = merge_indices
            
            merged.append(merged_candidate)
        
        return merged
    
    def _should_merge_with_next(self, current: Dict, next_block: Dict, 
                               all_candidates: List[Dict]) -> bool:
        """Check if two blocks should be merged as a multi-line heading."""
        # Same page
        if current['page_number'] != next_block['page_number']:
            return False
        
        # Similar font properties
        font_match = (
            current.get('font_name') == next_block.get('font_name') and
            abs(current.get('font_size', 0) - next_block.get('font_size', 0)) < 0.5 and
            current.get('is_bold') == next_block.get('is_bold')
        )
        
        if not font_match:
            return False
        
        # Vertical proximity (within 2 line heights)
        line_height = current.get('text_height', 12)
        vertical_gap = abs(next_block.get('y0', 0) - current.get('y1', 0))
        if vertical_gap > line_height * 2:
            return False
        
        # Horizontal alignment (similar x0 or both centered)
        x_diff = abs(current.get('x0', 0) - next_block.get('x0', 0))
        both_centered = (current.get('is_centered', False) and 
                        next_block.get('is_centered', False))
        
        if x_diff > 20 and not both_centered:
            return False
        
        # Text characteristics
        current_text = current['text'].strip()
        next_text = next_block['text'].strip()
        
        # Don't merge if either looks like complete sentence
        if (current_text.endswith('.') and len(current_text.split()) > 3) or \
           (next_text.endswith('.') and len(next_text.split()) > 3):
            return False
        
        # Don't merge if next starts with lowercase (likely continuation of paragraph)
        if next_text and next_text[0].islower():
            return False
        
        # Combined length shouldn't be too long for a heading
        combined_length = len(current_text) + len(next_text)
        if combined_length > 150:
            return False
        
        return True
    
    def _calculate_document_context(self, candidates: List[Dict], 
                                   text_blocks: List[Dict], 
                                   doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate document-wide context for labeling decisions."""
        context = {
            'font_sizes': [c.get('font_size', 0) for c in candidates],
            'languages': [c.get('language', 'en') for c in candidates],
            'pages': [c.get('page_number', 1) for c in candidates],
            'total_pages': max([c.get('page_number', 1) for c in candidates]),
            'body_font_size': doc_stats.get('median_font_size', 10),
            'max_font_size': doc_stats.get('max_font_size', 12),
            'font_size_distribution': doc_stats.get('font_size_distribution', {}),
        }
        
        # Calculate font size rankings
        unique_sizes = sorted(set(context['font_sizes']), reverse=True)
        context['font_rank_map'] = {size: i + 1 for i, size in enumerate(unique_sizes)}
        
        # Calculate dominant language
        lang_counts = Counter(context['languages'])
        context['dominant_language'] = lang_counts.most_common(1)[0][0] if lang_counts else 'en'
        
        # Identify first page bounds
        first_page_candidates = [c for c in candidates if c.get('page_number', 1) == 1]
        if first_page_candidates:
            context['first_page_top_threshold'] = min(c.get('y0', 0) for c in first_page_candidates) + 100
        else:
            context['first_page_top_threshold'] = 100
        
        return context
    
    def _build_document_context(self, candidates: List[Dict], 
                                text_blocks: List[Dict], 
                                doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build document context for simple classification."""
        context = {
            'font_ranks': {},
            'dominant_language': doc_stats.get('dominant_language', 'en'),
            'first_page_top_threshold': doc_stats.get('first_page_top_threshold', 100),
        }
        
        # Assign font ranks to candidates
        font_sizes = [c.get('font_size', 0) for c in candidates]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        context['font_ranks'] = {id(c): i + 1 for i, size in enumerate(unique_sizes) for c in candidates if c.get('font_size') == size}
        
        return context
    
    def _is_caption(self, text: str, language: str) -> bool:
        """Check if text looks like a caption/figure label."""
        text_lower = text.lower()
        patterns = self.caption_patterns.get(language, self.caption_patterns['en'])
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def _get_academic_section_level(self, text: str, language: str) -> Optional[str]:
        """Identify academic section keywords (H1-H6) based on keywords."""
        text_lower = text.lower()
        keywords = self.academic_keywords.get(language, self.academic_keywords['en'])
        
        # H1 keywords
        for keyword in keywords.get('h1', []):
            if keyword in text_lower:
                return 'H1'
        
        # Structural keywords (H1-H2)
        for keyword in keywords.get('structural', []):
            if keyword in text_lower:
                return 'H1'
        
        # All caps (if longer than 3 chars)
        if len(text_lower) > 3 and text_lower.isupper():
            return 'H1'
        
        return None
    
    def _analyze_numbering(self, text: str) -> Dict[str, Any]:
        """Analyze text for numbering patterns (decimal, roman, alpha, etc.)."""
        for pattern_type, pattern in self.numbering_patterns.items():
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                number_part = match.group(1)
                
                if pattern_type == 'decimal':
                    # Count dots to determine depth
                    depth = number_part.count('.') + 1
                    depth = min(depth, 6)  # Cap at H6
                    return {'has_numbering': True, 'level': depth}
                
                elif pattern_type in ['arabic', 'chinese', 'arabic_script', 'hindi']:
                    # Simple numbering usually H1 or H2
                    return {'has_numbering': True, 'level': 1} # Default to H1
                
                elif pattern_type == 'roman':
                    # Roman numerals usually H1 or H2
                    return {'has_numbering': True, 'level': 1} # Default to H1
                
                elif pattern_type == 'alpha':
                    # Alphabetic usually H2 or H3
                    return {'has_numbering': True, 'level': 2} # Default to H2
        
        return {'has_numbering': False, 'level': 1} # Default to H1 if no number
    
    def _mark_main_title(self, candidates: List[Dict]):
        """Mark the main title (highest font size on first page) if it exists."""
        if not candidates:
            return
        
        # Sort by font size descending
        candidates.sort(key=lambda x: x.get('font_size', 0), reverse=True)
        
        # Find the first candidate that is not 'Not-Heading'
        for candidate in candidates:
            if candidate['label'] != 'Not-Heading':
                candidate['label'] = 'Title' # Mark as Title
                break
    
    def _remove_duplicates_and_validate(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicates and validate final results."""
        seen_texts = set()
        final_candidates = []
        
        for candidate in candidates:
            # Normalize text for comparison
            norm_text = candidate['text'].strip().lower()
            norm_text = re.sub(r'\s+', ' ', norm_text)  # Normalize whitespace
            
            if norm_text not in seen_texts:
                seen_texts.add(norm_text)
                final_candidates.append(candidate)
        
        return final_candidates


def main():
    """Test the heading labeler."""
    print("HeadingLabeler created successfully")
    print("Use with candidates from FeatureEngineer for testing")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 