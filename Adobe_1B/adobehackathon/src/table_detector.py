"""
Table and shape detection module.
Identifies tables, figures, and mathematical elements to exclude from heading detection.
"""

import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any, Tuple, Set
import re

logger = logging.getLogger(__name__)


class TableDetector:
    """Detects tables, figures, and shapes in PDF documents."""
    
    def __init__(self):
        self.table_keywords = [
            'table', 'figure', 'chart', 'graph', 'diagram', 'image',
            'fig', 'tab', 'exhibit', 'appendix', 'plate'
        ]
        
        # Multi-language table keywords
        self.multilingual_table_keywords = {
            'hindi': ['तालिका', 'चित्र', 'आकृति', 'ग्राफ'],
            'chinese': ['表', '图', '表格', '图表', '图形'],
            'japanese': ['表', '図', '表格', 'グラフ', '図表'],
            'korean': ['표', '그림', '도표', '그래프', '도형'],
            'spanish': ['tabla', 'figura', 'gráfico', 'diagrama'],
            'french': ['tableau', 'figure', 'graphique', 'diagramme'],
            'german': ['tabelle', 'abbildung', 'grafik', 'diagramm'],
            'russian': ['таблица', 'рисунок', 'график', 'диаграмма'],
            'arabic': ['جدول', 'شكل', 'رسم', 'مخطط'],
            'turkish': ['tablo', 'şekil', 'grafik', 'diyagram']
        }
    
    def detect_tables_and_shapes(self, pdf_path: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        Detect all tables and shapes across all pages.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary mapping page numbers to list of table/shape regions
        """
        logger.info(f"Detecting tables and shapes in: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            all_regions = {}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                try:
                    regions = self._detect_page_regions(page, page_num + 1)
                    if regions:
                        all_regions[page_num + 1] = regions
                except Exception as page_error:
                    logger.warning(f"Error processing page {page_num + 1}: {page_error}")
                    continue
                    
            doc.close()
            logger.info(f"Detected table/shape regions on {len(all_regions)} pages")
            return all_regions
            
        except Exception as e:
            logger.error(f"Error detecting tables in {pdf_path}: {e}")
            return {}
    
    def _detect_page_regions(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Detect table and shape regions on a single page."""
        regions = []
        
        # Method 1: PyMuPDF built-in table detection
        tables = page.find_tables()
        for table in tables:
            bbox = table.bbox
            regions.append({
                'type': 'table',
                'bbox': bbox,
                'page': page_num,
                'confidence': 0.9,
                'method': 'pymupdf_builtin'
            })
        
        # Method 2: Shape and drawing detection
        drawings = page.get_drawings()
        if drawings:
            # Group nearby drawings into regions
            drawing_regions = self._group_drawings(drawings)
            for region in drawing_regions:
                regions.append({
                    'type': 'drawing',
                    'bbox': region['bbox'],
                    'page': page_num,
                    'confidence': 0.8,
                    'method': 'shape_analysis'
                })
        
        # Method 3: Text pattern-based table detection
        text_tables = self._detect_text_tables(page, page_num)
        regions.extend(text_tables)
        
        # Method 4: Caption-based detection
        caption_regions = self._detect_caption_regions(page, page_num)
        regions.extend(caption_regions)
        
        return regions
    
    def _group_drawings(self, drawings: List[Dict]) -> List[Dict[str, Any]]:
        """Group nearby drawings into larger regions."""
        if not drawings:
            return []
        
        regions = []
        processed = set()
        
        for i, drawing in enumerate(drawings):
            if i in processed:
                continue
                
            # Get bounding box for this drawing
            points = []
            for item in drawing.get('items', []):
                if 'rect' in item:
                    rect = item['rect']
                    points.extend([rect[:2], rect[2:]])
                elif 'quad' in item:
                    points.extend(item['quad'])
            
            if not points:
                continue
                
            # Find min/max coordinates with type safety
            try:
                x_coords = [float(p[0]) for p in points if len(p) >= 2]
                y_coords = [float(p[1]) for p in points if len(p) >= 2]
                
                if not x_coords or not y_coords:
                    continue
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
            except (ValueError, TypeError, IndexError):
                # Skip if coordinates can't be parsed
                continue
            
            # Look for nearby drawings to group
            group_bbox = [min_x, min_y, max_x, max_y]
            processed.add(i)
            
            for j, other_drawing in enumerate(drawings):
                if j <= i or j in processed:
                    continue
                
                # Check if drawings are close enough to group
                if self._should_group_drawings(drawing, other_drawing):
                    processed.add(j)
                    # Expand bounding box
                    # Similar logic for other_drawing points...
            
            # Only add significant regions
            width = max_x - min_x
            height = max_y - min_y
            if width > 50 and height > 20:  # Minimum size threshold
                regions.append({
                    'bbox': group_bbox,
                    'area': width * height
                })
        
        return regions
    
    def _should_group_drawings(self, drawing1: Dict, drawing2: Dict) -> bool:
        """Determine if two drawings should be grouped together."""
        # Simplified proximity check
        # In practice, would implement proper spatial clustering
        return True  # Placeholder
    
    def _detect_text_tables(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Detect tables based on text patterns and alignment."""
        regions = []
        
        # Get text with position information
        text_dict = page.get_text("dict")
        
        # Look for text blocks with tabular patterns
        aligned_blocks = []
        
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                
                # Check for table-like patterns
                if self._is_table_like_text(line_text):
                    aligned_blocks.append({
                        'text': line_text,
                        'bbox': line["bbox"],
                        'spans': len(line["spans"])
                    })
        
        # Group aligned blocks into table regions
        if len(aligned_blocks) >= 3:  # Minimum rows for a table
            table_regions = self._group_aligned_blocks(aligned_blocks)
            for region in table_regions:
                regions.append({
                    'type': 'text_table',
                    'bbox': region['bbox'],
                    'page': page_num,
                    'confidence': 0.7,
                    'method': 'text_pattern'
                })
        
        return regions
    
    def _is_table_like_text(self, text: str) -> bool:
        """Check if text looks like it belongs to a table."""
        # Multiple tabs or significant spacing
        if '\t' in text or '  ' in text:
            return True
        
        # Numeric patterns common in tables
        if re.search(r'\d+\.\d+.*\d+\.\d+', text):
            return True
        
        # Column-like patterns
        if re.search(r'\b\w+\s+\w+\s+\w+\s+\w+\b', text):
            return True
        
        return False
    
    def _group_aligned_blocks(self, blocks: List[Dict]) -> List[Dict[str, Any]]:
        """Group aligned text blocks into table regions."""
        # Simplified grouping - in practice would use proper clustering
        if not blocks:
            return []
        
        # Calculate bounding box for all blocks
        all_bboxes = [block['bbox'] for block in blocks]
        
        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[3] for bbox in all_bboxes)
        
        return [{
            'bbox': [min_x, min_y, max_x, max_y],
            'block_count': len(blocks)
        }]
    
    def _detect_caption_regions(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Detect figure/table captions and their associated regions."""
        regions = []
        
        text_dict = page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                
                if self._is_caption_text(line_text):
                    # Found a caption - estimate the figure/table region
                    caption_bbox = line["bbox"]
                    
                    # Look for associated content above or below
                    estimated_region = self._estimate_caption_region(
                        page, caption_bbox, line_text
                    )
                    
                    if estimated_region:
                        regions.append({
                            'type': 'caption_region',
                            'bbox': estimated_region,
                            'page': page_num,
                            'confidence': 0.6,
                            'method': 'caption_analysis',
                            'caption_text': line_text.strip()
                        })
        
        return regions
    
    def _is_caption_text(self, text: str) -> bool:
        """Check if text looks like a caption."""
        text_lower = text.lower().strip()
        
        # Check for caption patterns
        caption_patterns = [
            r'^(figure|fig|table|tab)\s*\d+',  # Figure 1, Table 2
            r'^(图|表|図)\s*\d+',  # Chinese/Japanese
            r'^(таблица|рисунок)\s*\d+',  # Russian
        ]
        
        for pattern in caption_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Check multilingual keywords
        for lang, keywords in self.multilingual_table_keywords.items():
            for keyword in keywords:
                if text_lower.startswith(keyword):
                    return True
        
        return False
    
    def _estimate_caption_region(self, page: fitz.Page, caption_bbox: List[float], 
                               caption_text: str) -> List[float]:
        """Estimate the region associated with a caption."""
        # Simplified estimation - look for content above/below caption
        page_height = page.rect.height
        
        # Assume figure/table is above caption
        if "figure" in caption_text.lower() or "fig" in caption_text.lower():
            # Estimate figure region above caption
            estimated_height = min(200, caption_bbox[1])  # Up to 200 points above
            return [
                caption_bbox[0] - 50,  # Extend left
                caption_bbox[1] - estimated_height,  # Above caption
                caption_bbox[2] + 50,  # Extend right  
                caption_bbox[3]  # Include caption
            ]
        
        # Assume table might be above or below
        estimated_height = 150
        return [
            caption_bbox[0] - 30,
            caption_bbox[1] - estimated_height,
            caption_bbox[2] + 30,
            caption_bbox[3] + estimated_height
        ]
    
    def is_text_in_table_region(self, text_bbox: List[float], 
                               table_regions: List[Dict[str, Any]]) -> bool:
        """Check if a text block overlaps with any table/shape region."""
        for region in table_regions:
            if self._bbox_overlap(text_bbox, region['bbox']):
                return True
        return False
    
    def _bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Check for overlap
        return not (x1_max < x2_min or x2_max < x1_min or 
                   y1_max < y2_min or y2_max < y1_min)


def main():
    """Test the table detector."""
    detector = TableDetector()
    
    # Test with round.pdf
    regions = detector.detect_tables_and_shapes("round.pdf")
    
    print(f"Detected table/shape regions on {len(regions)} pages")
    
    for page_num, page_regions in regions.items():
        print(f"\nPage {page_num}: {len(page_regions)} regions")
        for i, region in enumerate(page_regions):
            print(f"  Region {i+1}: {region['type']} - {region['method']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 