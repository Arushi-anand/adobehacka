"""
PDF Heading Extraction System with ML Integration
Processes PDFs and uses trained ML model for accurate heading classification
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import re
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Add src to Python path
sys.path.append('src')

from text_extractor import TextExtractor
from table_detector import TableDetector
from header_footer_detector import HeaderFooterDetector
from heading_labeler import HeadingLabeler
from feature_engineer import FeatureEngineer

# Import ML inference pipeline
sys.path.append('model')
try:
    from model.inference_pipeline import FastHeadingClassifier
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    print("WARNING: ML model not available. Using rule-based classification.")


def process_page_group(page_blocks, feature_engineer):
    """Process a group of text blocks (for multiprocessing)."""
    try:
        if not page_blocks:
            return pd.DataFrame()
        features_df = feature_engineer.extract_features(page_blocks)
        return features_df
    except Exception as e:
        print(f"Error in multiprocessing: {e}")
        return pd.DataFrame()


def process_pdf(pdf_path, text_extractor, table_detector, header_footer_detector, heading_labeler, feature_engineer, ml_classifier, output_dir):
    """Processes a single PDF file."""
    pdf_start_time = time.time()
    print(f"\nProcessing: {pdf_path.name}")
    
    try:
        # Phase 1: Extract and merge text blocks
        print("  Phase 1: Extracting text blocks...")
        text_blocks = text_extractor.extract_text_blocks(str(pdf_path))
        print(f"    Extracted {len(text_blocks)} text blocks")
        
        # Extract features
        print("  Phase 1: Extracting features...")
        feature_start_time = time.time()
        features_df = feature_engineer.extract_features(text_blocks)
        feature_time = time.time() - feature_start_time
        print(f"    Extracted {len(features_df.columns)} features ({feature_time:.2f}s)")
        
        # Detect exclusions
        print("  Phase 1: Detecting exclusions...")
        excluded_info = {}
        
        # Table detection
        table_regions = table_detector.detect_tables_and_shapes(str(pdf_path))
        excluded_info['tables'] = table_regions
        print(f"    Found {len(table_regions.get('text_tables', []))} tables")
        
        # Header/footer detection
        header_footer_regions = header_footer_detector.detect_headers_footers(text_blocks)
        excluded_info['headers_footers'] = header_footer_regions
        print(f"    Found {len(header_footer_regions.get('headers', []))} headers, {len(header_footer_regions.get('footers', []))} footers")
        
        excluded_info['index'] = {'excluded_regions': []}
        
        # Filter candidates
        print("  Phase 2: Filtering heading candidates...")
        candidates = []
        seen_texts = set()
        
        for i, block in enumerate(text_blocks):
            if _is_excluded(block, excluded_info):
                continue
            
            text = block['text'].strip()
            norm_text = text.lower().strip()
            if norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            
            if _is_potential_heading(block):
                # Add features
                if i < len(features_df):
                    block['features'] = features_df.iloc[i].to_dict()
                candidates.append(block)
        
        print(f"    {len(candidates)} heading candidates after filtering")
        
        # Phase 3: Heading labeling
        print("  Phase 3: Labeling headings...")
        if candidates:
            # Initial rule-based labeling
            labeled_candidates = heading_labeler.label_headings(candidates, text_blocks, {})
            
            # Apply ML classification if available
            if ml_classifier:
                print("  Phase 4: Applying ML classification...")
                ml_start_time = time.time()
                
                # Get ML predictions
                predictions, probabilities = ml_classifier.predict_batch(labeled_candidates)
                
                # Filter based on ML predictions
                final_headings = []
                for i, (candidate, pred, prob) in enumerate(zip(labeled_candidates, predictions, probabilities)):
                    if pred == 1:  # ML says it's a heading
                        candidate['ml_confidence'] = float(prob)
                        final_headings.append(candidate)
                
                ml_time = time.time() - ml_start_time
                print(f"    ML classification completed ({ml_time:.2f}s)")
                print(f"    {len(final_headings)} headings after ML filtering (from {len(labeled_candidates)} candidates)")
                
                labeled_candidates = final_headings
            
            # Count labels
            label_counts = {}
            for candidate in labeled_candidates:
                label = candidate.get('label', 'Unknown')
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print(f"    Label distribution: {label_counts}")
            
            # Save results
            output_file = output_dir / f"{pdf_path.stem}_headings.json"
            
            # Create output format
            headings = []
            for candidate in labeled_candidates:
                label = candidate.get('label', 'Not-Heading')
                
                # Skip Not-Heading
                if label == 'Not-Heading':
                    continue
                
                heading_data = {
                    'text': candidate['text'],
                    'level': label,
                    'page': candidate['page_number'],
                    'font_size': candidate['font_size'],
                    'font_name': candidate['font_name'],
                    'is_bold': candidate['is_bold'],
                    'position': {
                        'x0': candidate['x0'],
                        'y0': candidate['y0'],
                        'x1': candidate['x1'],
                        'y1': candidate['y1']
                    }
                }
                
                # Add ML confidence if available
                if 'ml_confidence' in candidate:
                    heading_data['ml_confidence'] = candidate['ml_confidence']
                
                headings.append(heading_data)
            
            # Output data
            # Find the title (first heading with level 'Title', else first heading)
            title = None
            for h in headings:
                if h['level'] == 'Title':
                    title = h['text']
                    break
            if not title and headings:
                title = headings[0]['text']
            # Prepare outline: only level, text, page
            outline = [
                {
                    'level': h['level'],
                    'text': h['text'],
                    'page': h['page']
                }
                for h in headings
            ]
            output_data = {
                'title': title if title else '',
                'outline': outline
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            pdf_time = time.time() - pdf_start_time
            print(f"    Saved {len(headings)} headings to: {output_file}")
            print(f"    PDF processed in {pdf_time:.2f} seconds")
            
        else:
            print("    No heading candidates found!")
            
    except Exception as e:
        print(f"  ERROR processing {pdf_path.name}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Setup directories
    input_dir = Path('input')
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("Initializing PDF processing components...")
    text_extractor = TextExtractor()
    table_detector = TableDetector()
    header_footer_detector = HeaderFooterDetector()
    heading_labeler = HeadingLabeler()
    feature_engineer = FeatureEngineer()
    
    # Initialize ML classifier if available
    ml_classifier = None
    if ML_AVAILABLE:
        try:
            print("Loading ML model for heading classification...")
            ml_classifier = FastHeadingClassifier()
            ml_classifier.load_models()
            print("ML model loaded successfully!")
        except Exception as e:
            print(f"Could not load ML model: {e}")
            ml_classifier = None
    
    # Get all PDF files
    pdf_files = list(input_dir.glob('*.pdf')) + list(input_dir.glob('*.PDF'))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Use multiprocessing to process PDFs in parallel
    with Pool(processes=cpu_count()) as pool:
        process_func = partial(process_pdf,
                               text_extractor=text_extractor,
                               table_detector=table_detector,
                               header_footer_detector=header_footer_detector,
                               heading_labeler=heading_labeler,
                               feature_engineer=feature_engineer,
                               ml_classifier=ml_classifier,
                               output_dir=output_dir)
        pool.map(process_func, pdf_files)

    # Cleanup ML model
    if ml_classifier:
        ml_classifier.cleanup()
    
    total_time = time.time() - total_start_time
    avg_time = total_time / len(pdf_files) if pdf_files else 0
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per PDF: {avg_time:.2f} seconds")
    if total_time < 10 * len(pdf_files):
        print("Performance target met: <10 seconds per PDF")
    else:
        print("Performance target not met: >10 seconds per PDF")


def _is_excluded(block, excluded_info):
    """Check if text block should be excluded based on detected regions."""
    x0, y0, x1, y1 = block['x0'], block['y0'], block['x1'], block['y1']
    page = block['page_number']
    
    # Check if in table regions
    for table in excluded_info.get('tables', {}).get('text_tables', []):
        if (table['page'] == page and 
            x0 >= table['x0'] and x1 <= table['x1'] and
            y0 >= table['y0'] and y1 <= table['y1']):
            return True
    
    # Check if text matches header/footer patterns
    text = block['text'].strip()
    headers = excluded_info.get('headers_footers', {}).get('headers', set())
    footers = excluded_info.get('headers_footers', {}).get('footers', set())
    
    if text in headers or text in footers:
        return True
    
    return False


def _is_potential_heading(block):
    """Improved criteria for potential heading candidates."""
    text = block['text'].strip()
    
    # Skip very short or very long text
    if len(text) < 3 or len(text) > 200:
        return False
    
    # Skip text with too many words
    word_count = len(text.split())
    if word_count > 20:
        return False
    
    # Skip if mostly punctuation or numbers
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < len(text) * 0.4:
        return False
    
    # Skip if contains tabs
    if '\t' in text:
        return False
    
    # Skip obvious fragments
    if text.endswith(',') or text.endswith(';') or '...' in text:
        return False
    
    # Skip citation fragments
    if re.match(r'^\s*\[\s*\d+\s*\]\s*$', text):
        return False
    
    # Good candidates
    font_size = block.get('font_size', 0)
    is_bold = block.get('is_bold', False)
    is_all_caps = text.isupper() and len(text) > 3
    starts_with_capital = text[0].isupper() if text else False
    
    # Heading score
    heading_score = 0
    if font_size > 10: heading_score += 2
    if is_bold: heading_score += 2
    if is_all_caps: heading_score += 1
    if starts_with_capital: heading_score += 1
    if word_count <= 8: heading_score += 1
    
    # Need at least some heading characteristics
    if heading_score >= 2:
        return True
    
    # Always include proper nouns and capitalized phrases
    if starts_with_capital and word_count <= 10:
        return True
    
    return False


if __name__ == "__main__":
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    
    print("PDF Heading Extraction with ML Integration")
    print("Using hybrid ML model for improved accuracy")
    print("Target: <10 seconds per PDF\n")
    
    main()