"""
Fast Inference Pipeline for Heading Classification

This module provides a fast inference engine that:
1. Loads the trained XGBoost model (no LLM needed for inference)
2. Processes PDF heading candidates
3. Classifies headings using the hybrid features
4. Integrates seamlessly with main.py

Target: <10 seconds per PDF
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import gc


class FastHeadingClassifier:
    def __init__(self, model_dir='model/trained_models'):
        self.model_dir = Path(model_dir)
        self.xgb_model = None
        self.scaler = None
        self.feature_names = None
        self.sentence_model = None
        self._embeddings_cache = {}
        
    def load_models(self):
        """Load trained models for inference."""
        print("⚡ Loading models for fast inference...")
        
        # Load XGBoost model
        xgb_path = self.model_dir / "heading_classifier.joblib"
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)
            print("✅ XGBoost model loaded")
        else:
            raise FileNotFoundError(f"Model not found at {xgb_path}")
        
        # Load scaler
        scaler_path = self.model_dir / "feature_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded")
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        # Load feature names
        feature_names_path = self.model_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"✅ Feature names loaded ({len(self.feature_names)} features)")
        else:
            raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
        
        # Load sentence transformer for local offline use
        try:
            # Import from Adobe_1B persona matcher for local model
            import sys
            adobe_1b_path = Path(__file__).parent.parent.parent / "Adobe_1B" / "src"
            sys.path.insert(0, str(adobe_1b_path))
            from persona_matcher import PersonaMatcher
            
            # Initialize with all-MiniLM-L6-v2
            persona_matcher = PersonaMatcher(model_name='all-MiniLM-L6-v2')
            self.sentence_model = persona_matcher.model
            print("✅ Sentence transformer loaded (all-MiniLM-L6-v2) for local inference")
        except Exception as e:
            print(f"⚠️ Could not load sentence transformer: {e}")
            print("  Proceeding without embeddings")
    
    def extract_structured_features(self, candidate):
        """Extract structural features from a heading candidate."""
        features = {}
        
        # Get features from candidate
        feature_dict = candidate.get('features', {})
        
        # Basic text features
        text = candidate.get('text', '')
        features['text_length'] = len(text)
        features['word_count'] = len(text.split()) if text else 0
        features['char_count'] = len(text)
        
        # Position and layout features
        features['page_number'] = candidate.get('page', 0)
        features['font_size'] = candidate.get('font_size', feature_dict.get('font_size', 0))
        features['is_bold'] = int(candidate.get('is_bold', feature_dict.get('is_bold', False)))
        features['is_italic'] = int(feature_dict.get('is_italic', False))
        
        # Advanced structural features
        structural_features = [
            'font_size_rank', 'font_size_percentile', 'is_largest_font', 'is_above_median_font',
            'is_all_uppercase', 'is_all_lowercase', 'starts_with_capital', 'is_title_case',
            'uppercase_ratio', 'lowercase_ratio', 'space_before', 'space_after',
            'has_significant_space_before', 'has_significant_space_after',
            'contains_academic_term', 'academic_term_count', 'has_arabic_numbering',
            'has_roman_numbering', 'numbering_level', 'x_position_ratio', 'y_position_ratio',
            'is_first_page', 'is_early_page', 'repetition_count', 'semantic_word_count'
        ]
        
        for feature_name in structural_features:
            if feature_name in feature_dict:
                value = feature_dict[feature_name]
                if isinstance(value, bool):
                    features[feature_name] = int(value)
                elif isinstance(value, (int, float)):
                    features[feature_name] = float(value)
                else:
                    features[feature_name] = 0
            else:
                features[feature_name] = 0
        
        # Level encoding (for existing labels)
        level = candidate.get('level', 'Unknown')
        level_mapping = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4, 'Title': 0, 'Unknown': -1}
        features['level_numeric'] = level_mapping.get(level, -1)
        features['is_h1'] = int(level == 'H1')
        features['is_h2'] = int(level == 'H2')
        
        features['similarity_score'] = 0.0  # Default for inference
        
        return features
    
    def predict_batch(self, candidates):
        """Predict headings for a batch of candidates."""
        if not self.xgb_model:
            self.load_models()
        
        start_time = time.time()
        
        # Extract texts for embeddings
        texts = [candidate.get('text', '') for candidate in candidates]
        
        # Generate embeddings if model available
        embeddings = None
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode(texts, show_progress_bar=False)
            except Exception as e:
                print(f"Warning: Could not generate embeddings: {e}")
        
        # Extract structural features
        structural_features = []
        for candidate in candidates:
            features = self.extract_structured_features(candidate)
            structural_features.append(features)
        
        # Convert to DataFrame
        structural_df = pd.DataFrame(structural_features)
        structural_df = structural_df.fillna(0)
        
        # Create embedding columns if embeddings available
        if embeddings is not None:
            # Add embedding features
            embedding_df = pd.DataFrame(embeddings, columns=[f'embed_{i}' for i in range(embeddings.shape[1])])
            combined_df = pd.concat([structural_df, embedding_df], axis=1)
        else:
            combined_df = structural_df
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in combined_df.columns:
                combined_df[feature] = 0
        
        # Select features in correct order
        X = combined_df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.xgb_model.predict(X_scaled)
        probabilities = self.xgb_model.predict_proba(X_scaled)[:, 1]
        
        inference_time = time.time() - start_time
        print(f"✅ Inference completed in {inference_time:.2f} seconds")
        
        return predictions, probabilities
    
    def classify_headings(self, candidates, threshold=0.3):
        """Classify heading candidates and filter based on predictions."""
        predictions, probabilities = self.predict_batch(candidates)
        
        # Filter candidates based on predictions
        classified_headings = []
        
        for i, (candidate, pred, prob) in enumerate(zip(candidates, predictions, probabilities)):
            if pred == 1 and prob >= threshold:
                # This is a valid heading
                candidate['ml_prediction'] = 'heading'
                candidate['ml_confidence'] = float(prob)
                classified_headings.append(candidate)
            else:
                # Optionally keep track of rejected candidates
                candidate['ml_prediction'] = 'not_heading'
                candidate['ml_confidence'] = float(1 - prob)
        
        return classified_headings
    
    def cleanup(self):
        """Clean up models to free memory."""
        if self.sentence_model:
            del self.sentence_model
        gc.collect()
        print("🧹 Models cleaned up")


def integrate_with_main(pdf_path, use_ml=True):
    """
    Integration function to be called from main.py
    
    Args:
        pdf_path: Path to the PDF file
        use_ml: Whether to use ML classification (default: True)
    
    Returns:
        List of heading candidates (filtered by ML if enabled)
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.text_extractor import TextExtractor
    from src.table_detector import TableDetector
    from src.header_footer_detector import HeaderFooterDetector
    from src.heading_labeler import HeadingLabeler
    from src.feature_engineer import FeatureEngineer
    
    # Extract and process PDF
    text_extractor = TextExtractor()
    table_detector = TableDetector()
    header_footer_detector = HeaderFooterDetector()
    heading_labeler = HeadingLabeler()
    feature_engineer = FeatureEngineer()
    
    # Extract text blocks
    text_blocks = text_extractor.extract_text_blocks(str(pdf_path))
    
    # Extract features
    features_df = feature_engineer.extract_features(text_blocks)
    
    # Detect exclusions
    excluded_info = {}
    table_regions = table_detector.detect_tables_and_shapes(str(pdf_path))
    excluded_info['tables'] = table_regions
    header_footer_regions = header_footer_detector.detect_headers_footers(text_blocks)
    excluded_info['headers_footers'] = header_footer_regions
    excluded_info['index'] = {'excluded_regions': []}
    
    # Filter candidates (same logic as main.py)
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
    
    # Label headings
    if candidates:
        labeled_candidates = heading_labeler.label_headings(candidates, text_blocks, {})
        
        # Apply ML classification if enabled
        if use_ml:
            try:
                classifier = FastHeadingClassifier()
                filtered_candidates = classifier.classify_headings(labeled_candidates)
                classifier.cleanup()
                return filtered_candidates
            except Exception as e:
                print(f"ML classification failed: {e}")
                print("Falling back to rule-based classification")
                return labeled_candidates
        else:
            return labeled_candidates
    
    return []


def _is_excluded(block, excluded_info):
    """Check if text block should be excluded."""
    x0, y0, x1, y1 = block['x0'], block['y0'], block['x1'], block['y1']
    page = block['page_number']
    
    for table in excluded_info.get('tables', {}).get('text_tables', []):
        if (table['page'] == page and 
            x0 >= table['x0'] and x1 <= table['x1'] and
            y0 >= table['y0'] and y1 <= table['y1']):
            return True
    
    text = block['text'].strip()
    headers = excluded_info.get('headers_footers', {}).get('headers', set())
    footers = excluded_info.get('headers_footers', {}).get('footers', set())
    
    if text in headers or text in footers:
        return True
    
    return False


def _is_potential_heading(block):
    """Check if block is a potential heading."""
    text = block['text'].strip()
    
    if len(text) < 3 or len(text) > 200:
        return False
    
    word_count = len(text.split())
    if word_count > 20:
        return False
    
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < len(text) * 0.4:
        return False
    
    if '\t' in text:
        return False
    
    # Good candidates
    font_size = block.get('font_size', 0)
    is_bold = block.get('is_bold', False)
    is_all_caps = text.isupper() and len(text) > 3
    starts_with_capital = text[0].isupper() if text else False
    
    heading_score = 0
    if font_size > 10: heading_score += 2
    if is_bold: heading_score += 2
    if is_all_caps: heading_score += 1
    if starts_with_capital: heading_score += 1
    if word_count <= 8: heading_score += 1
    
    if heading_score >= 2:
        return True
    
    if starts_with_capital and word_count <= 10:
        return True
    
    return False


if __name__ == "__main__":
    # Test the inference pipeline
    print("🚀 Testing Fast Inference Pipeline")
    
    classifier = FastHeadingClassifier()
    
    try:
        classifier.load_models()
        print("✅ Models loaded successfully!")
        
        # Test with dummy data
        test_candidates = [
            {
                'text': 'Introduction',
                'font_size': 14,
                'is_bold': True,
                'page': 1,
                'features': {}
            },
            {
                'text': 'This is a regular paragraph',
                'font_size': 10,
                'is_bold': False,
                'page': 1,
                'features': {}
            }
        ]
        
        predictions, probs = classifier.predict_batch(test_candidates)
        print(f"\nTest predictions: {predictions}")
        print(f"Probabilities: {probs}")
        
        classifier.cleanup()
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run train_pipeline.py first to generate the models.")