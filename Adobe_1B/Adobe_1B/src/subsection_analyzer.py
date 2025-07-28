import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SubsectionAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize subsection analyzer with embedding model"""
        print(f"Loading subsection analysis model: {model_name}")
        # Load from local models directory for hackathon reliability
        import os  
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', model_name)
        if os.path.exists(local_model_path):
            print(f"Loading local model from: {local_model_path}")
            self.model = SentenceTransformer(local_model_path)
        else:
            print(f"Local model not found, downloading: {model_name}")
            self.model = SentenceTransformer(model_name)
        
    def analyze_subsections(self, ranked_sections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """
        Analyze and rank subsections within the top ranked sections
        
        Args:
            ranked_sections: Top ranked sections from section ranker
            persona: User persona description
            job_to_be_done: Specific task description
            
        Returns:
            List of analyzed and ranked subsections
        """
        print(f"Analyzing subsections from {len(ranked_sections)} top sections...")
        
        all_subsections = []
        
        for section in ranked_sections:
            subsections = self._extract_subsections(section)
            all_subsections.extend(subsections)

        if not all_subsections:
            return []

        # Batch encode all subsection texts
        subsection_texts = [sub['text'] for sub in all_subsections]
        query = f"{persona}: {job_to_be_done}"
        
        query_embedding = self.model.encode([query])
        text_embeddings = self.model.encode(subsection_texts)
        
        similarities = cosine_similarity(query_embedding, text_embeddings)[0]

        for i, subsection in enumerate(all_subsections):
            similarity = similarities[i]
            length_score = self._calculate_length_score(subsection['text'])
            quality_score = self._calculate_content_quality_score(subsection['text'], persona, job_to_be_done)
            structure_score = self._calculate_structure_score(subsection['text'])
            
            final_score = (
                similarity * 0.6 +
                quality_score * 0.25 +
                length_score * 0.1 +
                structure_score * 0.05
            )
            subsection['relevance_score'] = min(final_score, 1.0)
        
        # Sort by relevance score
        sorted_subsections = sorted(all_subsections, key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"Generated {len(sorted_subsections)} subsections, top score: {sorted_subsections[0]['relevance_score']:.3f}")
        
        return sorted_subsections
    
    def _extract_subsections(self, section: Dict) -> List[Dict]:
        """
        Extract meaningful subsections from a larger section
        
        Args:
            section: Section dictionary with text and metadata
            
        Returns:
            List of subsection dictionaries
        """
        text = section['text']
        subsections = []
        
        # Method 1: Split by paragraph-like structures
        paragraphs = self._split_into_paragraphs(text)
        
        # Method 2: If paragraphs are too long, split by sentences
        refined_subsections = []
        for para in paragraphs:
            if len(para) > 400:  # If paragraph is long, split further
                sentences = self._split_into_sentences(para)
                grouped = self._group_sentences(sentences, max_length=350)
                refined_subsections.extend(grouped)
            else:
                refined_subsections.append(para)
        
        # Create subsection objects
        for i, subsection_text in enumerate(refined_subsections):
            if len(subsection_text.strip()) > 80:  # Minimum length threshold
                subsection = {
                    'document': section['document'],
                    'page_number': section['page'],
                    'section_title': self._generate_section_title(subsection_text),
                    'text': subsection_text.strip(),
                    'refined_text': self._refine_text(subsection_text),
                    'subsection_id': f"{section.get('section_id', 'unknown')}_sub{i+1}",
                    'parent_section_score': section.get('relevance_score', 0)
                }
                subsections.append(subsection)
        
        return subsections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraph-like chunks"""
        # Split by double newlines or other paragraph indicators
        paragraphs = re.split(r'\n\s*\n|\. {2,}', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+(?=\s+[A-Z]|\s*$)', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_sentences(self, sentences: List[str], max_length: int = 350) -> List[str]:
        """Group sentences into subsections of appropriate length"""
        groups = []
        current_group = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_length and current_group:
                # Finalize current group
                groups.append(' '.join(current_group))
                current_group = [sentence]
                current_length = sentence_length
            else:
                current_group.append(sentence)
                current_length += sentence_length
        
        # Add remaining sentences
        if current_group:
            groups.append(' '.join(current_group))
        
        return groups
    
    def _generate_section_title(self, text: str) -> str:
        """Generate a descriptive title for the subsection"""
        # Take first few words or identify key phrases
        words = text.split()[:8]
        title = ' '.join(words)
        
        # Clean up and add ellipsis if needed
        if len(text.split()) > 8:
            title += "..."
        
        return title
    
    def _refine_text(self, text: str, max_length: int = 500) -> str:
        """Refine text for output (truncate if too long)"""
        if len(text) <= max_length:
            return text
        
        # Try to cut at sentence boundary
        truncated = text[:max_length]
        last_sentence_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        
        if last_sentence_end > max_length * 0.7:  # If we can cut at a reasonable sentence boundary
            return truncated[:last_sentence_end + 1]
        else:
            return truncated + "..."
    
    
    def _calculate_length_score(self, text: str) -> float:
        """Score based on text length (prefer moderate lengths)"""
        length = len(text)
        
        if 150 <= length <= 400:  # Optimal range
            return 1.0
        elif 100 <= length < 150 or 400 < length <= 600:  # Good range
            return 0.8
        elif 80 <= length < 100 or 600 < length <= 800:  # Acceptable range
            return 0.6
        else:  # Too short or too long
            return 0.3
    
    def _calculate_content_quality_score(self, text: str, persona: str, job_to_be_done: str) -> float:
        """Score based on content quality indicators"""
        score = 0.5  # Base score
        
        # Check for technical/academic language
        technical_indicators = ['method', 'analysis', 'result', 'study', 'research', 'data', 'model', 'algorithm', 'approach', 'findings', 'conclusion']
        if any(indicator in text.lower() for indicator in technical_indicators):
            score += 0.2
        
        # Check for quantitative information
        if re.search(r'\d+%|\d+\.\d+|significant|correlation|p\s*[<>=]|n\s*=', text.lower()):
            score += 0.15
        
        # Check for methodology keywords if relevant to persona
        if 'researcher' in persona.lower() or 'research' in job_to_be_done.lower():
            methodology_terms = ['methodology', 'experiment', 'hypothesis', 'validation', 'evaluation', 'benchmark']
            if any(term in text.lower() for term in methodology_terms):
                score += 0.15
        
        return min(score, 1.0)
    
    def _calculate_structure_score(self, text: str) -> float:
        """Score based on text structure quality"""
        score = 0.5  # Base score
        
        # Check for complete sentences
        sentence_count = len(re.findall(r'[.!?]+', text))
        word_count = len(text.split())
        
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if 10 <= avg_sentence_length <= 25:  # Good sentence length
                score += 0.3
            elif 8 <= avg_sentence_length <= 30:  # Acceptable
                score += 0.2
        
        # Check for proper capitalization
        if re.search(r'^[A-Z]', text.strip()):
            score += 0.1
        
        # Penalize excessive repetition
        words = text.lower().split()
        if len(set(words)) / len(words) < 0.5:  # Too much repetition
            score -= 0.2
        
        return max(min(score, 1.0), 0.0)

# Test functionality
if __name__ == "__main__":
    analyzer = SubsectionAnalyzer()
    
    # Test with sample data
    sample_sections = [
        {
            'text': 'This is a test section with multiple sentences. It contains information about machine learning algorithms. The methodology involves training neural networks on large datasets. Results show significant improvements in accuracy.',
            'document': 'test.pdf',
            'page': 1,
            'section_id': 'test_section_1',
            'relevance_score': 0.8
        }
    ]
    
    persona = "PhD Researcher in Machine Learning"
    job = "Analyze methodologies and performance metrics"
    
    results = analyzer.analyze_subsections(sample_sections, persona, job)
    print(f"Generated {len(results)} subsections")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. Score: {result['relevance_score']:.3f} - {result['section_title']}")