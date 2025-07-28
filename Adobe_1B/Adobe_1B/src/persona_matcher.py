import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple

class PersonaMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize with all-MiniLM-L6-v2 sentence transformer model
        all-MiniLM-L6-v2 is a lightweight and efficient model for semantic similarity
        """
        # Load from local models directory for hackathon reliability
        import os
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', model_name)
        if os.path.exists(local_model_path):
            print(f"Loading local model from: {local_model_path}")
            self.model = SentenceTransformer(local_model_path)
        else:
            print(f"Local model not found, downloading: {model_name}")
            self.model = SentenceTransformer(model_name)
        
    def create_persona_profile(self, persona: str, job_to_be_done: str) -> Dict:
        """
        Create a comprehensive persona profile with embeddings
        """
        # Combine persona and job for better context understanding
        combined_context = f"{persona}. Task: {job_to_be_done}"
        
        # Generate embedding for the combined context
        persona_embedding = self.model.encode([combined_context])
        
        # Extract key terms and concepts
        key_terms = self._extract_key_terms(persona + " " + job_to_be_done)
        
        return {
            'persona': persona,
            'job_to_be_done': job_to_be_done,
            'combined_context': combined_context,
            'embedding': persona_embedding[0],
            'key_terms': key_terms
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract important terms and concepts from persona/job description
        """
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                     'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        # Split into words and clean
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
                
        return unique_terms[:20]  # Limit to top 20 terms
    
    def score_section_relevance(self, section_text: str, persona_profile: Dict) -> float:
        """
        Score how relevant a section is to the persona and job
        """
        if not section_text.strip():
            return 0.0
            
        # Generate embedding for the section
        section_embedding = self.model.encode([section_text])
        
        # Calculate cosine similarity with persona profile
        similarity = cosine_similarity(
            [persona_profile['embedding']], 
            section_embedding
        )[0][0]
        
        # Boost score based on key term matches
        key_term_boost = self._calculate_key_term_boost(section_text, persona_profile['key_terms'])
        
        # Combine similarity and key term boost
        final_score = similarity * 0.7 + key_term_boost * 0.3
        
        return float(final_score)
    
    def _calculate_key_term_boost(self, text: str, key_terms: List[str]) -> float:
        """
        Calculate boost based on key term matches
        """
        text_lower = text.lower()
        matches = sum(1 for term in key_terms if term in text_lower)
        
        # Normalize by number of key terms
        if len(key_terms) == 0:
            return 0.0
            
        return min(matches / len(key_terms), 1.0)
    
    def rank_sections(self, sections: List[Dict], persona_profile: Dict) -> List[Dict]:
        """
        Rank sections based on relevance to persona and job
        """
        scored_sections = []
        
        for section in sections:
            # Combine title and content for scoring
            section_text = f"{section.get('title', '')} {section.get('content', '')}"
            
            relevance_score = self.score_section_relevance(section_text, persona_profile)
            
            section_with_score = section.copy()
            section_with_score['relevance_score'] = relevance_score
            section_with_score['importance_rank'] = 0  # Will be set after sorting
            
            scored_sections.append(section_with_score)
        
        # Sort by relevance score (descending)
        scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Assign importance ranks
        for i, section in enumerate(scored_sections):
            section['importance_rank'] = i + 1
            
        return scored_sections

# Test the persona matcher
def test_persona_matcher():
    """
    Test the persona matcher with sample data
    """
    matcher = PersonaMatcher()
    
    # Sample persona and job
    persona = "PhD Researcher in Computational Biology"
    job = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    
    # Create persona profile
    profile = matcher.create_persona_profile(persona, job)
    print("Persona Profile Created:")
    print(f"Key Terms: {profile['key_terms']}")
    print(f"Embedding Shape: {profile['embedding'].shape}")
    
    # Sample sections
    sections = [
        {
            'document': 'paper1.pdf',
            'page': 1,
            'title': 'Graph Neural Networks for Drug Discovery',
            'content': 'This paper presents novel methodologies for applying graph neural networks to molecular property prediction and drug discovery pipelines.'
        },
        {
            'document': 'paper1.pdf',
            'page': 5,
            'title': 'Experimental Setup',
            'content': 'We evaluated our approach on standard datasets including ZINC, ChEMBL, and proprietary pharmaceutical datasets with performance benchmarks.'
        },
        {
            'document': 'paper2.pdf',
            'page': 2,
            'title': 'Related Work',
            'content': 'Previous work in computational biology has focused on traditional machine learning approaches for molecular analysis.'
        }
    ]
    
    # Rank sections
    ranked_sections = matcher.rank_sections(sections, profile)
    
    print("\nRanked Sections:")
    for section in ranked_sections:
        print(f"Rank {section['importance_rank']}: {section['title']} (Score: {section['relevance_score']:.3f})")

if __name__ == "__main__":
    test_persona_matcher()