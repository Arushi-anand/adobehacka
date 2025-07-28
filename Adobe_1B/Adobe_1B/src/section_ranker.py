from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any

class SectionRanker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize section ranker with embedding model"""
        print(f"Loading section ranking model: {model_name}")
        # Load from local models directory for hackathon reliability
        import os
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', model_name)
        if os.path.exists(local_model_path):
            print(f"Loading local model from: {local_model_path}")
            self.model = SentenceTransformer(local_model_path)
        else:
            print(f"Local model not found, downloading: {model_name}")
            # Load from local models directory for hackathon reliability  
        import os
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', model_name)
        if os.path.exists(local_model_path):
            print(f"Loading local model from: {local_model_path}")
            self.model = SentenceTransformer(local_model_path)
        else:
            print(f"Local model not found, downloading: {model_name}")
            self.model = SentenceTransformer(model_name)
        
    def rank_sections(self, sections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """
        Rank sections based on relevance to persona and job-to-be-done
        
        Args:
            sections: List of extracted sections from PDFs
            persona: User persona description
            job_to_be_done: Specific task description
            
        Returns:
            List of sections ranked by relevance
        """
        print(f"Ranking {len(sections)} sections for persona: {persona[:50]}...")
        
        if not sections:
            return []
        
        # Create query from persona + job
        query = f"{persona}: {job_to_be_done}"
        
        # Extract text from sections for encoding
        section_texts = [section['text'] for section in sections]
        
        # Get embeddings
        print("Computing embeddings...")
        query_embedding = self.model.encode([query])
        section_embeddings = self.model.encode(section_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, section_embeddings)[0]
        
        # Add relevance scores to sections and apply additional scoring
        scored_sections = []
        for i, section in enumerate(sections):
            # Base similarity score
            base_score = similarities[i]
            
            # Apply additional scoring factors
            content_score = self._calculate_content_quality_score(section, persona, job_to_be_done)
            length_score = self._calculate_length_score(section)
            position_score = self._calculate_position_score(section)
            
            # Combine scores with weights
            final_score = (
                base_score * 0.7 +          # Primary semantic similarity
                content_score * 0.2 +       # Content quality indicators
                length_score * 0.05 +       # Length appropriateness  
                position_score * 0.05       # Position/page information
            )
            
            # Create new section dict with score
            scored_section = section.copy()
            scored_section['relevance_score'] = final_score
            scored_section['base_similarity'] = base_score
            scored_sections.append(scored_section)
        
        # Sort by relevance score
        ranked_sections = sorted(scored_sections, key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"Ranking complete. Top score: {ranked_sections[0]['relevance_score']:.3f}")
        return ranked_sections
    
    def _calculate_content_quality_score(self, section: Dict, persona: str, job_to_be_done: str) -> float:
        """Calculate content quality score based on various indicators"""
        text = section['text'].lower()
        score = 0.5  # Base score
        
        # Academic/research indicators
        academic_terms = ['research', 'study', 'analysis', 'method', 'results', 'conclusion', 
                         'findings', 'data', 'experiment', 'hypothesis', 'literature', 'review']
        academic_count = sum(1 for term in academic_terms if term in text)
        if academic_count > 0:
            score += min(academic_count * 0.05, 0.2)  # Cap at 0.2
        
        # Technical indicators
        technical_terms = ['algorithm', 'model', 'framework', 'methodology', 'approach', 
                          'technique', 'implementation', 'evaluation', 'performance']
        technical_count = sum(1 for term in technical_terms if term in text)
        if technical_count > 0:
            score += min(technical_count * 0.03, 0.15)  # Cap at 0.15
        
        # Quantitative indicators (numbers, percentages, statistics)
        import re
        quant_patterns = [r'\d+%', r'\d+\.\d+', r'significant', r'correlation', 
                         r'p\s*[<>=]', r'n\s*=', r'\d+\s*participants?']
        quant_count = sum(1 for pattern in quant_patterns if re.search(pattern, text))
        if quant_count > 0:
            score += min(quant_count * 0.04, 0.1)  # Cap at 0.1
        
        # Persona-specific scoring
        if 'researcher' in persona.lower():
            research_terms = ['methodology', 'validation', 'benchmark', 'evaluation', 'metrics']
            research_count = sum(1 for term in research_terms if term in text)
            score += min(research_count * 0.03, 0.1)
        
        if 'student' in persona.lower():
            learning_terms = ['concept', 'principle', 'theory', 'example', 'definition', 'explanation']
            learning_count = sum(1 for term in learning_terms if term in text)
            score += min(learning_count * 0.02, 0.08)
        
        if 'analyst' in persona.lower():
            analysis_terms = ['trend', 'pattern', 'insight', 'comparison', 'metric', 'indicator']
            analysis_count = sum(1 for term in analysis_terms if term in text)
            score += min(analysis_count * 0.03, 0.1)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_length_score(self, section: Dict) -> float:
        """Score based on section length (prefer moderate lengths)"""
        text_length = len(section['text'])
        word_count = section.get('word_count', len(section['text'].split()))
        
        # Optimal range: 100-500 words
        if 100 <= word_count <= 500:
            return 1.0
        elif 50 <= word_count < 100 or 500 < word_count <= 800:
            return 0.8
        elif 30 <= word_count < 50 or 800 < word_count <= 1200:
            return 0.6
        else:
            return 0.3
    
    def _calculate_position_score(self, section: Dict) -> float:
        """Score based on position in document (slight preference for earlier content)"""
        page = section.get('page', 1)
        
        # Slight preference for earlier pages (abstracts, introductions often more relevant)
        if page <= 3:
            return 1.0
        elif page <= 10:
            return 0.9
        elif page <= 20:
            return 0.8
        else:
            return 0.7
    
    def get_ranking_explanation(self, sections: List[Dict], top_n: int = 5) -> Dict[str, Any]:
        """
        Get explanation of why top sections were ranked highly
        
        Args:
            sections: Ranked sections
            top_n: Number of top sections to explain
            
        Returns:
            Explanation dictionary
        """
        explanations = []
        
        for i, section in enumerate(sections[:top_n]):
            explanation = {
                'rank': i + 1,
                'document': section['document'],
                'page': section['page'],
                'overall_score': round(section.get('relevance_score', 0), 3),
                'base_similarity': round(section.get('base_similarity', 0), 3),
                'text_preview': section['text'][:150] + "..." if len(section['text']) > 150 else section['text'],
                'factors': {
                    'length_words': section.get('word_count', len(section['text'].split())),
                    'page_position': section['page']
                }
            }
            explanations.append(explanation)
        
        return {
            'top_sections_explanation': explanations,
            'ranking_methodology': {
                'semantic_similarity_weight': 0.7,
                'content_quality_weight': 0.2,
                'length_score_weight': 0.05,
                'position_score_weight': 0.05
            }
        }
    
    def filter_sections_by_threshold(self, ranked_sections: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        Filter sections by minimum relevance threshold
        
        Args:
            ranked_sections: Sections ranked by relevance
            threshold: Minimum relevance score to include
            
        Returns:
            Filtered sections above threshold
        """
        filtered = [s for s in ranked_sections if s.get('relevance_score', 0) >= threshold]
        print(f"Filtered {len(ranked_sections)} sections to {len(filtered)} above threshold {threshold}")
        return filtered

# Test functionality
if __name__ == "__main__":
    ranker = SectionRanker()
    
    # Test with sample data
    sample_sections = [
        {
            'text': 'This research presents a novel machine learning algorithm for drug discovery. Our methodology involves training deep neural networks on molecular datasets. Results show 15% improvement in prediction accuracy.',
            'document': 'research_paper.pdf',
            'page': 1,
            'word_count': 25
        },
        {
            'text': 'The weather was nice today. Many people went to the park.',
            'document': 'random.pdf', 
            'page': 5,
            'word_count': 10
        }
    ]
    
    persona = "PhD Researcher in Computational Biology"
    job = "Prepare a comprehensive literature review focusing on methodologies"
    
    ranked = ranker.rank_sections(sample_sections, persona, job)
    print(f"\nRanking Results:")
    for i, section in enumerate(ranked):
        print(f"{i+1}. Score: {section['relevance_score']:.3f} - {section['document']} (Page {section['page']})")
        print(f"   Preview: {section['text'][:100]}...")
    
    # Get explanation
    explanation = ranker.get_ranking_explanation(ranked, 2)
    print(f"\nTop section score breakdown:")
    print(f"Overall: {explanation['top_sections_explanation'][0]['overall_score']}")
    print(f"Similarity: {explanation['top_sections_explanation'][0]['base_similarity']}")