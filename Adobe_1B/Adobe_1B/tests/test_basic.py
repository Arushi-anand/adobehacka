#!/usr/bin/env python3
"""
Day 2 Test: Basic Persona Matching with Embeddings
Test the persona matching functionality
"""

import sys
import os

# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to Adobe_1B root, then into src
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
# Add to Python path
sys.path.insert(0, src_dir)

print(f"Looking for persona_matcher in: {src_dir}")
print(f"Directory exists: {os.path.exists(src_dir)}")
print(f"persona_matcher.py exists: {os.path.exists(os.path.join(src_dir, 'persona_matcher.py'))}")

from persona_matcher import PersonaMatcher
import json

def test_academic_research_scenario():
    """Test Case 1: Academic Research Scenario"""
    print("="*50)
    print("TEST CASE 1: Academic Research")
    print("="*50)
    
    matcher = PersonaMatcher()
    
    # Test inputs
    persona = "PhD Researcher in Computational Biology"
    job = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    
    # Create persona profile
    profile = matcher.create_persona_profile(persona, job)
    
    print(f"Persona: {persona}")
    print(f"Job: {job}")
    print(f"Key Terms: {profile['key_terms'][:10]}...")  # Show first 10 terms
    
    # Mock sections from research papers
    sections = [
        {
            'document': 'gnn_drug_paper1.pdf',
            'page': 1,
            'title': 'Graph Neural Networks for Drug Discovery',
            'content': 'This paper presents novel methodologies for applying graph neural networks to molecular property prediction and drug discovery pipelines. We introduce new architectures and training techniques.'
        },
        {
            'document': 'gnn_drug_paper1.pdf',
            'page': 3,
            'title': 'Methodology',
            'content': 'Our approach combines message passing neural networks with attention mechanisms. We use a multi-task learning framework to predict multiple molecular properties simultaneously.'
        },
        {
            'document': 'gnn_drug_paper1.pdf',
            'page': 7,
            'title': 'Experimental Results',
            'content': 'We evaluated our method on standard benchmarks including ZINC, ChEMBL, and Tox21 datasets. Performance metrics include ROC-AUC, precision, recall, and F1-score across different molecular property prediction tasks.'
        },
        {
            'document': 'gnn_drug_paper2.pdf',
            'page': 2,
            'title': 'Related Work',
            'content': 'Previous work in computational biology has explored various machine learning approaches for molecular analysis, including traditional fingerprint-based methods and deep learning approaches.'
        },
        {
            'document': 'gnn_drug_paper2.pdf',
            'page': 4,
            'title': 'Conclusion',
            'content': 'In conclusion, our approach shows promising results for drug discovery applications. Future work could explore larger datasets and more complex molecular representations.'
        }
    ]
    
    # Rank sections
    ranked_sections = matcher.rank_sections(sections, profile)
    
    print("\nRANKED SECTIONS:")
    for section in ranked_sections:
        print(f"Rank {section['importance_rank']}: {section['title']}")
        print(f"  Document: {section['document']}, Page: {section['page']}")
        print(f"  Relevance Score: {section['relevance_score']:.3f}")
        print(f"  Content Preview: {section['content'][:100]}...")
        print()

def test_business_analysis_scenario():
    """Test Case 2: Business Analysis Scenario"""
    print("="*50)
    print("TEST CASE 2: Business Analysis")
    print("="*50)
    
    matcher = PersonaMatcher()
    
    # Test inputs
    persona = "Investment Analyst"
    job = "Analyze revenue trends, R&D investments, and market positioning strategies"
    
    # Create persona profile
    profile = matcher.create_persona_profile(persona, job)
    
    print(f"Persona: {persona}")
    print(f"Job: {job}")
    print(f"Key Terms: {profile['key_terms'][:10]}...")
    
    # Mock sections from business reports
    sections = [
        {
            'document': 'tech_company_2024.pdf',
            'page': 5,
            'title': 'Revenue Analysis',
            'content': 'Total revenue for 2024 reached $45.2 billion, representing a 12% year-over-year growth. Cloud services contributed 35% of total revenue, showing strong market positioning in enterprise solutions.'
        },
        {
            'document': 'tech_company_2024.pdf',
            'page': 12,
            'title': 'R&D Investment Strategy',
            'content': 'Research and development expenses increased to $8.1 billion in 2024, up 18% from previous year. Major investments focus on artificial intelligence, cloud infrastructure, and cybersecurity solutions.'
        },
        {
            'document': 'tech_company_2024.pdf',
            'page': 8,
            'title': 'Employee Benefits',
            'content': 'The company enhanced employee benefits package including health insurance, retirement plans, and flexible work arrangements to attract top talent in competitive market.'
        },
        {
            'document': 'competitor_report_2024.pdf',
            'page': 3,
            'title': 'Market Positioning',
            'content': 'Competitive analysis shows strong market positioning in cloud services with 23% market share. Strategic partnerships and acquisitions have strengthened market presence against key competitors.'
        }
    ]
    
    # Rank sections
    ranked_sections = matcher.rank_sections(sections, profile)
    
    print("\nRANKED SECTIONS:")
    for section in ranked_sections:
        print(f"Rank {section['importance_rank']}: {section['title']}")
        print(f"  Relevance Score: {section['relevance_score']:.3f}")
        print()

def main():
    """Run all test cases"""
    print("Starting Day 2 Tests: Basic Persona Matching")
    print("This will test the embedding-based persona matching system\n")
    
    try:
        test_academic_research_scenario()
        test_business_analysis_scenario()
        
        print("="*50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Day 2 Goal Achieved: Basic persona matching working with embeddings")
        print("\nNext Steps for Day 3:")
        print("1. Build section ranking logic")
        print("2. Implement importance scoring algorithms")
        print("3. Add support for sub-section analysis")
        print("="*50)
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        print("Check that all dependencies are installed:")
        print("pip install sentence-transformers scikit-learn numpy")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())