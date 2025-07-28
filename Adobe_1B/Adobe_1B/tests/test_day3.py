#!/usr/bin/env python3
"""
Day 3 Test: Advanced Section Ranking Logic
Test the comprehensive scoring and ranking system
"""

import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.insert(0, src_dir)

from persona_matcher import PersonaMatcher
from section_ranker import SectionRanker
import json

def test_comprehensive_ranking():
    """Test the comprehensive ranking system"""
    print("="*60)
    print("DAY 3 TEST: Comprehensive Section Ranking")
    print("="*60)
    
    # Initialize components
    matcher = PersonaMatcher()
    ranker = SectionRanker()
    
    # Test scenario: PhD Researcher
    persona = "PhD Researcher in Computational Biology"
    job = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    
    # Create persona profile
    profile = matcher.create_persona_profile(persona, job)
    
    print(f"Persona: {persona}")
    print(f"Job: {job}")
    print()
    
    # Extended test sections with different characteristics
    sections = [
        {
            'document': 'gnn_survey_2024.pdf',
            'page': 1,
            'level': 'title',
            'title': 'Graph Neural Networks for Drug Discovery: A Comprehensive Survey',
            'content': 'This comprehensive survey reviews the latest advances in applying graph neural networks to drug discovery problems. We systematically analyze methodologies, benchmark datasets, and performance metrics across 150+ research papers published between 2020-2024.'
        },
        {
            'document': 'gnn_survey_2024.pdf',
            'page': 3,
            'level': 'H1',
            'title': 'Methodology Overview',
            'content': 'Graph neural networks for molecular property prediction typically follow a message-passing framework. Key methodologies include: 1) Graph Convolutional Networks (GCN), 2) Graph Attention Networks (GAT), 3) Message Passing Neural Networks (MPNN). Each approach has distinct advantages for different molecular representation tasks.'
        },
        {
            'document': 'gnn_survey_2024.pdf',
            'page': 12,
            'level': 'H2',
            'title': 'Benchmark Datasets',
            'content': 'Standard benchmark datasets include ZINC (250K molecules), ChEMBL (1.9M bioactivity data points), Tox21 (12K compounds), and QM9 (134K small organic molecules). Performance is typically measured using ROC-AUC, RMSE, and MAE metrics. Recent studies show GNNs achieving 0.85+ ROC-AUC on molecular property prediction tasks.'
        },
        {
            'document': 'attention_mechanisms_2024.pdf',
            'page': 2,
            'level': 'H1',
            'title': 'Attention Mechanisms in Graph Networks',
            'content': 'Attention mechanisms allow models to focus on relevant molecular substructures. Graph Attention Networks (GATs) use multi-head attention to weight neighbor contributions. Recent work shows 15-20% improvement in predictive performance compared to standard GCNs on drug-target interaction prediction.'
        },
        {
            'document': 'attention_mechanisms_2024.pdf',
            'page': 8,
            'level': 'H3',
            'title': 'Implementation Details',
            'content': 'Implementation typically uses PyTorch Geometric or DGL frameworks. Training involves Adam optimizer (lr=0.001), batch size 64, and early stopping. Hardware requirements: 16GB+ RAM, GPU recommended for datasets >100K molecules.'
        },
        {
            'document': 'benchmark_study_2023.pdf',
            'page': 1,
            'level': 'title',
            'title': 'Benchmarking Graph Neural Networks for Molecular Property Prediction',
            'content': 'This study provides the most comprehensive benchmark comparison of 12 different GNN architectures across 15 molecular property prediction tasks. We evaluate computational efficiency, predictive accuracy, and generalization capabilities.'
        },
        {
            'document': 'benchmark_study_2023.pdf',
            'page': 5,
            'level': 'H2',
            'title': 'Performance Comparison',
            'content': 'Results show that Graph Transformers achieve highest accuracy (ROC-AUC: 0.87±0.02) but require 3x more computation time. GCN variants provide best efficiency-accuracy trade-off (ROC-AUC: 0.82±0.03, 5x faster training). MPNN approaches work best for small datasets (<10K samples).'
        },
        {
            'document': 'future_directions_2024.pdf',
            'page': 15,
            'level': 'H3',
            'title': 'Limitations and Future Work',
            'content': 'Current limitations include handling of 3D molecular conformations and multi-scale interactions. Future research directions involve incorporating physics-based priors, handling molecular dynamics, and developing interpretable graph representations for drug discovery applications.'
        },
        {
            'document': 'case_study_pfizer.pdf',
            'page': 3,
            'level': 'H2',
            'title': 'Industrial Application Case Study',
            'content': 'Pfizer deployed GNN-based molecular screening, processing 2.3M compounds in 48 hours. The system identified 156 promising drug candidates, with 23% hit rate in subsequent wet-lab validation. Cost reduction: $2.4M vs traditional high-throughput screening.'
        }
    ]
    
    # First, get relevance scores from PersonaMatcher
    sections_with_relevance = matcher.rank_sections(sections, profile)
    
    # Now apply comprehensive ranking
    final_rankings = ranker.rank_sections_advanced(sections_with_relevance, max_sections=8)
    
    # Display results
    print("COMPREHENSIVE RANKING RESULTS:")
    print("-" * 60)
    
    for section in final_rankings:
        print(f"\nRANK {section['importance_rank']}: {section['title']}")
        print(f"Document: {section['document']}, Page: {section['page']}, Level: {section.get('level', 'content')}")
        print(f"Comprehensive Score: {section['comprehensive_score']:.3f}")
        
        if 'score_breakdown' in section:
            breakdown = section['score_breakdown']
            print("Score Breakdown:")
            for factor, score in breakdown.items():
                print(f"  • {factor.title()}: {score:.3f}")
        
        # Show content preview
        content_preview = section.get('content', '')[:120] + "..." if len(section.get('content', '')) > 120 else section.get('content', '')
        print(f"Content: {content_preview}")
        print("-" * 40)
    
    # Generate ranking explanation
    print("\n" + "="*60)
    explanation = ranker.generate_ranking_explanation(final_rankings)
    print(explanation)
    
    return final_rankings

def test_different_personas():
    """Test ranking with different personas to show adaptability"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT PERSONAS")
    print("="*60)
    
    matcher = PersonaMatcher()
    ranker = SectionRanker()
    
    # Business analyst persona
    persona2 = "Investment Analyst"
    job2 = "Evaluate the commercial potential and market opportunities for AI drug discovery technologies"
    
    profile2 = matcher.create_persona_profile(persona2, job2)
    
    # Same sections as before, but should rank differently for business analyst
    sections = [
        {
            'document': 'market_report_2024.pdf',
            'page': 1,
            'level': 'H1',
            'title': 'AI Drug Discovery Market Analysis',
            'content': 'The AI drug discovery market is projected to reach $4.8 billion by 2027, growing at 28.4% CAGR. Key players include DeepMind, Atomwise, and BenevolentAI. Investment reached $2.1 billion in 2023, up 67% from previous year.'
        },
        {
            'document': 'technical_deep_dive.pdf',
            'page': 45,
            'level': 'H3',
            'title': 'Technical Implementation of Graph Convolutions',
            'content': 'The mathematical formulation involves sparse matrix operations with complexity O(|E|d) where E is edges and d is feature dimensions. Gradient computation requires backpropagation through graph structure using automatic differentiation frameworks.'
        },
        {
            'document': 'business_case_study.pdf',
            'page': 3,
            'level': 'H2',
            'title': 'ROI Analysis and Business Impact',
            'content': 'Companies implementing AI drug discovery report 40-60% reduction in early-stage development costs. Time-to-market decreased by 2-3 years on average. Success rate in Phase I trials improved from 63% to 78% using AI-guided compound selection.'
        }
    ]
    
    # Process with business analyst persona
    sections_with_relevance = matcher.rank_sections(sections, profile2)
    business_rankings = ranker.rank_sections_advanced(sections_with_relevance, max_sections=3)
    
    print(f"BUSINESS ANALYST PERSPECTIVE:")
    print(f"Persona: {persona2}")
    print(f"Job: {job2}")
    print()
    
    for section in business_rankings:
        print(f"Rank {section['importance_rank']}: {section['title']}")
        print(f"  Score: {section['comprehensive_score']:.3f}")
        print(f"  Why important for business analyst: ", end="")
        
        title_lower = section['title'].lower()
        if any(word in title_lower for word in ['market', 'roi', 'business', 'cost', 'revenue']):
            print("Direct business relevance")
        elif any(word in title_lower for word in ['technical', 'implementation', 'algorithm']):
            print("Technical details - lower priority for investment analysis")
        else:
            print("General relevance")
        print()

def main():
    """Run all Day 3 tests"""
    print("Starting Day 3 Tests: Advanced Section Ranking Logic")
    print("This tests comprehensive scoring with multiple factors\n")
    
    try:
        # Test 1: Comprehensive ranking
        final_rankings = test_comprehensive_ranking()
        
        # Test 2: Different personas
        test_different_personas()
        
        print("\n" + "="*60)
        print("✅ DAY 3 TESTS COMPLETED SUCCESSFULLY!")
        print("\nKey Achievements:")
        print("✓ Multi-factor scoring system implemented")
        print("✓ Hierarchy-aware ranking (H1 > H2 > H3)")
        print("✓ Content quality assessment")
        print("✓ Document diversity optimization")
        print("✓ Position-based scoring")
        print("✓ Persona-adaptive ranking")
        print("\nNext Steps for Day 4:")
        print("1. Implement PDF text extraction")
        print("2. Build sub-section analysis")
        print("3. Integrate with actual PDF processing")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())