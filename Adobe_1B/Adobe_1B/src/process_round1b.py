import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Import your existing modules
from pdf_text_extractor import PDFTextExtractor
from section_ranker import SectionRanker
from subsection_analyzer import SubsectionAnalyzer
from output_builder import OutputBuilder
from outline_loader import OutlineLoader

def load_test_scenario(input_dir):
    """Load test scenario from input directory"""
    try:
        # Look for test_scenario.json or similar configuration file
        scenario_file = os.path.join(input_dir, 'test_scenario.json')
        if os.path.exists(scenario_file):
            with open(scenario_file, 'r') as f:
                scenario = json.load(f)
            return scenario['persona'], scenario['job_to_be_done']
        
        # Fallback: use default scenario for testing
        return "PhD Researcher in Computational Biology", "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    
    except Exception as e:
        print(f"Warning: Could not load test scenario: {e}")
        # Default fallback
        return "Research Analyst", "Extract key insights and methodologies from the provided documents"

def main():
    # Set up directories - unified structure
    # Check if running from project root or from Adobe_1B/Adobe_1B/src
    current_dir = Path.cwd()
    
    # Find the project root (where input/output folders are)
    if current_dir.name == 'src' and current_dir.parent.name == 'Adobe_1B':
        # Running from Adobe_1B/Adobe_1B/src
        project_root = current_dir.parent.parent.parent
    elif current_dir.name == 'Adobe_1B' and (current_dir / 'Adobe_1B').exists():
        # Running from Adobe_1B
        project_root = current_dir.parent
    else:
        # Assume we're at project root
        project_root = current_dir
    
    input_dir = str(project_root / "input")
    output_dir = str(project_root / "output")
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("=== Adobe Hackathon Round 1B: Document Intelligence ===")
        print(f"Processing documents from: {input_dir}")
        
        # Check for JSON files from adobehackathon output
        json_files = list(Path(input_dir).glob('*_headings.json'))
        
        if json_files:
            print(f"\nFound {len(json_files)} heading extraction files from adobehackathon")
            
            # Load persona and job-to-be-done
            persona, job_to_be_done = load_test_scenario(input_dir)
            print(f"Persona: {persona}")
            print(f"Job-to-be-done: {job_to_be_done}")
            
            # Initialize components
            print("\n1. Initializing components...")
            outline_loader = OutlineLoader()
            ranker = SectionRanker()
            subsection_analyzer = SubsectionAnalyzer()
            output_builder = OutputBuilder()
            
            # Load sections from JSON files
            print("\n2. Loading sections from heading extraction files...")
            all_sections = []
            
            for json_file in json_files:
                sections = outline_loader.load_from_json(str(json_file))
                all_sections.extend(sections)
                print(f"   - Loaded {len(sections)} sections from {json_file.name}")
        
        else:
            # Fallback to PDF extraction if no JSON files
            print("\nNo heading extraction files found, extracting from PDFs...")
            
            # Load persona and job-to-be-done
            persona, job_to_be_done = load_test_scenario(input_dir)
            print(f"Persona: {persona}")
            print(f"Job-to-be-done: {job_to_be_done}")
            
            # Initialize components
            print("\n1. Initializing components...")
            extractor = PDFTextExtractor()
            ranker = SectionRanker()
            subsection_analyzer = SubsectionAnalyzer()
            output_builder = OutputBuilder()
            
            # Extract text from all PDFs
            print("\n2. Extracting text from PDFs...")
            all_sections = extractor.extract_from_multiple_pdfs(input_dir)
        
        if not all_sections:
            print("ERROR: No sections extracted from PDFs")
            return 1
        
        # Get extraction statistics
        stats = extractor.get_document_stats(all_sections)
        print(f"   - Processed {stats['total_documents']} documents")
        print(f"   - Extracted {stats['total_sections']} sections")
        print(f"   - Average words per section: {stats['avg_words_per_section']:.1f}")
        
        # Rank sections based on persona and job
        print("\n3. Ranking sections by relevance...")
        ranked_sections = ranker.rank_sections(all_sections, persona, job_to_be_done)
        print(f"   - Ranked {len(ranked_sections)} sections")
        
        # Analyze subsections within top sections
        print("\n4. Analyzing subsections...")
        subsection_results = subsection_analyzer.analyze_subsections(
            ranked_sections[:15],  # Top 15 sections for subsection analysis
            persona, 
            job_to_be_done
        )
        print(f"   - Generated {len(subsection_results)} subsection analyses")
        
        # Build output JSON
        print("\n5. Building output...")
        # Get source files (either PDFs or JSONs)
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        json_files_names = [f for f in os.listdir(input_dir) if f.lower().endswith('_headings.json')]
        source_files = pdf_files if pdf_files else json_files_names
        
        output_data = output_builder.build_output(
            source_files, persona, job_to_be_done,
            ranked_sections, subsection_results
        )
        
        # Save output
        output_file = os.path.join(output_dir, "challenge1b_output.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"   - Output saved to: {output_file}")
        
        # Print summary
        print("\n=== Processing Complete ===")
        print(f"Top 5 most relevant sections:")
        for i, section in enumerate(ranked_sections[:5]):
            print(f"  {i+1}. {section['document']} (Page {section['page']}) - Score: {section.get('relevance_score', 'N/A'):.3f}")
        
        print(f"\nSUCCESS: Generated output with {len(output_data['extracted_sections'])} sections and {len(output_data['subsection_analysis'])} subsections")
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())