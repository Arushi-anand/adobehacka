#!/usr/bin/env python3
"""
Integrated PDF Processing Pipeline
1. First runs adobehackathon to extract headings from PDFs
2. Then runs Adobe_1B to analyze sections based on persona/job
"""

import os
import sys
import json
import shutil
from pathlib import Path
import subprocess

def run_adobehackathon():
    """Run the adobehackathon heading extraction"""
    print("=" * 60)
    print("STEP 1: Running Adobe Hackathon Heading Extraction")
    print("=" * 60)
    
    # Change to adobehackathon directory
    hackathon_dir = Path("Adobe_1B/adobehackathon")
    
    # Run the main_ml.py script
    try:
        result = subprocess.run(
            [sys.executable, "main_ml.py"],
            cwd=hackathon_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Heading extraction completed successfully!")
            print(result.stdout)
        else:
            print("❌ Heading extraction failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running heading extraction: {e}")
        return False
    
    # Copy output files to main input directory
    hackathon_output = hackathon_dir / "output"
    main_input = Path("input")
    
    json_files = list(hackathon_output.glob("*_headings.json"))
    print(f"\nCopying {len(json_files)} output files to input directory...")
    
    for json_file in json_files:
        shutil.copy2(json_file, main_input)
        print(f"  - Copied {json_file.name}")
    
    return True

def run_adobe_1b():
    """Run the Adobe_1B persona-based analysis"""
    print("\n" + "=" * 60)
    print("STEP 2: Running Adobe 1B Persona-Based Analysis")
    print("=" * 60)
    
    # Change to Adobe_1B src directory
    adobe_1b_dir = Path("Adobe_1B/Adobe_1B/src")
    
    # Run the process_round1b.py script
    try:
        result = subprocess.run(
            [sys.executable, "process_round1b.py"],
            cwd=adobe_1b_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Persona-based analysis completed successfully!")
            print(result.stdout)
        else:
            print("❌ Persona-based analysis failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running persona analysis: {e}")
        return False
    
    return True

def setup_directories():
    """Ensure input and output directories exist"""
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # Also ensure adobehackathon directories exist
    Path("Adobe_1B/adobehackathon/input").mkdir(exist_ok=True)
    Path("Adobe_1B/adobehackathon/output").mkdir(exist_ok=True)

def copy_pdfs_to_hackathon():
    """Copy PDFs from main input to adobehackathon input"""
    main_input = Path("input")
    hackathon_input = Path("Adobe_1B/adobehackathon/input")
    
    pdf_files = list(main_input.glob("*.pdf")) + list(main_input.glob("*.PDF"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in input directory!")
        return False
    
    print(f"Copying {len(pdf_files)} PDF files to adobehackathon input...")
    for pdf_file in pdf_files:
        shutil.copy2(pdf_file, hackathon_input)
        print(f"  - Copied {pdf_file.name}")
    
    return True

def create_test_scenario():
    """Create a test scenario file if it doesn't exist"""
    scenario_file = Path("input/test_scenario.json")
    
    if not scenario_file.exists():
        print("Creating default test scenario...")
        scenario = {
            "persona": "PhD Researcher in Computational Biology",
            "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
        }
        
        with open(scenario_file, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        print(f"  - Created {scenario_file}")

def main():
    """Main integration pipeline"""
    print("🚀 Integrated PDF Processing Pipeline")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Create test scenario if needed
    create_test_scenario()
    
    # Check for PDFs in input directory
    if not copy_pdfs_to_hackathon():
        print("\n❌ Please place PDF files in the 'input' directory and try again.")
        return 1
    
    # Step 1: Run heading extraction
    if not run_adobehackathon():
        print("\n❌ Heading extraction failed. Please check the errors above.")
        return 1
    
    # Step 2: Run persona-based analysis
    if not run_adobe_1b():
        print("\n❌ Persona-based analysis failed. Please check the errors above.")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - Heading extractions: Adobe_1B/adobehackathon/output/")
    print("  - Final analysis: output/challenge1b_output.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())