import json
from datetime import datetime
from typing import List, Dict, Any


class OutputBuilder:
    def __init__(self, max_sections: int = 12, max_subsections: int = 22):
        """
        Initialize output builder with configurable limits
        
        Args:
            max_sections: Maximum sections to include in output
            max_subsections: Maximum subsections to include in output
        """
        self.max_sections = max_sections
        self.max_subsections = max_subsections
    
    def build_output(self, pdf_files: List[str], persona: str, job_to_be_done: str, 
                    ranked_sections: List[Dict], subsection_results: List[Dict]) -> Dict[str, Any]:
        """
        Build the final JSON output according to challenge requirements
        
        Args:
            pdf_files: List of input PDF filenames
            persona: User persona description
            job_to_be_done: Specific task description
            ranked_sections: Top ranked sections
            subsection_results: Analyzed subsections
            
        Returns:
            Formatted output dictionary
        """
        
        # Input validation
        if not pdf_files:
            raise ValueError("pdf_files cannot be empty")
        if not ranked_sections:
            raise ValueError("ranked_sections cannot be empty")
        if not subsection_results:
            raise ValueError("subsection_results cannot be empty")
        
        # Build metadata
        metadata = {
            "input_documents": [self._standardize_document_name(pdf) for pdf in pdf_files],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat(),
            "total_sections_processed": len(ranked_sections),
            "total_subsections_processed": len(subsection_results)
        }
        
        # Build extracted sections (top sections based on configured limit)
        extracted_sections = []
        for i, section in enumerate(ranked_sections[:self.max_sections]):
            section_entry = {
                "document": self._standardize_document_name(section.get('document', section.get('document_title', 'unknown'))),
                "page_number": section.get('page', section.get('page_number', 1)),
                "section_title": self._generate_section_title(section),
                "importance_rank": section.get('importance_rank', i + 1)
            }
            
            # Add optional fields if available
            if 'relevance_score' in section or 'score' in section:
                score = section.get('relevance_score', section.get('score', 0))
                section_entry["relevance_score"] = round(float(score), 4)
            
            if 'word_count' in section:
                section_entry["word_count"] = section['word_count']
            
            if 'level' in section:
                section_entry["section_level"] = section['level']
                
            extracted_sections.append(section_entry)
        
        # Build subsection analysis (top subsections based on configured limit)
        subsection_analysis = []
        for i, subsection in enumerate(subsection_results[:self.max_subsections]):
            subsection_entry = {
                "document": self._standardize_document_name(subsection.get('document', 'unknown')),
                "refined_text": subsection.get('refined_text', '')[:500],  # Limit text length
                "page_number": subsection.get('page_number', subsection.get('page', 1)),
                "relevance_rank": subsection.get('relevance_rank', i + 1)
            }
            
            # Add optional fields
            if 'relevance_score' in subsection or 'score' in subsection:
                score = subsection.get('relevance_score', subsection.get('score', 0))
                subsection_entry["relevance_score"] = round(float(score), 4)
                
            if 'section_title' in subsection:
                subsection_entry["section_title"] = subsection['section_title'][:100]  # Limit title length
            
            if 'importance_rank' in subsection:
                subsection_entry["importance_rank"] = subsection['importance_rank']
                
            subsection_analysis.append(subsection_entry)
        
        # Combine into final output
        output = {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        # Add summary statistics
        output["summary"] = self._generate_summary(pdf_files, ranked_sections, subsection_results)
        
        return output
    
    def _standardize_document_name(self, doc_name: str) -> str:
        """
        Standardize document names for consistency
        
        Args:
            doc_name: Original document name
            
        Returns:
            Standardized document name
        """
        if not doc_name:
            return "Unknown Document"
            
        # Remove file extensions and normalize
        if doc_name.endswith('.pdf'):
            doc_name = doc_name[:-4]
        
        # Replace underscores with spaces and title case
        return doc_name.replace('_', ' ').title()
    
    def _generate_section_title(self, section: Dict) -> str:
        """
        Generate a descriptive title for a section
        
        Args:
            section: Section dictionary
            
        Returns:
            Generated section title
        """
        # Try different possible title fields
        title_fields = ['section_title', 'text', 'title', 'heading']
        
        for field in title_fields:
            if field in section and section[field]:
                title = str(section[field]).strip()
                if title:
                    return title[:100]  # Limit title length
        
        # Generate from text content if available
        text = section.get('content', section.get('text_content', ''))
        if text:
            # Take first sentence or first 8 words
            first_sentence = str(text).split('.')[0].strip()
            if len(first_sentence) <= 80:
                return first_sentence
            else:
                words = str(text).split()[:8]
                return ' '.join(words) + "..."
        
        # Fallback
        doc_name = section.get('document', section.get('document_title', 'unknown'))
        page_num = section.get('page', section.get('page_number', '?'))
        return f"Section from {self._standardize_document_name(doc_name)} page {page_num}"
    
    def _generate_summary(self, pdf_files: List[str], ranked_sections: List[Dict], 
                         subsection_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics
        
        Args:
            pdf_files: List of input PDF files
            ranked_sections: Ranked sections list
            subsection_results: Subsection results list
            
        Returns:
            Summary statistics dictionary
        """
        
        # Document-wise breakdown
        doc_breakdown = {}
        top_sections = ranked_sections[:self.max_sections]
        
        for section in top_sections:
            doc = self._standardize_document_name(
                section.get('document', section.get('document_title', 'unknown'))
            )
            if doc not in doc_breakdown:
                doc_breakdown[doc] = {
                    'sections_selected': 0,
                    'pages_covered': set(),
                    'avg_relevance': []
                }
            doc_breakdown[doc]['sections_selected'] += 1
            doc_breakdown[doc]['pages_covered'].add(
                section.get('page', section.get('page_number', 1))
            )
            
            # Get relevance score from either field
            score = section.get('relevance_score', section.get('score', 0))
            if score:
                doc_breakdown[doc]['avg_relevance'].append(float(score))
        
        # Convert sets to counts and calculate averages
        for doc in doc_breakdown:
            doc_breakdown[doc]['pages_covered'] = len(doc_breakdown[doc]['pages_covered'])
            if doc_breakdown[doc]['avg_relevance']:
                doc_breakdown[doc]['avg_relevance'] = round(
                    sum(doc_breakdown[doc]['avg_relevance']) / len(doc_breakdown[doc]['avg_relevance']), 3
                )
            else:
                doc_breakdown[doc]['avg_relevance'] = 0.0
        
        # Overall statistics
        total_sections_analyzed = len(ranked_sections)
        total_subsections_generated = len(subsection_results)
        
        # Relevance distribution
        relevance_stats = None
        if ranked_sections:
            # Try to get scores from either field
            relevance_scores = []
            for s in ranked_sections[:self.max_sections]:
                score = s.get('relevance_score', s.get('score', 0))
                if score:
                    relevance_scores.append(float(score))
            
            if relevance_scores:
                relevance_stats = {
                    'highest': round(max(relevance_scores), 4),
                    'lowest': round(min(relevance_scores), 4),
                    'average': round(sum(relevance_scores) / len(relevance_scores), 4)
                }
        
        summary = {
            'total_documents_processed': len(pdf_files),
            'total_sections_analyzed': total_sections_analyzed,
            'total_subsections_generated': total_subsections_generated,
            'sections_selected_for_output': len(top_sections),
            'subsections_selected_for_output': min(len(subsection_results), self.max_subsections),
            'document_breakdown': doc_breakdown
        }
        
        if relevance_stats:
            summary['relevance_statistics'] = relevance_stats
        
        return summary
    
    def save_output(self, output_data: Dict[str, Any], output_path: str) -> bool:
        """
        Save output to JSON file
        
        Args:
            output_data: Output dictionary to save
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Output saved successfully to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving output: {e}")
            return False
    
    def validate_output_format(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that output matches expected format
        
        Args:
            output_data: Output dictionary to validate
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required top-level keys
        required_keys = ['metadata', 'extracted_sections', 'subsection_analysis']
        for key in required_keys:
            if key not in output_data:
                validation['valid'] = False
                validation['errors'].append(f"Missing required key: {key}")
        
        # Check metadata structure
        if 'metadata' in output_data:
            metadata = output_data['metadata']
            required_metadata = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
            for key in required_metadata:
                if key not in metadata:
                    validation['warnings'].append(f"Missing metadata key: {key}")
        
        # Check extracted sections structure
        if 'extracted_sections' in output_data:
            for i, section in enumerate(output_data['extracted_sections']):
                required_section_keys = ['document', 'page_number', 'section_title', 'importance_rank']
                for key in required_section_keys:
                    if key not in section:
                        validation['warnings'].append(f"Section {i+1} missing key: {key}")
        
        # Check subsection analysis structure
        if 'subsection_analysis' in output_data:
            for i, subsection in enumerate(output_data['subsection_analysis']):
                required_subsection_keys = ['document', 'refined_text', 'page_number', 'relevance_rank']
                for key in required_subsection_keys:
                    if key not in subsection:
                        validation['warnings'].append(f"Subsection {i+1} missing key: {key}")
        
        # Check for reasonable data sizes (performance optimization)
        if len(output_data.get('extracted_sections', [])) > 50:
            validation['warnings'].append("Large number of extracted sections may impact performance")
        
        if len(output_data.get('subsection_analysis', [])) > 100:
            validation['warnings'].append("Large number of subsections may impact performance")
        
        return validation
    
    def create_debug_output(self, pdf_files: List[str], persona: str, job_to_be_done: str,
                          ranked_sections: List[Dict], subsection_results: List[Dict],
                          output_path: str = None) -> Dict[str, Any]:
        """
        Create detailed debug output for development
        
        Args:
            pdf_files: List of PDF files
            persona: Persona description
            job_to_be_done: Job description
            ranked_sections: Ranked sections
            subsection_results: Subsection results
            output_path: Optional path to save debug output
            
        Returns:
            Debug output dictionary
        """
        debug_output = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'input_files': pdf_files,
                'persona': persona,
                'job_to_be_done': job_to_be_done,
                'config': {
                    'max_sections': self.max_sections,
                    'max_subsections': self.max_subsections
                }
            },
            'section_details': [],
            'subsection_details': [],
            'statistics': {
                'total_sections': len(ranked_sections),
                'total_subsections': len(subsection_results),
                'sections_in_output': min(len(ranked_sections), self.max_sections),
                'subsections_in_output': min(len(subsection_results), self.max_subsections)
            }
        }
        
        # Detailed section information
        for i, section in enumerate(ranked_sections[:15]):  # Top 15 for debug
            section_detail = {
                'rank': section.get('importance_rank', i + 1),
                'document': self._standardize_document_name(
                    section.get('document', section.get('document_title', 'unknown'))
                ),
                'page': section.get('page', section.get('page_number', 1)),
                'relevance_score': section.get('relevance_score', section.get('score', 0)),
                'section_title': self._generate_section_title(section)[:100],
                'word_count': section.get('word_count', 0),
                'text_preview': str(section.get('text', section.get('content', '')))[:200] + "..." 
                               if len(str(section.get('text', section.get('content', '')))) > 200 
                               else str(section.get('text', section.get('content', '')))
            }
            debug_output['section_details'].append(section_detail)
        
        # Detailed subsection information
        for i, subsection in enumerate(subsection_results[:20]):  # Top 20 for debug
            subsection_detail = {
                'rank': subsection.get('relevance_rank', i + 1),
                'document': self._standardize_document_name(subsection.get('document', 'unknown')),
                'page': subsection.get('page_number', subsection.get('page', 1)),
                'relevance_score': subsection.get('relevance_score', subsection.get('score', 0)),
                'section_title': subsection.get('section_title', '')[:100],
                'text_length': len(str(subsection.get('refined_text', ''))),
                'text_preview': str(subsection.get('refined_text', ''))[:150] + "..." 
                               if len(str(subsection.get('refined_text', ''))) > 150 
                               else str(subsection.get('refined_text', ''))
            }
            debug_output['subsection_details'].append(subsection_detail)
        
        # Save debug output if path provided
        if output_path:
            debug_path = output_path.replace('.json', '_debug.json')
            try:
                import os
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                
                with open(debug_path, 'w', encoding='utf-8') as f:
                    json.dump(debug_output, f, indent=2, ensure_ascii=False)
                print(f"🔍 Debug output saved to: {debug_path}")
            except Exception as e:
                print(f"❌ Error saving debug output: {e}")
        
        return debug_output
    
    def create_sample_output(self) -> Dict[str, Any]:
        """
        Create sample output for testing and validation
        
        Returns:
            Sample output dictionary
        """
        sample_sections = [
            {
                'document_title': 'Breakfast Ideas',
                'text': 'Greek Yogurt Parfait',
                'page': 3,
                'score': 0.479,
                'importance_rank': 1,
                'word_count': 150
            },
            {
                'document_title': 'Lunch Ideas', 
                'text': 'Chicken Caesar Wrap',
                'page': 1,
                'score': 0.465,
                'importance_rank': 2,
                'word_count': 120
            }
        ]
        
        sample_subsections = [
            {
                'document': 'Breakfast Ideas',
                'page_number': 3,
                'refined_text': 'Greek yogurt parfait with mixed berries and granola provides high protein content...',
                'relevance_score': 0.485,
                'section_title': 'Greek Yogurt Parfait',
                'relevance_rank': 1
            },
            {
                'document': 'Lunch Ideas',
                'page_number': 1, 
                'refined_text': 'Chicken Caesar wrap offers balanced nutrition with lean protein and fresh vegetables...',
                'relevance_score': 0.470,
                'section_title': 'Chicken Caesar Wrap',
                'relevance_rank': 2
            }
        ]
        
        return self.build_output(
            pdf_files=['breakfast_ideas.pdf', 'lunch_ideas.pdf'],
            persona='Personal Chef and Meal Planning Specialist',
            job_to_be_done='Create a comprehensive weekly meal plan with high-protein, balanced nutrition',
            ranked_sections=sample_sections,
            subsection_results=sample_subsections
        )


# Test functionality
if __name__ == "__main__":
    print("Testing OutputBuilder...")
    
    builder = OutputBuilder()
    
    # Test with sample data
    sample_output = builder.create_sample_output()
    
    # Validate output
    validation = builder.validate_output_format(sample_output)
    print(f"Output validation: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
    
    if validation['errors']:
        print("Errors:", validation['errors'])
    if validation['warnings']:
        print("Warnings:", validation['warnings'])
    
    # Save sample output
    builder.save_output(sample_output, "output/sample_output.json")
    
    # Create debug output
    debug_output = builder.create_debug_output(
        ['breakfast_ideas.pdf', 'lunch_ideas.pdf'],
        'Personal Chef and Meal Planning Specialist',
        'Create a comprehensive weekly meal plan with high-protein, balanced nutrition',
        sample_output['extracted_sections'],
        sample_output['subsection_analysis'],
        "output/sample_output.json"
    )
    
    print("✅ OutputBuilder testing complete!")
