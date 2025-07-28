import json
from typing import List, Dict
from pathlib import Path

class OutlineLoader:
    """Load and convert outline data from adobehackathon JSON output"""
    
    def load_from_json(self, json_path: str) -> List[Dict]:
        """
        Load sections from adobehackathon heading extraction JSON
        
        Args:
            json_path: Path to the JSON file containing heading extraction results
            
        Returns:
            List of section dictionaries compatible with the ranking system
        """
        sections = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract document name from file path
            doc_name = Path(json_path).stem.replace('_headings', '') + '.pdf'
            
            # Get title
            title = data.get('title', '')
            
            # Process outline entries
            outline = data.get('outline', [])
            
            for i, entry in enumerate(outline):
                section = {
                    'document': doc_name,
                    'page': entry.get('page', 1),
                    'title': entry.get('text', ''),
                    'level': entry.get('level', 'Unknown'),
                    'content': entry.get('text', ''),  # Use title as content for now
                    'section_index': i,
                    'source': 'heading_extraction'
                }
                
                # Skip entries that are just page numbers or very short
                if len(section['title']) > 2 and not section['title'].startswith('[Page'):
                    sections.append(section)
            
            # If we have a title but it's not in the outline, add it as the first section
            if title and not any(s['title'] == title for s in sections):
                sections.insert(0, {
                    'document': doc_name,
                    'page': 1,
                    'title': title,
                    'level': 'Title',
                    'content': title,
                    'section_index': -1,
                    'source': 'heading_extraction'
                })
            
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
        
        return sections
    
    def load_multiple_jsons(self, input_dir: str) -> List[Dict]:
        """
        Load sections from all heading extraction JSON files in a directory
        
        Args:
            input_dir: Directory containing JSON files
            
        Returns:
            Combined list of sections from all files
        """
        all_sections = []
        json_files = list(Path(input_dir).glob('*_headings.json'))
        
        for json_file in json_files:
            sections = self.load_from_json(str(json_file))
            all_sections.extend(sections)
            
        return all_sections