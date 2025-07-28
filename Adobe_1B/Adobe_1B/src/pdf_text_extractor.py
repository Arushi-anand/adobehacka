import fitz  # PyMuPDF
import os
import re
from typing import List, Dict, Any

class PDFTextExtractor:
    def __init__(self):
        """Initialize PDF text extractor"""
        pass
    
    def extract_text_with_structure(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF while preserving structure and page information
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Dict]: List of text sections with metadata
        """
        try:
            doc = fitz.open(pdf_path)
            sections = []
            document_name = os.path.basename(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean and split text into meaningful sections
                cleaned_sections = self._process_page_text(text, page_num + 1, document_name)
                sections.extend(cleaned_sections)
            
            doc.close()
            return sections
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
    
    def _process_page_text(self, text: str, page_num: int, document_name: str) -> List[Dict[str, Any]]:
        """
        Process text from a single page into structured sections
        
        Args:
            text (str): Raw text from the page
            page_num (int): Page number
            document_name (str): Name of the PDF document
            
        Returns:
            List[Dict]: Processed text sections
        """
        sections = []
        
        # Split text into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        for i, paragraph in enumerate(paragraphs):
            if self._is_valid_section(paragraph):
                section = {
                    'text': paragraph.strip(),
                    'page': page_num,
                    'document': document_name,
                    'section_id': f"{document_name}_page{page_num}_section{i+1}",
                    'word_count': len(paragraph.split()),
                    'char_count': len(paragraph)
                }
                sections.append(section)
        
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into meaningful paragraphs
        
        Args:
            text (str): Raw text
            
        Returns:
            List[str]: List of paragraphs
        """
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Split by double line breaks (paragraph boundaries)
        paragraphs = text.split('\n\n')
        
        # Further split long paragraphs that might contain multiple ideas
        refined_paragraphs = []
        for para in paragraphs:
            if len(para) > 1000:  # If paragraph is too long
                # Try to split by sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) > 500:
                        if current_chunk:
                            refined_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                if current_chunk:
                    refined_paragraphs.append(current_chunk.strip())
            else:
                refined_paragraphs.append(para)
        
        return refined_paragraphs
    
    def _is_valid_section(self, text: str) -> bool:
        """
        Check if a text section is valid for processing
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if valid section
        """
        text = text.strip()
        
        # Minimum length requirement
        if len(text) < 50:
            return False
        
        # Must have at least 8 words
        if len(text.split()) < 8:
            return False
        
        # Skip headers, footers, page numbers
        if self._is_header_footer_or_pagenumber(text):
            return False
        
        # Skip table of contents entries
        if self._is_toc_entry(text):
            return False
        
        return True
    
    def _is_header_footer_or_pagenumber(self, text: str) -> bool:
        """Check if text is likely a header, footer, or page number"""
        text = text.strip().lower()
        
        # Common header/footer patterns
        header_footer_patterns = [
            r'^\d+$',  # Just a number (page number)
            r'^page \d+',  # "Page X"
            r'^\d+ of \d+$',  # "X of Y"
            r'^chapter \d+$',  # "Chapter X"
            r'^section \d+$',  # "Section X"
        ]
        
        for pattern in header_footer_patterns:
            if re.match(pattern, text):
                return True
        
        # Very short text that's likely not content
        if len(text) < 20 and not any(word in text for word in ['abstract', 'introduction', 'conclusion']):
            return True
        
        return False
    
    def _is_toc_entry(self, text: str) -> bool:
        """Check if text is likely a table of contents entry"""
        # Look for patterns like "Chapter 1 ......... 15"
        if re.search(r'\.{3,}', text) and re.search(r'\d+$', text.strip()):
            return True
        
        # Look for numbered list patterns without substantial content
        if re.match(r'^\d+\.?\s+[A-Z][^.]*\s*\d*$', text.strip()):
            return True
        
        return False
    
    def extract_from_multiple_pdfs(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """
        Extract text from multiple PDFs in a directory
        
        Args:
            pdf_directory (str): Directory containing PDF files
            
        Returns:
            List[Dict]: Combined sections from all PDFs
        """
        all_sections = []
        
        # Get all PDF files in directory
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"Processing: {pdf_file}")
            
            sections = self.extract_text_with_structure(pdf_path)
            all_sections.extend(sections)
            
            print(f"Extracted {len(sections)} sections from {pdf_file}")
        
        print(f"Total sections extracted: {len(all_sections)}")
        return all_sections
    
    def get_document_stats(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about extracted sections
        
        Args:
            sections (List[Dict]): List of extracted sections
            
        Returns:
            Dict: Statistics about the extraction
        """
        if not sections:
            return {}
        
        documents = {}
        total_words = 0
        total_chars = 0
        
        for section in sections:
            doc_name = section['document']
            if doc_name not in documents:
                documents[doc_name] = {
                    'sections': 0,
                    'pages': set(),
                    'words': 0,
                    'chars': 0
                }
            
            documents[doc_name]['sections'] += 1
            documents[doc_name]['pages'].add(section['page'])
            documents[doc_name]['words'] += section['word_count']
            documents[doc_name]['chars'] += section['char_count']
            
            total_words += section['word_count']
            total_chars += section['char_count']
        
        # Convert sets to counts
        for doc in documents:
            documents[doc]['pages'] = len(documents[doc]['pages'])
        
        return {
            'total_documents': len(documents),
            'total_sections': len(sections),
            'total_words': total_words,
            'total_characters': total_chars,
            'documents': documents,
            'avg_words_per_section': total_words / len(sections) if sections else 0,
            'avg_chars_per_section': total_chars / len(sections) if sections else 0
        }

# Example usage and testing
if __name__ == "__main__":
    extractor = PDFTextExtractor()
    
    # Test with single PDF
    # sections = extractor.extract_text_with_structure("sample.pdf")
    
    # Test with multiple PDFs
    # sections = extractor.extract_from_multiple_pdfs("input/")
    
    # Get statistics
    # stats = extractor.get_document_stats(sections)
    # print("Extraction Statistics:")
    # print(f"Total documents: {stats['total_documents']}")
    # print(f"Total sections: {stats['total_sections']}")
    # print(f"Average words per section: {stats['avg_words_per_section']:.1f}")
    
    pass