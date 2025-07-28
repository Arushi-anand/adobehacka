"""
ADOBE_1B - Advanced Document Processing Pipeline

A sophisticated text processing and analysis pipeline for outline loading,
persona matching, section ranking, and intelligent output building.
"""

__version__ = "1.0.0"
__author__ = "ADOBE_1B Team"

from .outline_loader import OutlineLoader
from .persona_matcher import PersonaMatcher
from .section_ranker import SectionRanker
from .subsection_analyzer import SubsectionAnalyzer
from .output_builder import OutputBuilder

__all__ = [
    "OutlineLoader",
    "PersonaMatcher", 
    "SectionRanker",
    "SubsectionAnalyzer",
    "OutputBuilder",
] 