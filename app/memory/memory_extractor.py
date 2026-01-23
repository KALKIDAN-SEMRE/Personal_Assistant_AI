"""
Memory extraction logic - identifies meaningful facts from conversations.
Only extracts candidate memories, not every message.
"""
import logging
import re
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """
    Extracts meaningful memories from conversation messages.
    Uses pattern matching and heuristics to identify facts worth remembering.
    """
    
    # Patterns that suggest a memory-worthy statement
    MEMORY_PATTERNS = [
        r"i (like|love|prefer|enjoy|hate|dislike)",
        r"i (am|was|will be)",
        r"my (favorite|preferred|favourite)",
        r"i (want|need|wish|hope)",
        r"i (work|study|live) (at|in|for)",
        r"my (name|age|birthday|email|phone)",
        r"i (have|own|don't have)",
        r"i (can't|cannot|can) (do|eat|drink)",
        r"i'm (allergic|intolerant) (to|of)",
        r"my (goal|objective|plan) (is|to)",
    ]
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize memory extractor.
        
        Args:
            min_confidence: Minimum confidence threshold for extraction
        """
        self.min_confidence = min_confidence
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.MEMORY_PATTERNS]
    
    def extract_candidates(self, messages: List) -> List[Dict]:
        """
        Extract candidate memories from conversation messages.
        
        Args:
            messages: List of ChatMessage objects
            
        Returns:
            List of candidate memory dictionaries with text and confidence
        """
        candidates = []
        
        # Focus on user messages (assistant responses less likely to contain user facts)
        user_messages = [msg for msg in messages if msg.role == "user"]
        
        for msg in user_messages:
            text = msg.content.strip()
            if not text or len(text) < 10:  # Skip very short messages
                continue
            
            # Check if message matches memory patterns
            confidence = self._calculate_confidence(text)
            
            if confidence >= self.min_confidence:
                # Extract a clean memory statement
                memory_text = self._extract_memory_text(text)
                
                if memory_text:
                    candidates.append({
                        "text": memory_text,
                        "confidence": confidence,
                        "source_message": text,
                        "metadata": {
                            "extracted_from": "conversation",
                            "pattern_matched": True
                        }
                    })
        
        logger.debug(f"Extracted {len(candidates)} memory candidates from {len(user_messages)} user messages")
        return candidates
    
    def _calculate_confidence(self, text: str) -> float:
        """
        Calculate confidence that a message contains a memory-worthy fact.
        
        Args:
            text: Message text
            
        Returns:
            Confidence score between 0 and 1
        """
        text_lower = text.lower()
        
        # Check pattern matches
        pattern_matches = sum(1 for pattern in self.compiled_patterns if pattern.search(text))
        
        if pattern_matches == 0:
            return 0.0
        
        # Base confidence from pattern matches
        confidence = min(0.5 + (pattern_matches * 0.2), 1.0)
        
        # Boost confidence for statements with specific structures
        if any(phrase in text_lower for phrase in ["i am", "i like", "i prefer", "my favorite"]):
            confidence = min(confidence + 0.2, 1.0)
        
        # Reduce confidence for questions
        if text.strip().endswith("?"):
            confidence *= 0.5
        
        # Reduce confidence for very long messages (likely not a simple fact)
        if len(text) > 200:
            confidence *= 0.7
        
        return confidence
    
    def _extract_memory_text(self, text: str) -> Optional[str]:
        """
        Extract a clean memory statement from the message.
        
        Args:
            text: Original message text
            
        Returns:
            Cleaned memory text or None
        """
        # Simple extraction: take the sentence containing the memory pattern
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(pattern.search(sentence) for pattern in self.compiled_patterns):
                # Clean up the sentence
                sentence = sentence.strip('.,!?;:')
                if len(sentence) > 10 and len(sentence) < 200:
                    return sentence
        
        # If no sentence matches, return cleaned version of original
        cleaned = text.strip('.,!?;:')
        if 10 < len(cleaned) < 200:
            return cleaned
        
        return None
    
    def should_store(self, candidate: Dict) -> bool:
        """
        Determine if a candidate memory should be stored.
        
        Args:
            candidate: Candidate memory dictionary
            
        Returns:
            True if should be stored
        """
        return candidate.get("confidence", 0) >= self.min_confidence


# Global memory extractor instance
memory_extractor = MemoryExtractor()
