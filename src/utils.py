"""
utils.py — Text Processing Utilities
=====================================
Provides text preprocessing, tokenization, chunking, and formatting helpers.
"""

import re
import logging

log = logging.getLogger(__name__)

# Rough token-to-character ratio for GPT models (1 token ≈ 4 chars)
TOKENS_PER_CHAR_RATIO = 0.25


def count_tokens(text: str) -> int:
    """
    Estimate token count using character-based approximation.
    
    This is a rough estimate. Actual token count depends on the tokenizer.
    For accurate counts, use tiktoken library.
    
    Args:
        text: Input text string.
    
    Returns:
        Estimated token count.
    """
    # Rough approximation: 1 token ≈ 4 characters
    return max(1, int(len(text) * TOKENS_PER_CHAR_RATIO))


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing extra whitespace and artifacts.
    
    Args:
        text: Raw input text.
    
    Returns:
        Cleaned text.
    """
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple newlines (keep max 2 consecutive)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove common artifacts from web scraping
    text = re.sub(r'\[.*?\]', '', text)  # Remove [links]
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text


def chunk_text(text: str, chunk_size: int = 3000) -> list[str]:
    """
    Split text into overlapping chunks for processing long documents.
    
    Args:
        text: Input text to chunk.
        chunk_size: Target size per chunk in tokens.
    
    Returns:
        List of text chunks.
    """
    if not text or chunk_size <= 0:
        return [text] if text else []
    
    # Convert token size to approximate character size
    char_size = int(chunk_size / TOKENS_PER_CHAR_RATIO)
    
    # Split by paragraphs first to preserve context
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        # If single paragraph exceeds chunk size, split it by sentences
        if para_size > char_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = []
            sent_size = 0
            
            for sent in sentences:
                sent_len = len(sent)
                if sent_size + sent_len > char_size and sent_chunk:
                    chunks.append(' '.join(sent_chunk))
                    sent_chunk = [sent]
                    sent_size = sent_len
                else:
                    sent_chunk.append(sent)
                    sent_size += sent_len + 1
            
            if sent_chunk:
                chunks.append(' '.join(sent_chunk))
        
        elif current_size + para_size > char_size and current_chunk:
            # Start new chunk
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        
        else:
            # Add to current chunk
            current_chunk.append(para)
            current_size += para_size + 2  # +2 for \n\n
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    log.info("Split text into %d chunks", len(chunks))
    return chunks


def format_summary(summary: str, style: str) -> str:
    """
    Format the summary output based on the requested style.
    
    Args:
        summary: Raw summary text from the model.
        style: One of 'brief', 'detailed', 'bullet'.
    
    Returns:
        Formatted summary string.
    """
    if not summary:
        return "No summary generated."
    
    summary = summary.strip()
    
    if style == "bullet":
        # Ensure bullet points are properly formatted
        lines = summary.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('•') and not line.startswith('-') and not line.startswith('*'):
                formatted_lines.append(f"• {line}")
            else:
                formatted_lines.append(line)
        return '\n'.join(formatted_lines)
    
    elif style == "brief":
        # Return as-is for brief summaries
        return summary
    
    else:  # detailed
        # Add slight formatting for detailed summaries
        return summary
