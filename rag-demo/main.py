import pdfplumber
import re

def create_string_from_pdf():
    with pdfplumber.open('csharp-in-a-nutshell.pdf') as pdf:
        full_text = ''

        for page in pdf.pages:
            text = page.extract_text()
            full_text += text + ' '
    
        return full_text

def create_chunks(text, max_chunk_size=1000, overlap_sentences=2):
    sentences = re.split(r'(?<=[.!?])\s+', text) 

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            current_chunk = current_chunk[-overlap_sentences:]
            current_size = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks