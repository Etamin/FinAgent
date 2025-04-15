import os
import pickle
from PyPDF2 import PdfReader
from chonkie import SentenceChunker

def parse_pdf(pdf_path):
    """Extract raw text from a PDF file"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=512, chunk_overlap=0):
    """Chunk text using Chonky's SentenceChunker"""
    chunker = SentenceChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return chunker.chunk(text)

def save_chunks(chunks, output_path):
    """Save chunks to a .pkl file"""
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)

def process_pdf(pdf_path, output_path, chunk_size=512, chunk_overlap=0):
    """Main processing pipeline."""
    print(f"Parsing PDF: {pdf_path}")
    text = parse_pdf(pdf_path)

    print(f"Chunking text into sentences (chunk_size={chunk_size}, overlap={chunk_overlap})")
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    print(f"Saving {len(chunks)} chunks to: {output_path}")
    save_chunks(chunks, output_path)

if __name__ == "__main__":
    pdf_path = 'data/original_pdf/AMCOR_2023Q4_EARNINGS.pdf'
    output_path = 'data/pdf/chunks/pdf_chunks.pkl'
    chunk_size = 512
    chunk_overlap = 128

    process_pdf(pdf_path, output_path, chunk_size, chunk_overlap)