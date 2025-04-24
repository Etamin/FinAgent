import os
import pickle
from PyPDF2 import PdfReader
from chonkie import SentenceChunker
from chunkers import semantic_chunker_bert
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


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

    pdf_name = os.path.basename(pdf_path)
    text_file_path = f'pdf_rag/data/parsed_pdf/{pdf_name}.txt'

    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Parsed text saved to: {text_file_path}")

    return text

def chunk_text(document, chunk_size=512, chunk_overlap=0):
    """Chunk text using Chonky's SentenceChunker"""
    chunks = semantic_chunker_bert(document, max_tokens=chunk_size, chunk_overlap=chunk_overlap)
    
    return chunks

def save_chunks(chunks, output_path):
    """Save chunks to a .pkl file"""
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)

def load_document(file_path) -> Document:
    text_loader = TextLoader(file_path)
    text = text_loader.load()

    return text[0]

def process_pdf(pdf_path, chunk_size=512, chunk_overlap=0):
    """Main processing pipeline."""
    print(f"Parsing PDF: {pdf_path}")
    # Parse the PDF and save the text
    parse_pdf(pdf_path)

    # Load the parsed text as a langchain_core.documents.Document
    pdf_name = os.path.basename(pdf_path)
    text_file_path = f'pdf_rag/data/parsed_pdf/{pdf_name}.txt'
    document = load_document(file_path=text_file_path)

    print(f"Chunking text into sentences (chunk_size={chunk_size}, overlap={chunk_overlap})")
    chunks = chunk_text(document, chunk_size, chunk_overlap)

    output_path = f'pdf_rag/data/chunks/{pdf_name}.pkl'
    print(f"Saving {len(chunks)} chunks to: {output_path}")
    save_chunks(chunks, output_path)

if __name__ == "__main__":    
    chunk_size = 512
    chunk_overlap = 128

    folder_path = 'pdf_rag/data/original_pdf'
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            process_pdf(pdf_path, chunk_size, chunk_overlap)
            print('' + '-' * 50 + '\n')
    print("All PDFs processed.")