import pickle
from chonkie import TokenChunker, WordChunker, SentenceChunker, SDPMChunker, LateChunker
from autotiktokenizer import AutoTikTokenizer
from langchain_core.documents.base import Document
from tokenizers import Tokenizer
import tiktoken
from langchain_community.document_loaders import TextLoader
from semantic_text_splitter import TextSplitter


tokenizer='gpt2'
tokenizer = AutoTikTokenizer.from_pretrained(tokenizer_name_or_path=tokenizer)

def token_chunker(document, tokenizer=tokenizer, chunk_size=512, chunk_overlap=0):
    
    chunker = TokenChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,  # maximum tokens per chunk
        chunk_overlap=chunk_overlap  # overlap between chunks
    )

    chonkie_chunks = chunker.chunk(document.page_content)
    chunks = [Document(page_content=chunk.text, metadata={'source': document.metadata['source']}) for chunk in chonkie_chunks]

    return chunks


def word_chunker(document, tokenizer=tokenizer, mode='advanced', chunk_size=512, chunk_overlap=0):
    """
    mode: chunking mode
        simple: basic space-based splitting
        advanced: handles punctuation and special cases
    """
    

    chunker = WordChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # mode=mode
    )

    chonkie_chunks = chunker.chunk(document.page_content)
    chunks = [Document(page_content=chunk.text, metadata={'source': document.metadata['source']}) for chunk in chonkie_chunks]

    return chunks
    

def sentence_chunker(document, tokenizer=tokenizer, mode='simple', chunk_size=512, chunk_overlap=0):
    """
    mode: Sentence detection mode
        "simple" (rule-based)
    """
    

    chunker = SentenceChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        mode=mode, 
    )

    chonkie_chunks = chunker.chunk(document.page_content)
    chunks = [Document(page_content=chunk.text, metadata={'source': document.metadata['source']}) for chunk in chonkie_chunks]

    return chunks


def semantic_chunker(document, embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     chunk_size=512, similarity_threshold=0.7):
    from chonkie import SemanticChunker
    
    # chunker = SemanticChunker(
    #     embedding_model=embedding_model,
    #     max_chunk_size=max_chunk_size,
    #     similarity_threshold=similarity_threshold,
    # )

    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-8M",  # Default model
        threshold=similarity_threshold,              # Similarity threshold (0-1) or (1-100) or "auto"
        chunk_size=chunk_size,                   # Maximum tokens per chunk
        min_sentences=1                              # Initial sentences per chunk
    )

    chonkie_chunks = chunker.chunk(document.page_content)
    chunks = [Document(page_content=chunk.text, metadata={'source': document.metadata['source']}) for chunk in chonkie_chunks]

    return chunks


def sdpm_chunker(document, embedding_model="minishlab/potion-base-8M",
                 chunk_size=512, similarity_threshold=0.5, chunk_overlap=0):
    encoder = tiktoken.get_encoding("gpt2")
    
    def tokenize(text):
        return encoder.encode(text)
    
    def detokenize(tokens):
        return encoder.decode(tokens)
    chunker = SDPMChunker(
        embedding_model=embedding_model,    # Default embedding model.
        threshold=similarity_threshold,     # Similarity threshold.
        chunk_size=chunk_size,              # Maximum tokens per chunk.
        min_sentences=1,                    # Initial sentences per chunk.
        skip_window=1                       # This parameter is used internally by SDPMChunker.
    )
    
    # Generate the initial chunks using SDPMChunker.
    raw_chunks = chunker.chunk(document.page_content)
    chunks = [
        Document(page_content=chunk.text, metadata={'source': document.metadata['source']})
        for chunk in raw_chunks
    ]
    
    # If no overlap is requested, return the chunks as is.
    if chunk_overlap <= 0:
        return chunks
    
    # Create new chunks that include an overlap from the previous chunk.
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            # The first chunk remains unchanged.
            overlapped_chunks.append(chunk)
        else:
            # Tokenize the previous chunk's text.
            prev_tokens = tokenize(chunks[i - 1].page_content)
            # Get the last 'chunk_overlap' tokens.
            overlap_tokens = prev_tokens[-chunk_overlap:] if len(prev_tokens) >= chunk_overlap else prev_tokens
            overlap_text = detokenize(overlap_tokens)
            # Prepend the overlapping tokens to the current chunk's text.
            new_text = overlap_text + " " + chunk.page_content
            overlapped_chunks.append(Document(page_content=new_text, metadata=chunk.metadata))
    
    return overlapped_chunks


def late_chunker(document, embedding_model="all-MiniLM-L6-v2",
                 chunk_size=512, chunk_overlap=0):
    encoder = tiktoken.get_encoding("gpt2")
    
    def tokenize(text):
        return encoder.encode(text)
    
    def detokenize(tokens):
        return encoder.decode(tokens)
    chunker = LateChunker(
        embedding_model=embedding_model,
        mode = "sentence",
        chunk_size=chunk_size,
        min_sentences_per_chunk=1,
        min_characters_per_sentence=12,
    )
    
    # Generate the initial chunks using SDPMChunker.
    raw_chunks = chunker.chunk(document.page_content)
    
    chunks = [
        Document(page_content=chunk.text, metadata={'source': document.metadata['source']})
        for chunk in raw_chunks
    ]
 
    # If no overlap is requested, return the chunks as is.
    if chunk_overlap <= 0:
        return chunks
    
    # Create new chunks that include an overlap from the previous chunk.
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            # The first chunk remains unchanged.
            overlapped_chunks.append(chunk)
        else:
            # Tokenize the previous chunk's text.
            prev_tokens = tokenize(chunks[i - 1].page_content)
            # Get the last 'chunk_overlap' tokens.
            overlap_tokens = prev_tokens[-chunk_overlap:] if len(prev_tokens) >= chunk_overlap else prev_tokens
            overlap_text = detokenize(overlap_tokens)
            # Prepend the overlapping tokens to the current chunk's text.
            new_text = overlap_text + " " + chunk.page_content
            overlapped_chunks.append(Document(page_content=new_text, metadata=chunk.metadata))
    
    return overlapped_chunks



def semantic_chunker_bert(document, max_tokens=512, chunk_overlap=0):

    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens, overlap=chunk_overlap)

    bert_chunks = splitter.chunks(document.page_content)
    chunks = [Document(page_content=chunk, metadata={'source': document.metadata['source']}) for chunk in bert_chunks]
    
    return chunks

if __name__ == '__main__':
    pass
