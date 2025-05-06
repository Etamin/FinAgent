# PDF RAG Project

This folder contains scripts and [Haystack](https://haystack.deepset.ai/) pipelines for question answering (QA) on PDF documents.

## Scripts

- **`pdf_rag/pipelines/index_chunks.py`**  
  Builds the indexing pipeline: parses, chunks, and indexes PDF content into a local vector store (`qdrant/` folder).  
  - Parser: [Docling](https://github.com/docling-project/docling)  
  - Chunking: [HybridChunker](https://docling-project.github.io/docling/examples/hybrid_chunking/)   (from Docling)  
  - Vector store: [Qdrant](https://qdrant.tech/)

- **`pdf_rag/pipelines/generate_answer.py`**  
  Takes a user query and generates an answer using a local LLM ([Ollama](https://ollama.com/)) or an [OpenAI](https://openai.com/) model.
