# PDF RAG Project

The folder contains scripts and Haystack pipelines for PDF-based QA tasks.

## Scripts

### Offline scripts

-   **`1-process_pdf.py`**: This script is responsible for parsing the input PDF document and splitting its content into smaller, manageable chunks.
-   **`2-index_chunks.py`**: This script takes the generated chunks and indexes them into a Qdrant vector store. The vector store is persisted to disk within the `qdrant/` directory.

### Online scripts
-   **`3-generate_answer.py`**: This script takes user query and answer it using a local LLM.