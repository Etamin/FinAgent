# FinAgent: Intelligent Financial Question Answering

FinAgent is an intelligent assistant designed to answer financial questions by drawing information from various sources, including:
*   A local SQL database (for bank account information, transactions).
*   The Yahoo Finance API (for stock market data).
*   A collection of PDF documents (using Retrieval Augmented Generation - RAG).

The system features a Gradio-based web interface and supports multilingual queries (English, German, Luxembourgish), translating them for internal processing and providing answers in the original language.

## Key Features

*   **Multi-Source Question Answering:** Dynamically routes questions to the most appropriate data source (SQL, API, or PDF RAG).
*   **Retrieval Augmented Generation (RAG):** Extracts information from PDF documents to answer complex queries.
*   **SQL Database Interaction:** Queries a local SQLite database for structured financial data.
*   **Live Stock Data:** Fetches stock market information via the Yahoo Finance API.
*   **Multilingual Support:** Handles questions in English, German, and Luxembourgish.
*   **LLM-Powered:** Utilizes Large Language Models (via Ollama) for translation, classification, query generation, and natural language response generation.
*   **Interactive UI:** Provides a user-friendly web interface built with Gradio.
*   **Haystack Pipelines:** Built using the Haystack framework for robust NLP pipelines.

## Setup & Prerequisites

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd FinAgent
    ```

2.  **Python Environment:**
    Ensure you have Python 3.8+ installed. It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install the required Python packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ollama & LLMs:**
    *   Install Ollama: https://ollama.com/
    *   Pull the necessary LLM models used by the project (e.g., `gemma3:12b`):
        ```bash
        ollama pull gemma3:12b
        ```

5.  **Qdrant Vector Store:**
    Qdrant is used for the PDF RAG pipeline. The vector store will be created locally in a `qdrant/` directory when you run the indexing script.

6.  **Sample Database:**
    The project expects a SQLite database named `bank_demo.db` in the project root for SQL-based queries. Ensure this file is present and contains the `transactions` table as expected by the `sqlpipe` function.

7.  **PDF Documents for RAG:**
    *   Place your PDF documents in the `pdf_rag/data/original_pdf/` directory. These documents will be indexed for the RAG pipeline.

8.  **(Optional) OpenAI API Key:**
    If you plan to use OpenAI models for the RAG pipeline (as an alternative to Ollama), ensure your `OPENAI_API_KEY` environment variable is set.

## Running the Project

There are two main steps to get FinAgent running:

### 1. Index PDF Documents (for RAG pipeline)

This step parses, chunks, and indexes the content of your PDFs into the Qdrant vector store.

*   The primary script for this is `pdf_rag/pipelines/index_chunks.py`. 
*   Run:

    ```bash
    python pdf_rag/pipelines/index_chunks.py
    ```
    This will create/update the `qdrant/` folder in your project root. This step only needs to be done once or whenever you add/update PDF documents.

### 2. Run the FinAgent Application (Gradio UI)

Once the prerequisites are met and PDF documents (if used) are indexed, start the main application:

```bash
python haystack_yahooapi.py
```

This will:
*   Initialize the Haystack pipelines.
*   Start a Gradio web server.
*   Print the URL to access the UI in your console (e.g., `http://127.0.0.1:7860` or a public Gradio link if `share=True`).

Open the URL in your web browser to interact with FinAgent.

## Project Structure

```
FinAgent/
├── haystack_yahooapi.py    # Main application, UI, and core logic
├── bank_demo.db            # Sample SQLite database for SQL queries
├── pdf_rag/                # Sub-module for PDF Retrieval Augmented Generation
│   ├── README.md           # README specific to the PDF RAG component
│   ├── data/
│   │   └── original_pdf/   # Place your PDF documents here
│   ├── pipelines/
│   │   ├── index_chunks.py # Script to index PDF documents
│   │   └── generate_answer.py # Script for RAG answer generation
│   └── ...
├── logs/                   # Directory for log files (created at runtime)
├── qdrant/                 # Directory for Qdrant vector store (created by indexing)
└── ... (other potential files like requirements.txt)
```

## Logging

The application generates detailed logs in the `logs/` directory, with filenames like `FinAgent_log_YYYY-MM-DD_HH-MM-SS.log`. These logs include Haystack component tracing and execution times.