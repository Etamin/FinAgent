import os
from pathlib import Path
import pickle
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack import Document
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from docling_haystack.converter import DoclingConverter, ExportType
from docling.chunking import HybridChunker



def get_embedders(embedder_name):
    device = ComponentDevice.from_str("cuda:0")

    docs_embedder = SentenceTransformersDocumentEmbedder(model=embedder_name, trust_remote_code=True, device=device)
    q_embedder = SentenceTransformersTextEmbedder(model=embedder_name, trust_remote_code=True, device=device)

    if embedder_name == 'bge-base-inst':
        instruction = "Represent this sentence for searching relevant passages:"
        docs_embedder = SentenceTransformersDocumentEmbedder(model=embedder_name, trust_remote_code=True, device=device, prefix=instruction)
        q_embedder = SentenceTransformersTextEmbedder(model=embedder_name, trust_remote_code=True, device=device, prefix=instruction)

    return docs_embedder, q_embedder

def build_indexing_pipeline(document_store, embedder_name):
    """Create the pipeline for indexing documents into the Qdrant store."""
    chunker = HybridChunker(tokenizer=embedder_name)
    export_type = ExportType.DOC_CHUNKS     # To convert PDF to chunks in loading time
    converter = DoclingConverter(export_type=export_type,
                                 chunker=chunker)
    
    docs_embedder, _ = get_embedders(embedder_name)
    docs_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

    # Add components to the pipeline
    pipeline = Pipeline()
    pipeline.add_component(instance=converter, name="converter")
    pipeline.add_component(instance=docs_embedder, name="embedder")
    pipeline.add_component(instance=docs_writer, name="writer")
    
    # Connect the components
    pipeline.connect("converter", "embedder")
    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline


if __name__ == '__main__':
    embedders_mapping = {
            'gte-base': 'Alibaba-NLP/gte-base-en-v1.5',
            'mutli-qa': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'bge-base-inst': 'BAAI/bge-base-en-v1.5',
            'alibaba-modern': 'Alibaba-NLP/gte-modernbert-base',
            'nomic-modern': 'nomic-ai/modernbert-embed-base',
            'modernbert-large': 'answerdotai/ModernBERT-large',
            'multi-qa-cos': 'sentence-transformers/multi-qa-mpnet-base-cos-v1',
            'all-minilm': 'sentence-transformers/all-MiniLM-L6-v2'
    }

    # 1- Choose embedder
    embedder_name = embedders_mapping["gte-base"]

    # 2- Create / reopen the (single) Qdrant store
    document_store = QdrantDocumentStore(
        path="qdrant",
        index="dense",
        recreate_index=True,  # set True if you need a clean slate
        hnsw_config={"m": 16, "ef_construct": 100},
        similarity="cosine",
        on_disk=True,
    )

    # 3- Create the pipeline
    indexing_pipeline = build_indexing_pipeline(document_store, embedder_name)

    # 4- Run the pipeline
    parent_folder = Path("/home/laura/PDay/FinAgent/pdf_rag/data/original_pdf")
    pdf_paths = list(parent_folder.glob("*.pdf"))
  
    indexing_pipeline.run({"converter": {"paths": pdf_paths}})

    print(f"Finished. Vector store now holds {document_store.count_documents()} documents.")