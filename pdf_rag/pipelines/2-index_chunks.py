import os
import pickle
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack import Document
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


def load_chunks(chunks_path):
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    documents = [Document(content=chunk.page_content, meta=chunk.metadata) for chunk in chunks]
    print(f"{len(documents)} documents loaded from {os.path.basename(chunks_path)}")
    return documents

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
    """Create the embedder->writer pipeline once and reuse it for every batch."""
    docs_embedder, _ = get_embedders(embedder_name)
    docs_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

    pipeline = Pipeline()
    pipeline.add_component(instance=docs_embedder, name="embedder")
    pipeline.add_component(instance=docs_writer, name="writer")
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
            'multi-qa-cos': 'sentence-transformers/multi-qa-mpnet-base-cos-v1'
    }

    # 1- Choose embedder
    embedder_name = embedders_mapping["gte-base"]

    # 2- Create / reopen the (single) Qdrant store
    document_store = QdrantDocumentStore(
        path="qdrant",
        index="dense",
        recreate_index=False,  # set True if you need a clean slate
        hnsw_config={"m": 16, "ef_construct": 100},
        similarity="cosine",
        on_disk=True,
    )

    # 3️- Build the pipeline ONCE
    indexing_pipeline = build_indexing_pipeline(document_store, embedder_name)

    # 4- Index every .pkl in the chunks folder
    chunks_folder = "pdf_rag/data/chunks"
    for filename in os.listdir(chunks_folder):
        if filename.endswith(".pkl"):
            chunks_path = os.path.join(chunks_folder, filename)
            documents = load_chunks(chunks_path)
            indexing_pipeline.run({"documents": documents})

    print(f"Finished. Vector store now holds {document_store.count_documents()} documents.")