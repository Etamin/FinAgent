import pickle
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack import Document
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


def load_chunks(chunks_path):
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    documents = [Document(content=chunk.text) for chunk in chunks]
    print(f'{len(documents)} documents have been loaded!')
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

def index_docs(documents, embedder_name):
    path = 'qdrant'
    document_store = QdrantDocumentStore(
        # ":memory:",
        path=path,
        # url='localhost:6333',
        index="dense",
        recreate_index=False,
        hnsw_config={"m": 16, "ef_construct": 100},  # Optional
        # Please choose one of the options: cosine, dot_product, l2
        similarity='cosine',
        on_disk=True
    )
    
    # Components
    docs_embedder, _ = get_embedders(embedder_name)
    print(f'docs embedder: {docs_embedder.model}')
    docs_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

    # Pipeline
    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(instance=docs_embedder, name="embedder")
    indexing_pipeline.add_component(instance=docs_writer, name="writer")
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    # print(indexing_pipeline)

    indexing_pipeline.run({"documents": documents})

    print(document_store.count_documents())

    return document_store


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

    # Parameters
    embedder_name = embedders_mapping['gte-base']

    # Load chunks
    chunks_path = '/FinAgent/data/chunks/pdf_chunks.pkl'
    chunks = load_chunks(chunks_path)

    # Run indexing pipeline
    document_store = index_docs(chunks, embedder_name)