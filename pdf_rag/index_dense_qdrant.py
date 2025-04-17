import os
import pickle
import time
import pandas as pd
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from datasets import load_dataset
from haystack import Document
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.rankers import TransformersSimilarityRanker


def format_execution_time(start_time, end_time):
    # Calculate the time difference
    execution_time = end_time - start_time

    # Convert to hh:mm:ss format
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # time_formatted = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    time_formatted = "{}h {}m {}s".format(int(hours), int(minutes), int(seconds))

    return time_formatted

def load_data():
    name = "bilgeyucel/seven-wonders"
    dataset = load_dataset(name, split="train")
    documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
    return documents

def load_dataset():
    df = pd.read_json("./datasets/financebench_open_source.jsonl", lines=True)
    dataset = df.copy(deep=True)
    dataset = dataset.sort_values(by='doc_name')

    return dataset

def load_chunks(chunks_path):
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    documents = [Document(content=chunk.page_content, meta=chunk.metadata) for chunk in chunks]
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
    path = 'databases/qdrant/dense'
    document_store = QdrantDocumentStore(
        ":memory:",
        # path=path,
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

def retrieve_context(query, document_store, top_k, embedder_name):
    # Components
    _, q_embedder = get_embedders(embedder_name)
    print(f'queries embedder: {q_embedder.model}')
    retriever = QdrantEmbeddingRetriever(document_store=document_store)

    # Pipeline
    retrieving_pipeline = Pipeline()

    retrieving_pipeline.add_component(instance=q_embedder, name='q_embedder')
    retrieving_pipeline.add_component(instance=retriever, name='retriever')
    retrieving_pipeline.connect("q_embedder.embedding", "retriever.query_embedding")
    # print(retrieving_pipeline)

    retriever_args = {
        'top_k': top_k,
        'scale_score': False,
        'return_embedding': True,
    }
    results = retrieving_pipeline.run({'q_embedder': {'text': query}, 'retriever': retriever_args})['retriever']['documents']
    # print(results)

    return results

def rerank_context(query, contexts, top_k):
    device = ComponentDevice.from_str("cuda:0")

    # ranker_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ranker_name = 'Alibaba-NLP/gte-reranker-modernbert-base'
    # ranker_name = 'BAAI/bge-reranker-base'
    ranker = TransformersSimilarityRanker(model=ranker_name, device=device)
    ranker.warm_up()

    results = ranker.run(query=query, documents=contexts, top_k=top_k)
    reranked_docs = results['documents']

    return reranked_docs

def process_row(row, document_store, top_k, top_k_r, embedder_name):
    query = row['question']
    retrieved = retrieve_context(query, document_store, top_k, embedder_name)
    reranked = rerank_context(query, retrieved, top_k_r)

    # Document to dict
    retrieved = [doc.to_dict() for doc in retrieved]
    reranked = [doc.to_dict() for doc in reranked]

    return pd.Series({
        'retrieved_contexts': retrieved,
        'reranked_contexts': reranked
    })


def run_retriever(documents, dataset, embedder_name, top_k, top_k_r):
    start_time = time.time()

    document_store = index_docs(documents, embedder_name)

    end_time = time.time()
    indexing_time = end_time - start_time
    indexing_time = format_execution_time(start_time, end_time)
    print(f"Indexing process finished. Total indexing time: {indexing_time}")

    start_time = time.time()

    dataset.loc[:, ['retrieved_contexts', 'reranked_contexts']] = dataset.apply(
    lambda row: process_row(row, document_store, top_k, top_k_r, embedder_name),
    axis=1
    )

    end_time = time.time()
    indexing_time = end_time - start_time
    indexing_time = format_execution_time(start_time, end_time)
    print(f"Retrieval process finished. Total Retrieval time: {indexing_time}")

    return dataset

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
    
    parsers = ['pypdf2', 'pymupdf', 'pdfminer', 'unstructured-ocr']
    chunking_methods = ['token', 'word', 'sentence', 'recursive', 'semantic-bert', 'sdpm', 'late']
    # chunking_methods = ['sdpm', 'late']
    size_overlaps = ['1024_256']
    # embedders = ['multi-qa-cos', 'gte-base', 'mutli-qa', 'bge-base-inst', 'alibaba-modern', 'nomic-modern']

    embedders = ['mutli-qa', 'bge-base-inst', 'alibaba-modern']

    top_k = 30
    top_k_r = 5

    for embedder in embedders:
        embedder_name = embedders_mapping[embedder]
        for size_overlap in size_overlaps:
            for parser in parsers:
                for chunking_method in chunking_methods:
                    chunks_path = os.path.join('chunks', chunking_method, f'{parser}_{size_overlap}.pkl')
                    print(chunks_path)
                    chunks = load_chunks(chunks_path)
                    
                    finance_bench_df = load_dataset()

                    dataframe = run_retriever(chunks, finance_bench_df, embedder_name, top_k, top_k_r)

                    # Save dataframe
                    contexts_path = os.path.join('contexts', embedder_name.replace('/', '-'), size_overlap, f'{parser}_{chunking_method}.pkl')
                    os.makedirs(os.path.dirname(contexts_path), exist_ok=True)
                    dataframe.to_pickle(contexts_path)
