import time
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.rankers import TransformersSimilarityRanker
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder
import requests
from retrieval_pipeline import run_retriever, run_generator

def format_execution_time(start_time, end_time):
    # Calculate the time difference
    execution_time = end_time - start_time

    # Convert to hh:mm:ss format
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # time_formatted = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    time_formatted = "{}h {}m {}s".format(int(hours), int(minutes), int(seconds))

    return time_formatted


def get_embedders(embedder_name):
    device = ComponentDevice.from_str("cuda:0")

    docs_embedder = SentenceTransformersDocumentEmbedder(model=embedder_name, trust_remote_code=True, device=device)
    q_embedder = SentenceTransformersTextEmbedder(model=embedder_name, trust_remote_code=True, device=device)

    if embedder_name == 'bge-base-inst':
        instruction = "Represent this sentence for searching relevant passages:"
        docs_embedder = SentenceTransformersDocumentEmbedder(model=embedder_name, trust_remote_code=True, device=device, prefix=instruction)
        q_embedder = SentenceTransformersTextEmbedder(model=embedder_name, trust_remote_code=True, device=device, prefix=instruction)

    return docs_embedder, q_embedder

def load_index():
    persistent_path = "qdrant"
    document_store = QdrantDocumentStore(
        # ":memory:",
        path=persistent_path,
        index="dense",
        recreate_index=False,
        hnsw_config={"m": 16, "ef_construct": 100},  # Optional
        # Please choose one of the options: cosine, dot_product, l2
        similarity='cosine',
        on_disk=True
    )

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

def run_retriever(query, embedder_name, top_k, top_k_r):
    # Load Qdrant index
    document_store = load_index()

    # Run retriever and reranker
    start_time = time.time()

    retrieved_context = retrieve_context(query, document_store, top_k, embedder_name)
    reranked_context = rerank_context(query, retrieved_context, top_k_r)

    end_time = time.time()
    retrieval_time = end_time - start_time
    retrieval_time = format_execution_time(start_time, end_time)
    print(f"Retrieval process finished. Total Retrieval time: {retrieval_time}")

    return reranked_context

def run_generator(query, contexts, llm):
    template = """
    CONTEXT:
    {% for document in documents %}
        {{ loop.index }}. {{ document.content }}
    {% endfor %}

    QUESTION: {{ query }}?
    
    INSTRUCTIONS: Answer the QUESTION using only the information provided in the CONTEXT above. Do not rely on any prior knowledge. If the CONTEXT does not contain sufficient information to answer the QUESTION, respond with 'No answer'.

    """

    system_prompt = 'You are a financial expert specializing in corporate financial reports and filings.'

    prompt_builder = PromptBuilder(template=template)

    generator = OllamaGenerator(
        model=llm,
        url="http://localhost:11434", # http://localhost:11434 for local inference and http://trux-dgx01.uni.lux:11434/ for DGX
        system_prompt=system_prompt,
        timeout=750
    )

    # Create the pipeline and add components
    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)
    pipe.connect("prompt_builder", "generator")

    try:
        result = pipe.run({
            "prompt_builder": {
                "documents": contexts,
                "query": query
            }
        })
        answer = result["generator"]["replies"][0]
        return answer
    except requests.exceptions.Timeout:
        # Handle the timeout exception
        print("A timeout occurred while processing the row.")
        return "TIMEOUT"
    except Exception as e:
        # Handle other potential exceptions
        print(f"An error occurred: {e}")

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
    top_k = 30
    top_k_r = 5
    embedder_name = embedders_mapping['gte-base']
    llm = 'gemma3:4b'

    # User question
    query = 'What is the capital of Morocco?'
    # Run retriever
    contexts = run_retriever(query, embedder_name, top_k, top_k_r)
    answer = run_generator(query, contexts, llm)
    print(f"Answer: {answer}")