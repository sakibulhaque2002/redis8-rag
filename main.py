import time
import redis
import fitz
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from openai import OpenAI
import numpy as np
from typing import Dict, Any

# Connect to vLLM servers
embedding_client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
reranker_client = OpenAI(base_url="http://localhost:8002", api_key="EMPTY")


#Chunking function
def chunk_full_text(text, max_words=2000, overlap=100):
    print(f"Maximum words in each chunk: {max_words}")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

# Device selection
device = "cuda"
print("Using device:", device)

# Embedding function using model
def get_embeddings(texts):
    response = embedding_client.embeddings.create(
        model="BAAI/bge-m3",
        input=texts
    )
    return [np.array(r.embedding, dtype=np.float16) for r in response.data]

# Checking the existence of DB
def index_exists(redis_client, index_name):
    try:
        redis_client.ft(index_name).info()
        return True
    except Exception:
        return False

# Checking the emptiness of DB
def is_index_populated(redis_client, index_name):
    try:
        info = redis_client.ft(index_name).info()
        return info.get("num_docs", 0) > 0
    except Exception:
        return False

# Redis connection
r = redis.Redis(host='localhost', port=6379, decode_responses=False)
INDEX_NAME = "pdf_chunked_idx"

# Ask user if they want a fresh database
user_input = input("Do you want to use a fresh database? (y/n): ").strip().lower()
if user_input == 'y':
    chunk_size=int(input("Enter the chunk size: "))
    r.ft(INDEX_NAME).dropindex(delete_documents=True)
    print(f"✅ Existing index and documents deleted. Fresh database ready.")
else:
    print("✅ Using existing database and index.")

# Checking the persistence of Redis
if index_exists(r, INDEX_NAME) and is_index_populated(r, INDEX_NAME):
    print("✅ Existing Redis index found — skipping PDF re-embedding.")
else:
    print("⚙️ No existing Redis data found. Processing PDF and creating embeddings...")

    # Read PDF
    PDF_PATH = "data/english.pdf"
    doc = fitz.open(PDF_PATH)
    full_text = " ".join([page.get_text().replace("\n", " ") for page in doc])

    # Creating chunks
    all_chunks = chunk_full_text(full_text, max_words=chunk_size)
    print(f"✅ Total chunks: {len(all_chunks)}")

    # Generating embeddings
    embeddings = get_embeddings(all_chunks)
    VECTOR_DIM = len(embeddings[0])
    print(f"✅ Generated embeddings of size {VECTOR_DIM}")

    # Creating index if DB does not exist
    r.ft(INDEX_NAME).create_index(
        fields=[
            TextField("content"),
            VectorField(
                "embedding",
                "FLAT",
                {"TYPE": "FLOAT16", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"}
            )
        ],
        definition=IndexDefinition(prefix=["chunk:"], index_type=IndexType.HASH)
    )

    # Store chunks in Redis
    start_time = time.time()
    for i, emb in enumerate(embeddings):
        r.hset(f"chunk:{i}", mapping={
            "content": all_chunks[i],
            "embedding": np.array(emb, dtype=np.float16).tobytes()
        })
    end_time = time.time()
    insertion_time = end_time - start_time
    insertion_throughput = len(embeddings) / insertion_time

    print("\n📊 Insertion Metrics")
    print(f"✅ Total insertion time: {insertion_time:.4f} seconds")
    print(f"✅ Insertion throughput: {insertion_throughput:.2f} embeddings/sec")

# List of Queries
queries = [
    "Where was Charles KingsIey born?",
    "What was the institution where Walter de la Mare studied?",
    "When did Anne die?"
]

# Reranking function using model
def rerank(query, docs):
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": query,
        "documents": docs
    }

    response = reranker_client.post(
        path="/v1/rerank",
        body=payload,
        cast_to=Dict[str, Any]
    )

    results = response["results"]
    scores = [item["relevance_score"] for item in results]
    return scores

# Top k elements retrieval
TOP_K = 3
total_query_time = 0
total_rerank_time = 0

# Generating results for each query
for query_text in queries:
    query_start = time.time()
    q_emb = get_embeddings([query_text])[0]
    q_vector = np.array(q_emb, dtype=np.float16).tobytes()

    # Cosine similarity search
    q = Query(f"*=>[KNN {TOP_K} @embedding $vector AS score]") \
        .return_fields("content", "score") \
        .sort_by("score", asc=True) \
        .paging(0, TOP_K)
    results = r.ft(INDEX_NAME).search(q, query_params={"vector": q_vector})

    query_end = time.time()
    query_latency = query_end - query_start
    total_query_time += query_latency

    print(f"\n🔍 Query: {query_text} (latency: {query_latency:.4f}s)")
    for rank, doc in enumerate(results.docs, start=1):
        print(f"({float(doc.score):.4f}). {doc.content} ")


    # Rerank
    rerank_start = time.time()
    documents = [doc.content for doc in results.docs]
    rerank_scores = rerank(query_text, documents)
    reranked_results = sorted(
        zip(results.docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )
    rerank_end = time.time()
    rerank_time = rerank_end - rerank_start
    total_rerank_time += rerank_time

    print(f"\n✅ Query: {query_text} (after reranking)")
    for rank, (doc, score) in enumerate(reranked_results, start=1):
        print(f"{rank}. ({score:.4f}) {doc.content}")

avg_query_latency = total_query_time / len(queries)
avg_rerank_latency = total_rerank_time / len(queries)
avg_total_latency = (total_query_time + total_rerank_time) / len(queries)

query_throughput = len(queries) / total_query_time
rerank_throughput= len(queries) / total_rerank_time
total_throughput = len(queries) / (total_query_time + total_rerank_time)

print("\n📊 Query Metrics")
print(f"✅ Average query latency: {avg_query_latency:.4f} seconds")
print(f"✅ Average rerank latency: {avg_rerank_latency:.4f} seconds")
print(f"✅ Average total latency: {avg_total_latency:.4f} seconds")
print()
print(f"✅ Query throughput: {query_throughput:.2f} queries/sec")
print(f"✅ Rerank throughput: {rerank_throughput:.2f} queries/sec")
print(f"✅ Total throughput: {total_throughput:.2f} queries/sec")