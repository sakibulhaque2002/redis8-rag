import time
import redis
import fitz

from openai import OpenAI
import numpy as np
from typing import Dict, Any


# Connect to vLLM servers
embedding_client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
reranker_client = OpenAI(base_url="http://localhost:8002", api_key="EMPTY")

# Device selection
device = "cuda"
print("Using device:", device)

use_redis = False
use_qdrant = True

if use_redis:
    from redis.commands.search.field import VectorField, TextField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    r = redis.Redis(host='localhost', port=6380, decode_responses=False)
    REDIS_INDEX = "pdf_chunked_idx"

    # Checking the existence of redis db
    def index_exists(redis_client, index_name):
        try:
            redis_client.ft(index_name).info()
            return True
        except Exception:
            return False

if use_qdrant:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams

    qclient = QdrantClient(host="localhost", port=6333)
    QDRANT_COLLECTION = "pdf_chunked_coll"

    # Checking the existence of qdrant db
    def collection_exists(client: QdrantClient, name: str) -> bool:
        try:
            client.get_collection(name)
            return True
        except Exception:
            return False

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

# Embedding function using model
def get_embeddings(texts):
    response = embedding_client.embeddings.create(
        model="BAAI/bge-m3",
        input=texts
    )
    return [np.array(r.embedding, dtype=np.float16) for r in response.data]



# Ask user if they want a fresh database
user_input = input("Do you want to use a fresh database for both Redis and Qdrant? (y/n): ").strip().lower()
if user_input == 'y':
    chunk_size = int(input("Enter chunk size: "))

    if use_redis:
        try:
            r.ft(REDIS_INDEX).dropindex(delete_documents=True)
            print("✅ Existing Redis index and documents deleted.")
        except Exception:
            print("No existing Redis index found.")

    if use_qdrant:
        try:
            qclient.delete_collection(collection_name=QDRANT_COLLECTION)
            print("✅ Existing Qdrant collection deleted.")
        except Exception:
            print("No existing Qdrant collection found.")

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

else:
    print("✅ Using existing databases.")

# Checking the persistence of Redis
if use_redis and index_exists(r, REDIS_INDEX):
    print("✅ Existing Redis index found — skipping PDF re-embedding.")

if use_qdrant and collection_exists(qclient, QDRANT_COLLECTION):
    print("✅ Qdrant collection found — skipping re-embedding.")


if use_redis and not index_exists(r, REDIS_INDEX):
    print("⚙️ No existing Redis data found. Processing PDF and creating embeddings...")
    # Creating redis index if DB does not exist
    r.ft(REDIS_INDEX).create_index(
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

    print("\n📊 Redis Insertion Metrics")
    print(f"✅ Total insertion time: {insertion_time:.4f} seconds")
    print(f"✅ Insertion throughput: {insertion_throughput:.2f} embeddings/sec")


if use_qdrant and not collection_exists(qclient, QDRANT_COLLECTION):
    print("⚙️ No Qdrant collection found — will process PDF and create embeddings.")

    VECTOR_DIM = len(embeddings[0])
    qclient.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance="Cosine")
    )


    batch_size = 128
    total_points = len(embeddings)
    start_time = time.time()
    for batch_start in range(0, total_points, batch_size):
        batch_end = min(batch_start + batch_size, total_points)
        points = [
            {
                "id": i,
                "vector": np.array(embeddings[i], dtype=np.float16).tolist(),
                "payload": {"content": all_chunks[i]}
            }
            for i in range(batch_start, batch_end)
        ]
        qclient.upsert(collection_name=QDRANT_COLLECTION, points=points)
    end_time = time.time()

    insertion_time = end_time - start_time
    throughput = total_points / insertion_time
    print("\n📊 Qdrant Insertion Metrics")
    print(f"✅ Total insertion time: {insertion_time:.4f}s")
    print(f"✅ Insertion throughput: {throughput:.2f} embeddings/sec")

# List of Queries
queries = [
    "Where was Charles KingsIey born?"
    # "What was the institution where Walter de la Mare studied?",
    # "When did Anne die?",
    # "When did Emily visited Brussels?",
    # "After how many attempts did Benjamin succeed in gaining a seat in Parliament?",
    # "What happened in 1604 at Hampton Court?",
    # "What caused the end of his medical career?"
]

# user_input = input("Enter your queries (separated by commas): ")
# queries = [q.strip() for q in user_input.split(",") if q.strip()]

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

# --- Redis querying metrics collection ---
total_query_time_redis = 0.0
total_rerank_time_redis = 0.0

# --- Qdrant querying metrics collection ---
total_query_time_qdrant = 0.0
total_rerank_time_qdrant = 0.0

print("\n\n===== Running queries for Redis and Qdrant (with reranking) =====")
# Generating results for each query
# for query_text in queries:
for qi, query_text in enumerate(queries):

    q_emb = get_embeddings([query_text])[0]
    q_vector = np.array(q_emb, dtype=np.float16).tobytes()
    q_list = np.array(q_emb, dtype=np.float16).tolist()

    # --- Qdrant search ---
    if use_qdrant:
        q_start = time.time()
        results = qclient.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q_list,
            limit=TOP_K,
            with_payload=True
        )
        q_end = time.time()
        q_latency = q_end - q_start
        total_query_time_qdrant += q_latency

        print(f"\n🔎 Qdrant Query: {query_text} (latency: {q_latency:.4f}s)")
        qdrant_docs = []
        for result in results:
            for points in result[1]:
                doc_text = points.payload["content"]
                qdrant_docs.append(doc_text)
                print(f"({points.score:.4f}). {doc_text}")

        # Rerank for Qdrant
        rerank_start = time.time()
        rerank_scores_q = rerank(query_text, qdrant_docs)
        reranked_q = sorted(zip(qdrant_docs, rerank_scores_q), key=lambda x: x[1], reverse=True)
        rerank_end = time.time()
        rerank_time_q = rerank_end - rerank_start
        total_rerank_time_qdrant += rerank_time_q

        print(f"\n✅ Qdrant Query: {query_text} (after reranking)")
        for rank, (doc, score) in enumerate(reranked_q, start=1):
            print(f"  {rank}. ({score:.4f}) {doc}")

    # --- Redis search ---
    if use_redis:
        query_start = time.time()
        q = Query(f"*=>[KNN {TOP_K} @embedding $vector AS score]") \
            .return_fields("content", "score") \
            .sort_by("score", asc=True) \
            .paging(0, TOP_K)
        results = r.ft(REDIS_INDEX).search(q, query_params={"vector": q_vector})

        query_end = time.time()
        query_latency = query_end - query_start
        total_query_time_redis += query_latency

        print(f"\n🔍 Redis Query: {query_text} (latency: {query_latency:.4f}s)")
        for rank, doc in enumerate(results.docs, start=1):
            print(f"({float(doc.score):.4f}). {doc.content} ")

        # Rerank for redis
        documents = [doc.content for doc in results.docs]
        rerank_start = time.time()
        rerank_scores = rerank(query_text, documents)
        reranked_results = sorted(
            zip(results.docs, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        rerank_end = time.time()
        rerank_time = rerank_end - rerank_start
        total_rerank_time_redis += rerank_time

        print(f"\n✅ Redis Query: {query_text} (after reranking)")
        for rank, (doc, score) in enumerate(reranked_results, start=1):
            print(f"  {rank}. ({score:.4f}) {doc.content}")


# ---------- Final Metrics Calculation ----------
# Redis
if use_redis:
    avg_query_latency_redis = total_query_time_redis / len(queries)
    avg_rerank_latency_redis = total_rerank_time_redis / len(queries)
    avg_total_latency_redis = (total_query_time_redis + total_rerank_time_redis) / len(queries)
    query_throughput_redis = len(queries) / total_query_time_redis if total_query_time_redis > 0 else 0.0
    rerank_throughput_redis = len(queries) / total_rerank_time_redis if total_rerank_time_redis > 0 else 0.0
    total_throughput_redis = len(queries) / (total_query_time_redis + total_rerank_time_redis) if (total_query_time_redis + total_rerank_time_redis) > 0 else 0.0

    # Print Redis metrics
    print("\n\n📊 Redis Query Metrics")
    print(f"✅ Average query latency: {avg_query_latency_redis:.4f} seconds")
    print(f"✅ Average rerank latency: {avg_rerank_latency_redis:.4f} seconds")
    print()
    print(f"✅ Query throughput: {query_throughput_redis:.2f} queries/sec")
    print(f"✅ Rerank throughput: {rerank_throughput_redis:.2f} queries/sec")

# Qdrant
if use_qdrant:
    avg_query_latency_q = total_query_time_qdrant / len(queries)
    avg_rerank_latency_q = total_rerank_time_qdrant / len(queries)
    avg_total_latency_q = (total_query_time_qdrant + total_rerank_time_qdrant) / len(queries)
    query_throughput_q = len(queries) / total_query_time_qdrant if total_query_time_qdrant > 0 else 0.0
    rerank_throughput_q = len(queries) / total_rerank_time_qdrant if total_rerank_time_qdrant > 0 else 0.0
    total_throughput_q = len(queries) / (total_query_time_qdrant + total_rerank_time_qdrant) if (total_query_time_qdrant + total_rerank_time_qdrant) > 0 else 0.0

    # Print Qdrant metrics
    print("\n\n📊 Qdrant Query Metrics")
    print(f"✅ Average query latency: {avg_query_latency_q:.4f} seconds")
    print(f"✅ Average rerank latency: {avg_rerank_latency_q:.4f} seconds")
    print()
    print(f"✅ Query throughput: {query_throughput_q:.2f} queries/sec")
    print(f"✅ Rerank throughput: {rerank_throughput_q:.2f} queries/sec")