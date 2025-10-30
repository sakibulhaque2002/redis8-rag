# main_rag_chunked.py
import time
import redis
import numpy as np
import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import CrossEncoder

# -----------------------------
# Step 1: Read PDF
# -----------------------------
PDF_PATH = "data/banglaTest.pdf"
doc = fitz.open(PDF_PATH)

pdf_texts = []
for page in doc:
    text = page.get_text().replace("\n", " ")
    pdf_texts.append(text)

print(f"✅ Loaded {len(pdf_texts)} pages")

# -----------------------------
# Step 2: Split pages into smaller chunks
# -----------------------------
def chunk_text(text, max_words=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

all_chunks = []
for page_text in pdf_texts:
    chunks = chunk_text(page_text, max_words=50)  # 100 words per chunk
    all_chunks.extend(chunks)

print(f"✅ Total chunks: {len(all_chunks)}")

# -----------------------------
# Step 3: Generate embeddings
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer('BAAI/bge-m3', device=device)

print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")

embeddings = model.encode(
    all_chunks,
    batch_size=2,
    show_progress_bar=True,
    device=device
)

VECTOR_DIM = len(embeddings[0])
print(f"✅ Generated embeddings of size {VECTOR_DIM}")

# -----------------------------
# Step 4: Connect to Redis
# -----------------------------
r = redis.Redis(host='localhost', port=6379, db=0)
INDEX_NAME = "pdf_chunked_idx"

try:
    r.ft(INDEX_NAME).dropindex(delete_documents=False)
except:
    pass

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

# -----------------------------
# Step 5: Store chunks in Redis
# -----------------------------
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

print(f"✅ Stored {len(all_chunks)} chunks in Redis")

# -----------------------------
# Step 6: Queries and retrieval
# -----------------------------
queries = [
    # "What did the person wear while farming?"
    # "বাংলাদেশের রাজধানী কোন শহর?",
    # "বাংলাদেশের স্বাধীনতা কখন অর্জিত হয়েছিল?",
    # "বাংলাদেশের প্রধান রফতানি খাত কোনটি?",
    # "বাংলাদেশের পদ্মা সেতু কোন ধরনের প্রতীক হিসেবে পরিচিত?",
    # "বাংলাদেশের সবচেয়ে বড় ধর্মীয় উৎসবগুলো কী কী?"
    "বাংলাদেশ কবে মুক্তি পায়?"
]

TOP_K = 3  # retrieve top 3 relevant chunks
total_query_time = 0

for query_text in queries:
    query_start = time.time()

    q_emb = model.encode([query_text], device=device)[0]

    q_vector = np.array(q_emb, dtype=np.float16).tobytes()

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

avg_latency = total_query_time / len(queries)
query_throughput = len(queries) / total_query_time

print("\n📊 Query Metrics")
print(f"✅ Average query latency: {avg_latency:.4f} seconds")
print(f"✅ Query throughput: {query_throughput:.2f} queries/sec")


# -----------------------------
# Step 6b: Initialize BGE reranker
# -----------------------------
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
print("✅ Loaded BGE reranker")

# -----------------------------
# Step 6c: Rerank retrieved chunks
# -----------------------------

total_query_time=0

for query_text in queries:
    query_start = time.time()

    q_emb = model.encode([query_text], device=device)[0]
    q_vector = np.array(q_emb, dtype=np.float16).tobytes()

    q = Query(f"*=>[KNN {TOP_K} @embedding $vector AS score]") \
        .return_fields("content", "score") \
        .sort_by("score", asc=True) \
        .paging(0, TOP_K)

    results = r.ft(INDEX_NAME).search(q, query_params={"vector": q_vector})

    # Prepare (query, doc) pairs for reranking
    rerank_inputs = [(query_text, doc.content) for doc in results.docs]

    # Rerank
    rerank_scores = reranker.predict(rerank_inputs)

    # Combine docs with rerank scores
    reranked_results = sorted(
        zip(results.docs, rerank_scores),
        key=lambda x: x[1],  # sort by reranker score descending
        reverse=True
    )

    query_end = time.time()

    query_latency = query_end - query_start
    total_query_time += query_latency

    print(f"\n✅ Query: {query_text} (after reranking)")
    for rank, (doc, score) in enumerate(reranked_results, start=1):
        print(f"{rank}. ({score:.4f}) {doc.content}")

avg_latency = total_query_time / len(queries)
query_throughput = len(queries) / total_query_time

print("\n📊 Query Metrics")
print(f"✅ Average query latency: {avg_latency:.4f} seconds")
print(f"✅ Query throughput: {query_throughput:.2f} queries/sec")