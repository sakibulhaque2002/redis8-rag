# main_rag_chunked.py
import time
import redis
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# -----------------------------
# Step 1: Read PDF
# -----------------------------
PDF_PATH = "/home/sakib/Desktop/test.pdf"
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
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks)
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
            {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"}
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
        "embedding": np.array(emb, dtype=np.float32).tobytes()
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
    "What is the capital of Bangladesh?",
    "Which rivers form the delta of Bangladesh?",
    "When did Bangladesh gain independence?",
    "What is the official language of Bangladesh?",
    "Name two major religious festivals in Bangladesh."
]

TOP_K = 2  # retrieve top 3 relevant chunks
total_query_time = 0

for query_text in queries:
    q_emb = model.encode([query_text])[0]
    q_vector = np.array(q_emb, dtype=np.float32).tobytes()

    q = Query(f"*=>[KNN {TOP_K} @embedding $vector AS score]") \
        .return_fields("content", "score") \
        .sort_by("score", asc=True) \
        .paging(0, TOP_K)

    query_start = time.time()
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