# main.py
import time
import psutil
import os
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# -----------------------------
# Step 1: Prepare plant paragraphs
# -----------------------------
plant_paragraph = """
Plants are living organisms that belong to the kingdom Plantae. They can produce their own food through photosynthesis and are essential for life on Earth.
Photosynthesis occurs mainly in the leaves of plants, where chlorophyll captures sunlight to convert carbon dioxide and water into glucose and oxygen.
Plant roots anchor them into the soil and absorb water and minerals necessary for their growth.
Stems provide support to plants, helping them stand upright and transport nutrients between roots and leaves.
Leaves are the primary site of photosynthesis and are adapted in different plants to various environmental conditions.
Flowers are the reproductive organs of most plants and play a crucial role in pollination and seed formation.
Pollination can occur through wind, insects, or animals, leading to fertilization and fruit development.
Seeds allow plants to reproduce and spread, giving rise to new plants when conditions are favorable.
Some plants adapt to extreme environments, such as deserts or wetlands, by modifying their leaves, roots, or stems.
Overall, plants maintain the balance of oxygen and carbon dioxide in the atmosphere and provide food, shelter, and medicine to humans and animals.
"""

plant_chunks = [chunk.strip() for chunk in plant_paragraph.strip().split('\n') if chunk]

# -----------------------------
# Step 2: Generate embeddings
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(plant_chunks)
print("✅ Generated", len(embeddings), "embeddings of size", len(embeddings[0]))

# -----------------------------
# Step 3: Connect to Redis
# -----------------------------
r = redis.Redis(host='localhost', port=6379, db=0)

# -----------------------------
# Step 4: Create vector index
# -----------------------------
VECTOR_DIM = len(embeddings[0])
INDEX_NAME = "plant_idx"

# Drop index if exists (clean run)
try:
    r.ft(INDEX_NAME).dropindex(delete_documents=False)
except:
    pass

# Create index
r.ft(INDEX_NAME).create_index(
    fields=[
        TextField("content"),
        VectorField(
            "embedding",
            "FLAT",
            {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"}
        )
    ],
    definition=IndexDefinition(prefix=["plant:"], index_type=IndexType.HASH)
)

# -----------------------------
# Step 5: Store embeddings and measure insertion metrics
# -----------------------------
process = psutil.Process(os.getpid())

start_time = time.time()
for i, emb in enumerate(embeddings):
    r.hset(f"plant:{i}", mapping={
        "content": plant_chunks[i],
        "embedding": np.array(emb, dtype=np.float32).tobytes()
    })
end_time = time.time()


insertion_time = end_time - start_time
insertion_throughput = len(embeddings) / insertion_time

print("\n📊 Insertion Metrics")
print(f"✅ Total insertion time: {insertion_time:.4f} seconds")
print(f"✅ Insertion throughput: {insertion_throughput:.2f} embeddings/sec")

# -----------------------------
# Step 6: Queries and measure query performance
# -----------------------------
queries = [
    "How do plants make food?",
    "Which part helps plants stand upright?",
    "What are flowers for in plants?",
    "How do plants survive extreme conditions?"
]

query_embeddings = model.encode(queries)
TOP_K = 2

total_query_time = 0

for qi, q_emb in enumerate(query_embeddings):
    query_vector = np.array(q_emb, dtype=np.float32).tobytes()
    q = Query(f"*=>[KNN {TOP_K} @embedding $vector AS score]") \
        .return_fields("content", "score") \
        .sort_by("score", asc=True) \
        .paging(0, TOP_K)

    query_start = time.time()

    results = r.ft(INDEX_NAME).search(q, query_params={"vector": query_vector})

    query_end = time.time()

    query_latency = query_end - query_start
    total_query_time += query_latency

    print(f"\n🔍 Query: {queries[qi]} (latency: {query_latency:.4f}s)")
    for doc in results.docs:
        print(f"  ✅ {doc.content} (score={doc.score})")

avg_latency = total_query_time / len(queries)
query_throughput = len(queries) / total_query_time

print("\n📊 Query Metrics")
print(f"✅ Average query latency: {avg_latency:.4f} seconds")
print(f"✅ Query throughput: {query_throughput:.2f} queries/sec")
