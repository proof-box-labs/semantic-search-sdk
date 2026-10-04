# 🧠 Qdrant + Vector Databases — A C# Engineer's Complete Guide

> **The vibe:** you know SQL. You know `WHERE title LIKE '%rag%'`. Now you need *"find me stuff that **means** this"* — and suddenly everyone's yelling HNSW, RRF, quantization, cross-encoders at you.
> This guide takes you from *"what even is a vector"* to *"my multi-tenant RAG service survived the on-call page"* — in C#, with pictures.

```
        YOU ARE HERE                                           YOU LEAVE HERE
   ┌────────────────────┐                                ┌────────────────────────┐
   │ "vector db = magic │  ──── PART 1 ─── PART 2 ───▶   │ hybrid retrieval,      │
   │  search thing?"    │        theory     hands-on     │ reranking, quantization│
   └────────────────────┘                   ──PART 3──▶  │ sharding, DR, RBAC     │
                                            production   └────────────────────────┘
```

> Three parts, in the order you asked for:
> **Part 1** — end-to-end vector-database theory *and* Qdrant, every major capability, with C#.
> **Part 2** — a quick hands-on you can clone, run, and see results in ~10 minutes.
> **Part 3** — a production-grade project (`DocMind`: a .NET RAG service on Qdrant + Semantic Kernel + MCP) built out with real deployment, resilience, security, backup, and observability detail.
>
> All code uses the official `Qdrant.Client` NuGet package (gRPC, port 6334). Compiled from the official Qdrant docs — see the source map at the end.

### 🗺️ How to read this (pick your energy level)

| Your situation | Read this | Skip for now |
|---|---|---|
| "I have 10 minutes and a Docker daemon" | **Part 2** first, then Part 1 | Part 3 until you ship |
| "I need to understand the machine" | Part 1, §1 → §10, in order | — |
| "I'm designing a real service Monday" | §5, §8, §9, then **all of Part 3** | §8.6 (niche) |
| "Something is slow/wrong in prod" | §3.12 ops runbook → jump to the § it cites | everything else |

**Legend for the callouts you'll see:**

```
 💡 mental model  — the analogy that makes it click
 ⚠️  gotcha       — the thing that silently ruins your recall at 2am
 🎯 default       — what to do when you have no eval set and no time
 📐 the math      — receipts, for when you need to defend a number
```

---

## Table of Contents

**Part 1 — Vector DBs with Qdrant (end-to-end)**
1. Vector database fundamentals (stack-agnostic)
2. Qdrant architecture & data model
3. Collections
4. Points — writing and reading data
5. Indexing — payload, full-text, HNSW, sparse
6. Filtering
7. Search & the Query API
8. Advanced retrieval — prefetch, hybrid fusion, formula, recommend/discover, matrix
   - 8.7 Sparse retrieval & BM25 · 8.8 Reranking (fusion / late-interaction / **cross-encoder**) · 8.9 Top-k, oversampling & MMR · 8.10 Query-side techniques (rewriting, multi-query, HyDE, contextual retrieval) · 8.11 Reference pipeline
9. Quantization
10. Production internals — sharding, replication, consistency, optimizers, snapshots, security, observability

**Part 2 — Quick hands-on with C#**

**Part 3 — Production project: DocMind (RAG on Qdrant + SK + MCP)**

---

# PART 1 — Vector Databases with Qdrant

## 1. Vector database fundamentals

A vector database stores **embeddings** — fixed-length arrays of floats produced by a model — and answers *"which stored vectors are nearest to this query vector?"* Nearness is a proxy for semantic similarity, which is what makes semantic search, RAG, recommendations, and deduplication possible.

> 💡 **Mental model.** A relational DB is a *filing cabinet*: you find things by matching labels exactly. A vector DB is a *map*: every document gets coordinates in "meaning space", and search = "who's near me?". `"sourdough starter"` and `"bread yeast culture"` share zero keywords but they're neighbours on the map.

```
              MEANING SPACE  (2 of 768 dimensions, obviously)
      ▲
      │            🍞 "sourdough starter"
      │        🍞 "bread yeast culture"
      │   🍞 "slow-roast lamb"
      │                                   🔍 query: "ANN indexing"
      │                                        ↓ nearest
      │                          📀 "tuning HNSW for recall"
      │                     📀 "intro to vector databases"
      │                           📀 "hybrid dense + sparse"
      └───────────────────────────────────────────────────────▶
        keyword search sees 6 unrelated strings. the map sees 2 clusters.
```

**Embeddings & dimensionality.** An embedding model maps text/image/audio to a vector of fixed dimension `d` (e.g. 384 for `all-MiniLM-L6-v2`, 768 for `bge-base`, 1536 for OpenAI `text-embedding-3-small`, 3072 for `-3-large`). Higher `d` usually captures more nuance but costs more memory and compute. Memory for raw `float32` vectors ≈ `num_vectors × d × 4 bytes` — 1M × 1536 × 4 ≈ **6.1 GB** before any index or payload. This single fact drives most production decisions (quantization, on-disk storage, sharding).

> 📐 **The RAM bill, visualized.** 1M vectors, `float32`, before index or payload:
>
> ```
>  d=384   ████                       1.5 GB    (all-MiniLM-L6-v2)
>  d=768   ████████                   3.1 GB    (bge-base)
>  d=1536  ████████████████           6.1 GB    (text-embedding-3-small)
>  d=3072  ████████████████████████  12.3 GB    (text-embedding-3-large)
>  └─ every § about quantization (§9) and on-disk storage exists because of this chart
> ```

**Distance metrics.** Three matter in practice:
- **Cosine** — angle between vectors; the default for most text embeddings. Insensitive to magnitude.
- **Dot product** — cosine × magnitudes; correct when your model is trained for it (many modern models are) and vectors are normalized.
- **Euclidean (L2)** — straight-line distance; common for image/where magnitude matters.

```
   COSINE — "same direction?"      DOT — cosine × length        EUCLIDEAN — "how far?"
        ▲   b                           ▲   b (long)                 ▲   b
        │  ╱                            │  ╱                         │  ╱│
        │ ╱ θ ← only the angle          │ ╱ ← angle AND how          │ ╱ │ ← straight-line
        │╱________ a                    │╱    confident it is        │╱  │   gap counts
        length ignored                  magnitude counts             ●───┘
```

Pick the metric your embedding model was **trained with**. Using the wrong one silently degrades recall. If your model expects cosine, either choose `Cosine` (Qdrant normalizes internally) or normalize yourself and use `Dot` (faster).

> ⚠️ **Gotcha.** The wrong metric doesn't throw. It doesn't log. It just returns slightly-worse results forever — and you blame the LLM.

**Exact vs approximate (ANN).** Exact nearest-neighbour compares the query against every stored vector — O(N), accurate, and too slow past a few hundred thousand vectors. **Approximate Nearest Neighbour (ANN)** trades a little recall for orders-of-magnitude speed. Qdrant's ANN index is **HNSW** (Hierarchical Navigable Small World): a multi-layer proximity graph where search greedily hops toward the query. You tune it with `m` (edges per node), `ef_construct` (build-time breadth), and `ef` (search-time breadth). More on this in §5.

```
  EXACT: check literally everyone.         ANN (HNSW): a skip-list, but for space.
  ● ● ● ● ● ● ● ● ● ● ● ● ●
  ● ● ● ● ● ● ● ● ● ● ● ● ●           L2  ●───────────────────●   ← few nodes, huge hops
  ● ● ● ● ● ● ● ● ● ● ● ● ●               │   (descend)       │
  ● ● ● ● ● ● ● ● ● ● ● ● ●           L1  ●─────●───────●─────●   ← more nodes, medium hops
  ● ● ● ● ● ● ● ● ● ● ● ● ●               │     │       │
  O(N). perfectly correct.            L0  ●─●─●─●─●─●─●─●─●─●─●   ← everybody, tiny hops
  fine at 10k. death at 10M.                      ▲ greedy landing
                                      ~O(log N). ~99% as correct. orders of magnitude faster.
```

> 💡 **Mental model for HNSW.** It's a *travel itinerary*. Top layer = intercontinental flights (few airports, huge hops). Middle = domestic. Bottom = walking the last street. You never walk from Dhaka to Berlin — you descend levels until the final hop is tiny.

**Recall — the metric that actually matters.** Recall@k = (relevant items the ANN returned in top-k) ÷ (relevant items exact search would return). You measure it by running a sample of queries both exact (`exact: true`) and approximate, then comparing. If recall is too low, raise `ef`/`m`; if latency is too high, lower them or add quantization + rescoring. Never ship an ANN system without measuring recall on your own data — defaults are a starting point, not an answer.

```
          exact top-10 (ground truth)  ▓▓▓▓▓▓▓▓▓▓
          ANN top-10                   ▓▓▓▓▓▓▓▓░░   8 of 10 matched
                                                    → recall@10 = 0.8

          ef ↑ / m ↑  ──▶  recall ↑ , latency ↑
          ef ↓ / m ↓  ──▶  latency ↓ , recall ↓
          (there is no third option. this is the whole trade.)
```

**Chunking.** RAG quality is decided before the vector DB is ever touched: how you split documents. Fixed-size windows (e.g. 500–1000 tokens with 10–15% overlap) are the simple baseline; structure-aware splitting (by heading, paragraph, sentence, or code block) is better when documents have structure. Overlap preserves context across boundaries. Store enough metadata per chunk (document id, section, page) to reconstruct provenance. Part 3 uses a real chunker.

```
  a 4000-token doc, chunked at ~512 tokens with ~12% overlap:

  ├────── chunk 0 ──────┤
                  ├────── chunk 1 ──────┤
                                  ├────── chunk 2 ──────┤
                  └ overlap ┘  ← the sentence straddling the boundary still lives
                                 whole inside ONE chunk. that's the entire point.

  every chunk carries: { document_id, chunk_index, heading, source_uri, page }
                        └────────── so you can cite it later ──────────┘
```

**Normalization.** For cosine/dot, normalize vectors to unit length so scores are comparable. Many embedding APIs already return normalized vectors; local models via Ollama/GGUF often do not — check and normalize if needed.

**Where Qdrant sits.** Qdrant ships **only HNSW** (no IVF/DiskANN), which keeps the tuning surface small. It adds first-class **payload** (JSON metadata) with **filterable** indexes, **sparse vectors** (BM25/SPLADE-style), **multivectors** (ColBERT late interaction), **quantization**, **multitenancy**, and a unified **Query API**. It's written in Rust, is gRPC-first, and has a maintained C# client.

---

## 2. Qdrant architecture & data model

**The hierarchy:**
- **Collection** — the top-level container. Fixes vector dimensionality and distance metric(s). Analogous to a table.
- **Point** — one record: an `id` (unsigned int or UUID), one or more **vectors**, and a **payload** (arbitrary JSON).
- **Payload** — JSON metadata attached to a point (tenant, category, timestamps, source, text). Filterable and, with an index, fast to filter.
- **Segment** — internal storage/index unit. A collection's data is split across segments that are searched in parallel and periodically optimized/merged.
- **Shard** — a horizontal partition of a collection. Shards distribute data across nodes in a cluster. Each shard can have **replicas** for HA.

```
  COLLECTION  "docmind_chunks"        ← fixes dim + distance (like a table schema)
  │   dim=768 · Cosine
  ├── SHARD 0 ──────────────────┐     ← horizontal partition, lives on a node
  │   ├── replica on node-A     │        (replicas = HA, §10)
  │   └── replica on node-B     │
  │        ├── SEGMENT ─────────┤     ← storage+index unit, searched in parallel,
  │        ├── SEGMENT          │        merged/vacuumed by the optimizer
  │        └── SEGMENT          │
  │             └── POINT ──────┤
  │                  ├ id       1   (or a UUID)
  │                  ├ vectors  { "dense":  [0.05, 0.61, …768],
  │                  │            "sparse": {12: 0.8, 880: 0.5} }   ← named vectors
  │                  └ payload  { tenant_id, title, tags[], views, published_at }
  │                               └─ arbitrary JSON · filterable · indexable (§5)
  └── SHARD 1 …
```

> 💡 **Rosetta stone for SQL brains:**
> `collection` ≈ table · `point` ≈ row · `payload` ≈ the non-vector columns · `payload index` ≈ a B-tree index · `HNSW index` ≈ the thing SQL doesn't have · `segment` ≈ an internal SSTable-ish file · `shard` ≈ a partition.

**Storage.** Vectors and payload live in memory, memory-mapped files (mmap), or on disk, configurable per collection and per component. HNSW graphs are held in RAM by default for speed. Quantized vectors can be pinned to RAM while originals sit on disk — the key memory lever (§9).

**Collection status** signals health: `green` (ready), `yellow` (optimizing — usable but indexing in progress), `grey` (pending optimization, since v1.9), `red` (error). After a bulk load, expect `yellow` until the HNSW index finishes building.

```
   🟢 green   ready, indexed, serving             → chill
   🟡 yellow  optimizing / indexing in progress   → still usable! normal after a bulk load
   ⬜ grey    pending optimization (v1.9+)        → queued, nothing is on fire
   🔴 red     error                               → page someone (alerts in §10)
```

**Named vectors.** A single point can hold multiple vectors under names — e.g. a `dense` semantic vector plus a `sparse` keyword vector, or vectors from different models. This is what makes hybrid search and multi-stage retrieval possible in one collection.

**Ports.** REST on **6333**, gRPC on **6334**, and inter-node p2p on **6335**. The C# client uses gRPC (6334).

```
   6333  REST + the web dashboard  →  http://localhost:6333/dashboard
   6334  gRPC                      →  👈 the C# client talks here
   6335  p2p between cluster nodes →  never expose this publicly
```

---

## 3. Collections

Install the client:

```bash
dotnet add package Qdrant.Client
```

Connect (gRPC, 6334):

```csharp
using Qdrant.Client;
using Qdrant.Client.Grpc;

var client = new QdrantClient("localhost", 6334);
// Cloud/TLS:  new QdrantClient("xyz.cloud.qdrant.io", 6334, https: true, apiKey: "…");
```

**Create a simple collection:**

```csharp
await client.CreateCollectionAsync(
    collectionName: "articles",
    vectorsConfig: new VectorParams { Size = 768, Distance = Distance.Cosine });
```

**Named vectors (dense + sparse for hybrid):**

```csharp
await client.CreateCollectionAsync(
    collectionName: "articles",
    vectorsConfig: new VectorParamsMap
    {
        Map =
        {
            ["dense"] = new VectorParams { Size = 768, Distance = Distance.Cosine },
        }
    },
    sparseVectorsConfig: new SparseVectorConfig(
        ("sparse", new SparseVectorParams())));
```

> 💡 One point, two vectors, two different ways to be found:
>
> ```
>   POINT #42  "Hybrid search with dense + sparse"
>     ├─ "dense"  [0.05, 0.61, …]   → found by MEANING  ("semantic retrieval methods")
>     └─ "sparse" {12:0.8, 880:0.5} → found by WORDS    ("dense + sparse")
>   fuse both at query time (§8.2) and you catch what either leg alone would miss.
> ```

**Useful create-time options:**
- `OnDisk = true` on `VectorParams` — store original vectors as mmap files instead of RAM (big memory saver; pair with quantized-in-RAM, §9).
- `HnswConfig` — override `m`, `EfConstruct` per collection (§5).
- `QuantizationConfig` — enable quantization at creation (§9).
- `uint8`/`float16` datatypes and multivector config for ColBERT-style late interaction.

**Inspect / manage:**

```csharp
var info = await client.GetCollectionInfoAsync("articles");   // status, counts, config
bool exists = await client.CollectionExistsAsync("articles");
var names  = await client.ListCollectionsAsync();
await client.DeleteCollectionAsync("articles");
```

**Update parameters live** (e.g. tighten HNSW or change optimizer thresholds) with `UpdateCollectionAsync` — omit the vector config since it's already set.

**Aliases** give you zero-downtime reindexing: build `articles_v2`, then atomically repoint the alias.

```csharp
await client.CreateAliasAsync(aliasName: "articles", collectionName: "articles_v2");
// or swap atomically:
await client.UpdateAliasesAsync(new[]
{
    new AliasOperations { DeleteAlias = new DeleteAlias { AliasName = "articles" } },
    new AliasOperations { CreateAlias = new CreateAlias { AliasName = "articles", CollectionName = "articles_v2" } },
});
```

> 💡 **Aliases = blue/green deploys for your index.** Changed embedding model? You can't mutate dims in place — you rebuild beside it and flip the pointer.
>
> ```
>   app ──▶ alias "articles" ──▶ articles_v1   (live, serving traffic)
>                                articles_v2   (building… ingesting… 🟡 → 🟢)
>
>   ────────────── atomic swap ──────────────▶
>
>   app ──▶ alias "articles" ──▶ articles_v2   ✅ zero downtime, instant rollback
>                                articles_v1   (keep it a day, then drop it)
> ```

**Multitenancy.** The recommended pattern for many tenants is **one collection + a tenant field in the payload** (indexed as a tenant field, §5), not one-collection-per-tenant. It scales to far more tenants and keeps the HNSW graph tenant-aware. Every query then carries a `tenant_id` filter. Part 3 uses this.

```
  ❌ collection-per-tenant              ✅ one collection + tenant_id in payload
  ┌──────┐┌──────┐┌───────┐            ┌────────────────────────────────┐
  │acme  ││globex││initech│  × 5000     │ point{ tenant_id:"acme",    …} │
  │ HNSW ││ HNSW ││ HNSW  │            │ point{ tenant_id:"globex",  …} │
  └──────┘└──────┘└───────┘            │ point{ tenant_id:"initech", …} │
  5000 graphs, 5000× overhead,         └────────────────────────────────┘
  an ops nightmare, doesn't scale      one tenant-aware graph (IsTenant=true, §5);
                                       every query carries a tenant filter
```

---

## 4. Points — writing and reading data

A point = `id` + `vectors` + `payload`.

```
   ┌─ POINT ───────────────────────────────────────────────────┐
   │ id       1                     ← uint64 or UUID            │
   │ vectors  [0.05, 0.61, … 768]   ← what SEARCH matches on    │
   │ payload  { title, tenant_id,   ← what FILTERS match on,    │
   │            tags[], published,    and what you show the     │
   │            views }               user / feed to the LLM    │
   └───────────────────────────────────────────────────────────┘
```

**Upsert (insert or overwrite):**

```csharp
await client.UpsertAsync("articles", new[]
{
    new PointStruct
    {
        Id = 1,
        Vectors = new float[] { 0.05f, 0.61f, /* … 768 dims */ },
        Payload =
        {
            ["title"]     = "Intro to RAG",
            ["tenant_id"] = "acme",
            ["tags"]      = new[] { "rag", "search" },
            ["published"] = true,
            ["views"]     = 1200,
        }
    }
});
```

IDs are unsigned integers or UUIDs. Payload accepts strings, numbers, bools, arrays, and nested objects.

**Named vectors on a point:**

```csharp
new PointStruct
{
    Id = Guid.NewGuid().ToString(),
    Vectors = new Dictionary<string, Vector>
    {
        ["dense"]  = new float[] { /* … */ },
        ["sparse"] = new (float[] values, uint[] indices)( new[]{0.8f,0.5f}, new uint[]{12,880} ),
    },
    Payload = { ["tenant_id"] = "acme" }
}
```

**Batch upserts** are the throughput path — send hundreds to a few thousand points per call rather than one at a time. Part 3 batches with retries.

```
  ❌ 10,000 calls × 1 point            ✅ 40 calls × 256 points
     ▓ round-trip ░ work                 ▓ round-trip ░░░░░░░░ work
     ▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░           ▓░░░░░░░░▓░░░░░░░░▓░░░░░░░░
     you pay the latency tax 10,000      same data, a fraction of the
     times to move 10,000 rows           wall clock
```

**Retrieve by id / scroll / count:**

```csharp
var pts = await client.RetrieveAsync("articles", ids: new ulong[] { 1, 2, 3 },
                                     withPayload: true, withVectors: false);

// Scroll = paginate through points (optionally filtered), no query vector:
var page = await client.ScrollAsync("articles",
    filter: MatchKeyword("tenant_id", "acme"),
    limit: 100);   // page.NextPageOffset drives the next call

var n = await client.CountAsync("articles", filter: MatchKeyword("tenant_id", "acme"));
```

> 💡 **Retrieve vs Scroll vs Query.** `Retrieve` = "give me these exact ids" (a `WHERE id IN (…)`). `Scroll` = "walk everything matching this filter, page by page" — a cursored `SELECT`, **no vector involved**. `Query` = "who's nearest?" (§7). Use scroll for exports and backfills, never for search.

**Payload operations** (without re-sending vectors):

```csharp
await client.SetPayloadAsync("articles",                 // merge keys
    payload: new Dictionary<string, Value> { ["reviewed"] = true },
    ids: new ulong[] { 1 });

await client.OverwritePayloadAsync("articles", payload, ids: new ulong[] { 1 }); // replace all
await client.DeletePayloadAsync("articles", keys: new[] { "reviewed" }, ids: new ulong[] { 1 });
await client.ClearPayloadAsync("articles", ids: new ulong[] { 1 });
```

```
  Set…        {a:1, b:2}  +  {b:9, c:3}   →  {a:1, b:9, c:3}   merge     🤝
  Overwrite…  {a:1, b:2}  ←  {b:9, c:3}   →  {b:9, c:3}        replace   💣
  Delete…     {a:1, b:2}  −  keys["b"]    →  {a:1}             surgical  🔪
  Clear…      {a:1, b:2}  →  {}                                nuke      ☢️
```

**Delete** by id or by filter:

```csharp
await client.DeleteAsync("articles", ids: new ulong[] { 1, 2 });
await client.DeleteAsync("articles", filter: MatchKeyword("tenant_id", "defunct-tenant"));
```

**Vector ops & schema migration.** Update just a point's vectors with `UpdateVectorsAsync`; drop a vector with `DeleteVectorsAsync`. Since v1.18 you can add or remove a **named vector across the whole collection** (`CreateVectorNameAsync` / `DeleteVectorNameAsync`) — the migration path when you add, say, a sparse vector to an existing dense-only collection.

---

## 5. Indexing — payload, full-text, HNSW, sparse

Two independent index systems: **payload indexes** (make filters fast) and the **vector/HNSW index** (makes similarity search fast). Getting both right is most of production performance.

```
   ┌──────────────── ONE COLLECTION, TWO INDEX SYSTEMS ────────────────┐
   │                                                                   │
   │   PAYLOAD INDEXES              │        VECTOR INDEX (HNSW)       │
   │   "tenant_id = acme"           │        "nearest to [0.05,…]"     │
   │   keyword / int / float / bool │        m · ef_construct · ef     │
   │   geo / datetime / text / uuid │                                  │
   │   ── makes FILTERS fast ──     │        ── makes SEARCH fast ──   │
   │                                │                                  │
   │            └──────── filterable HNSW welds them together ─────┘   │
   │              (so a filtered search stays graph-fast, not a scan)  │
   └───────────────────────────────────────────────────────────────────┘
```

### Payload indexes
Without a payload index, a filter on that field is a linear scan. Create the index on fields you filter by:

```csharp
await client.CreatePayloadIndexAsync("articles", "tenant_id", PayloadSchemaType.Keyword);
await client.CreatePayloadIndexAsync("articles", "views",     PayloadSchemaType.Integer);
await client.CreatePayloadIndexAsync("articles", "published_at", PayloadSchemaType.Datetime);
```

Supported types: `Keyword`, `Integer`, `Float`, `Bool`, `Geo`, `Datetime`, `Text` (full-text), `Uuid`. Integer indexes can be parameterized for `lookup` (equality) and/or `range`. Indexes can be built `on_disk` to save RAM.

```
  no index →  filter = read every point, check the field   ░░░░░░░░░░ 😵 O(N)
  indexed  →  jump straight to the matching set            ▓          😎 O(log N)-ish
```

> **Order matters:** create payload indexes **before** or during ingest when you want *filterable HNSW* (below) to build tenant/filter-aware graph edges. Add them after a big load and you may need the graph to catch up.

**Tenant & principal optimizations:**
- Mark the tenant field with `IsTenant = true` — Qdrant then physically co-locates each tenant's vectors, dramatically speeding tenant-scoped queries.
- Mark a frequently-filtered "hot" field with `IsPrincipal = true` to bias storage/index layout toward it.

```
  IsTenant = false            IsTenant = true
  A B A C B A C A B C  →      A A A A | B B B | C C C
  tenants interleaved         each tenant's vectors physically co-located
  every query touches         a tenant-scoped query touches only its own block
  everyone's data pages
```

### Full-text index
For substring/keyword matching inside text payload:

```csharp
await client.CreatePayloadIndexAsync("articles", "body", PayloadSchemaType.Text);
```

Configurable tokenizers: `word`, `whitespace`, `prefix`, `multilingual`; plus `lowercase`, `ascii_folding`, Snowball `stemmer`, `stopwords`, `min/max_token_len`, and **phrase matching**. This powers the `Text`/`MatchText`/`MatchPhrase` filters in §6 — distinct from *sparse-vector* keyword search, which is a scored retrieval channel (§8).

> ⚠️ **Don't mix these two up:**
> ```
>   full-text INDEX (§5/§6)  →  a FILTER. yes/no. "does this doc contain the term?"
>   sparse VECTOR   (§8.7)   →  a RETRIEVER. scored. "how well does it match, ranked?"
>   same words involved, completely different jobs.
> ```

### Vector index — HNSW tuning
The three knobs:

| Param | What it does | Higher = |
|---|---|---|
| `m` | edges per node in the graph | better recall, more memory, slower build |
| `ef_construct` | candidate list size at build time | better graph quality, slower build |
| `ef` (search) | candidate list size at query time | better recall, slower query |

```
        m = how many friends each node keeps          ef = how wide you look while walking
        ┌──●──┐        ┌──●──┐                        ef=16   🔦  narrow beam, fast, misses things
     ●──┤  ●  ├──●  vs ●══╪══●══╪══●                  ef=128  💡  wide beam, slower, finds more
        └──●──┘        └──●──┘                        ┌──────────────────────────────────┐
        m=4 sparse     m=32 dense                     │ ef is PER QUERY — tune it live.  │
        light, lossy   heavy, accurate                │ m is baked into the graph — a    │
                                                      │ change means a rebuild.          │
                                                      └──────────────────────────────────┘
```

Set collection defaults at create time; override `ef` per query (§7). Typical starting points: `m=16`, `ef_construct=100`, query `ef=64–128`. Tune against measured recall.

**Filterable HNSW.** Qdrant weaves payload constraints into the graph so a filtered search stays fast instead of degrading to a scan — provided the payload index exists. `full_scan_threshold` sets the point count below which Qdrant just brute-forces (faster for tiny/very-selective sets).

**ACORN** (search param, §7) improves filtered search when filters are *highly selective*; enable it with a `max_selectivity` guard (~0.4) so it only kicks in when worthwhile.

**Rebuilding HNSW.** Changing HNSW params triggers a rebuild. A known trick to force a clean rebuild is bumping `ef_construct` by +1. You can also disable HNSW entirely (`m=0`) for a vector used only for rescoring (§8) — it frees memory since rescoring doesn't traverse the graph.

### Sparse vector index
Sparse vectors (high-dimensional, mostly-zero; BM25/SPLADE-style) have their own index. It can live `on_disk`, and you can apply an **IDF modifier** so Qdrant weights terms by inverse document frequency — important for BM25-like behaviour. Sparse + dense fused together is the canonical hybrid recipe (§8).

```
   DENSE  (768 numbers, all non-zero)      SPARSE (30,000 dims, ~20 non-zero)
   [0.05, 0.61, -0.12, 0.44, … ]           { 12: 0.8,  880: 0.5,  4021: 1.2 }
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓             ░░░▓░░░░░░░░░░▓░░░░░░░░░░░░░▓░░░
    every dim carries a bit of meaning      only the terms the doc actually uses
    great at paraphrase                     great at "error code E4012"
```

---

## 6. Filtering

Filters combine with vector search or stand alone (scroll/count/delete). Qdrant has three boolean clauses; the C# client maps them to operators:

- **must** → `&` (AND)
- **should** → `|` (OR)
- **must_not** → `!` (NOT)

```
      must / &       should / |        must_not / !
      ┌───┐          ┌───┬───┐         ┌─────────┐
      │ ▓ │ ALL      │ ▓ │ ▓ │ ANY     │ ░░░░░░░ │ NONE
      └───┘ of these └───┴───┘ of these └─────────┘ of these
```

Import the helpers:

```csharp
using static Qdrant.Client.Grpc.Conditions;
```

**Building blocks:**

```csharp
// equality / any-of / except
MatchKeyword("tenant_id", "acme");
Match("status", new[] { "active", "trial" });          // any of
// range
Range("views", new Qdrant.Client.Grpc.Range { Gte = 100, Lt = 10000 });
// datetime range, geo (box / radius / polygon), values_count, is_empty / is_null
HasId(new PointId[] { 1, 2 });
// full-text (needs a Text index)
MatchText("body", "vector database");
```

**Composition** with operators:

```csharp
var filter =
      MatchKeyword("tenant_id", "acme")
    & (MatchKeyword("lang", "en") | MatchKeyword("lang", "de"))
    & !MatchKeyword("status", "archived")
    & Range("views", new() { Gte = 50 });
```

> 💡 Read that filter as a funnel — it's just a `WHERE` clause with better ergonomics:
>
> ```
>   all points          ████████████████████████████████
>   & tenant = acme     ██████████
>   & lang ∈ {en,de}    ███████
>   & ! archived        ██████
>   & views ≥ 50        ███      ← the HNSW search only walks this survivors' set
> ```

Also available: `MatchAny`/`MatchExcept`, `IsEmpty`, `IsNull`, `HasVector` (points that have a given named vector), nested-object conditions (`Nested`), and geo filters (`GeoBoundingBox`, `GeoRadius`, `GeoPolygon`). Everything you filter on frequently should have a payload index (§5).

---

## 7. Search & the Query API

Since v1.10 everything routes through one **Query API** (`QueryAsync`) — plain search, search-by-id, recommend, discover, hybrid, and multi-stage all share it.

```
                        ┌──────────────────┐
     query: float[]  ──▶│                  │──▶ plain nearest-neighbour
     query: 42UL     ──▶│                  │──▶ "more like point 42"
     query: Rrf()    ──▶│   QueryAsync()   │──▶ hybrid fusion (§8.2)
     query: Recommend──▶│   ONE endpoint   │──▶ positives / negatives (§8.4)
     query: Discover ──▶│   TO RULE THEM   │──▶ context-constrained (§8.5)
     query: Formula  ──▶│      ALL         │──▶ business-logic rescoring (§8.3)
     + prefetch[]    ──▶└──────────────────┘    multi-stage pipelines (§8.1)
```

**Nearest-neighbour search:**

```csharp
var hits = await client.QueryAsync(
    collectionName: "articles",
    query: new float[] { 0.05f, 0.61f, /* … */ },
    limit: 10,
    payloadSelector: true);                 // return payload

foreach (var h in hits)
    Console.WriteLine($"{h.Id}  score={h.Score}  {h.Payload["title"].StringValue}");
```

**Search by an existing point's id** (find similar to a known item):

```csharp
var similar = await client.QueryAsync("articles", query: 42UL, limit: 10);
```

**Search parameters** (`SearchParams`):
- `HnswEf` — per-query `ef`; raise for recall, lower for speed.
- `Exact = true` — brute-force exact search; use it to **measure recall** of the ANN path.
- `IndexedOnly = true` — skip not-yet-indexed segments (predictable latency during heavy ingest).
- `Quantization` — control rescore/oversampling/ignore (§9).
- `Acorn` — enable ACORN for highly selective filters.

```csharp
var hits = await client.QueryAsync("articles",
    query: qvec, limit: 10,
    filter: MatchKeyword("tenant_id", "acme"),
    searchParams: new SearchParams { HnswEf = 128 },
    scoreThreshold: 0.6f,                        // drop weak matches
    payloadSelector: new WithPayloadSelector      // include/exclude specific keys
    {
        Include = new PayloadIncludeSelector { Fields = { "title", "url" } }
    });
```

> 🎯 **`Exact = true` is your measuring tape, not a production mode.** Run a sample of queries both ways, diff the top-k, and you have a real recall number instead of a feeling:
> ```
>   for each sampled query:
>       truth = QueryAsync(… SearchParams{ Exact = true })     ← ground truth, slow
>       ann   = QueryAsync(… SearchParams{ HnswEf = 128 })     ← what prod actually does
>       recall += |truth ∩ ann| / k
>   ship when the number, not the vibe, is good.
> ```

**Named vector** — specify which vector to search: pass `usingVector: "dense"`.

**Batching.** `QueryBatchAsync` runs many queries in one round-trip — much cheaper than N calls.

**Pagination.** Use `offset` + `limit`. Because ANN ordering can shift as data changes, for stable deep pagination either (a) keep `ef` high, (b) filter on an indexed monotonic field, or (c) prefer id-based cursoring via scroll for exports.

**Grouping** — collapse multiple points that share a field (e.g. many chunks of one document) into one result per group:

```csharp
var groups = await client.QueryGroupsAsync(
    collectionName: "chunks",
    query: qvec,
    groupBy: "document_id",
    limit: 5,          // number of groups
    groupSize: 2);     // best points per group
```

> 💡 **Why grouping exists.** Without it, one very relevant document eats your whole top-k:
> ```
>   ungrouped top-5          grouped by document_id (limit=5, groupSize=2)
>   1. doc-A chunk 3   😑    doc-A ─┬ chunk 3
>   2. doc-A chunk 4   😑          └ chunk 4
>   3. doc-A chunk 5   😑    doc-B ─┬ chunk 1
>   4. doc-A chunk 2   😑          └ chunk 7
>   5. doc-B chunk 1         doc-C ─┬ chunk 2  ← 5 documents of context,
>   ↑ one doc, five slots           └ chunk 9    not one doc five times
> ```

**Lookup-in-groups** enriches each group with data from another collection (e.g. fetch the parent document record) via a `withLookup` parameter — the "chunks point at documents" pattern used in Part 3.

**Random sampling** and a **query-planning** step (Qdrant decides plain-vs-filtered-vs-exact automatically) round out the API.

---

## 8. Advanced retrieval — the modern RAG patterns

This is where most of the value lives for a real RAG system, and where the v1 of this guide was thin. Everything below is the **current** C# API.

```
        ┌──────────────── the §8 upgrade ladder ────────────────┐
        │ 8.1 prefetch      cheap first, accurate second        │
        │ 8.2 hybrid+fusion dense + sparse, RRF / DBSF   ⭐ big  │
        │ 8.3 formula       recency / popularity / geo          │
        │ 8.4 recommend     positives & negatives               │
        │ 8.5 discover      context pairs bend the space        │
        │ 8.6 matrix        pairwise distances → clustering     │
        │ 8.7 sparse/BM25   exact terms dense vectors fumble    │
        │ 8.8 rerank        cross-encoder precision      ⭐ big  │
        │ 8.9 top-k & MMR   how wide, how narrow, how diverse   │
        │ 8.10 query-side   rewrite / multi-query / HyDE        │
        │ 8.11 the whole pipeline, assembled                    │
        └───────────────────────────────────────────────────────┘
```

### 8.1 Prefetch & multi-stage queries
A query can carry one or more **`prefetch`** sub-queries. Qdrant runs the prefetch(es) first, then applies the main query over their results. Prefetches nest. This enables *cheap-then-accurate* pipelines:
- quantized vector first → full-precision rescored,
- short **Matryoshka (MRL)** byte vector first → full vector,
- dense vector first → **ColBERT multivector** re-scoring.

```
   ┌── PREFETCH ─────────────┐        ┌── MAIN QUERY ──────────┐
   │ cheap vector, wide net  │  ────▶ │ expensive, precise     │ ──▶ top 10
   │ mrl_byte · limit 1000   │        │ full vector · rescore  │
   └─────────────────────────┘        └────────────────────────┘
       1000 candidates, fast              10 winners, accurate

   💡 like a job screen: cheap CV filter → expensive interview.
      you don't interview 1000 people, and you don't hire off a CV.
```

**Re-score a cheap candidate set with the full vector:**

```csharp
await client.QueryAsync(
    collectionName: "chunks",
    prefetch: new List<PrefetchQuery>
    {
        new() { Query = new float[] { 1, 23, 45, 67 }, Using = "mrl_byte", Limit = 1000 }
    },
    query: new float[] { 0.01f, 0.299f, 0.45f, 0.67f /* full */ },
    usingVector: "full",
    limit: 10);
```

**Re-score with a ColBERT multivector** (`float[][]`):

```csharp
await client.QueryAsync("chunks",
    prefetch: new List<PrefetchQuery>
    {
        new() { Query = new float[] { 0.01f, 0.45f, 0.67f }, Limit = 100 }
    },
    query: new float[][] { new[]{0.1f,0.2f}, new[]{0.2f,0.1f}, new[]{0.8f,0.9f} },
    usingVector: "colbert",
    limit: 10);
```

Tip: disable HNSW (`m=0`) on a vector used only for rescoring — rescoring doesn't use the graph, so you save the memory.

### 8.2 Hybrid search & fusion (dense + sparse)
Retrieve with **both** a sparse (keyword) and dense (semantic) vector, then fuse. This is the single highest-leverage retrieval upgrade for text RAG.

```
                    query: "error code E4012 in the payments module"
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
        SPARSE leg (words)                            DENSE leg (meaning)
        nails "E4012" exactly                         gets "payment failure docs"
        top-20 ──────────┐                            ┌────────── top-20
                         ▼                            ▼
                    ┌──────────────────────────────────────┐
                    │  FUSION   RRF (by rank) / DBSF (by   │
                    │           normalized score)          │
                    └──────────────────┬───────────────────┘
                                       ▼
                             one ranked list. 🏆
                     catches what either leg alone would drop.
```

**Reciprocal Rank Fusion (RRF)** — rank-based, robust, the safe default:

```csharp
await client.QueryAsync(
    collectionName: "chunks",
    prefetch: new List<PrefetchQuery>
    {
        new() { Query = new (float, uint)[] { (0.22f, 1), (0.8f, 42) }, Using = "sparse", Limit = 20 },
        new() { Query = new float[] { 0.01f, 0.45f, 0.67f },            Using = "dense",  Limit = 20 },
    },
    query: new Rrf());                       // fuse by rank
```

> 📐 **RRF on the back of a napkin** (`score = Σ 1/(k + rank)`, k=60):
>
> ```
>   doc    dense rank   sparse rank        RRF score                 final
>   ────────────────────────────────────────────────────────────────────────
>   D1        1            9         1/61 + 1/69  = 0.0309            🥇 1
>   D2        2            3         1/62 + 1/63  = 0.0320  ← wait…   🥇 1  (highest!)
>   D3       —            1               1/61    = 0.0164            🥉 3
>   ────────────────────────────────────────────────────────────────────────
>   the doc that BOTH retrievers liked wins over the doc either one loved alone.
>   ranks only → no score-scale drama. that's why it's the default.
> ```

RRF options: set the constant `k` (`new Rrf { K = 60 }`, v1.16) and **weighted RRF** (`new Rrf { Weights = { 3.0f, 1.0f } }`, v1.17) to favour a stronger retriever. Tune weights on a held-out eval split, not by eye; leave them equal if you have no eval set.

**Distribution-Based Score Fusion (DBSF)** — normalizes each retriever's score distribution (3-sigma) then sums; use when you trust raw score magnitudes:

```csharp
await client.QueryAsync("chunks",
    prefetch: new List<PrefetchQuery>
    {
        new() { Query = new (float, uint)[] { (0.22f, 1), (0.8f, 42) }, Using = "sparse", Limit = 20 },
        new() { Query = new float[] { 0.01f, 0.45f, 0.67f },            Using = "dense",  Limit = 20 },
    },
    query: Fusion.Dbsf);
```

**Choosing:** eval set available → tuned **weighted RRF**; trust raw scores, no eval set → **DBSF**; neither → **RRF** (default). A naive alpha-weighted linear blend of dense+sparse is unreliable because the two score scales differ per query — RRF (ranks) and DBSF (normalized) both avoid that trap.

```
   have an eval set? ──yes──▶ weighted RRF, weights tuned on held-out data
          │
          no
          ▼
   trust raw score magnitudes? ──yes──▶ DBSF
          │
          no
          ▼
   🎯 plain RRF. (naive alpha-blending of raw scores = the trap. don't.)
```

> ⚠️ **Distributed gotcha:** in a multi-shard collection, fusion only fuses *across shards* when it is the **main** query (top-level `query` with the retrievers as its prefetches). Put fusion inside a prefetch and each shard fuses locally — a per-shard ranking, not global.
> ```
>   ✅ main query = Rrf()        shard0 ─┐
>                                shard1 ─┼─▶ fuse globally ──▶ one true ranking
>                                shard2 ─┘
>   ❌ Rrf() inside a prefetch   shard0 ─▶ fuse ─┐
>                                shard1 ─▶ fuse ─┼─▶ merge of 3 local rankings 🙃
>                                shard2 ─▶ fuse ─┘
> ```

### 8.3 Custom scoring — formula queries (v1.14)
Layer business logic (recency, popularity, geo) on top of a fused result. Typical pattern: fuse with RRF in a prefetch, then wrap it in a **formula** that adds a decay term:

```csharp
await client.QueryAsync(
    collectionName: "chunks",
    prefetch:
    [
        new PrefetchQuery
        {
            Prefetch =
            {
                new PrefetchQuery { Query = new (float, uint)[]{ (0.22f,1),(0.8f,42) }, Using="sparse", Limit=100 },
                new PrefetchQuery { Query = new float[]{ 0.01f,0.45f,0.67f },            Using="dense",  Limit=100 },
            },
            Query = new Rrf(), Limit = 100
        },
    ],
    query: new Formula
    {
        Expression = new SumExpression
        {
            Sum =
            {
                "$score",                        // the fused RRF score
                new MultExpression
                {
                    Mult =
                    {
                        0.1f,                    // cap decay's contribution
                        Expression.FromExpDecay(new()
                        {
                            X      = Expression.FromDateTimeKey("published_at"),
                            Target = Expression.FromDateTime("2026-07-01T00:00:00Z"),
                            Scale  = 86400 * 180, // 180 days, in seconds
                            Midpoint = 0.5f
                        })
                    }
                }
            }
        }
    },
    limit: 10);
```

```
   what the decay term actually does to "freshness":
   1.0 ┤●
       │ ●●
   0.5 ┤   ●●●        ← midpoint: 180 days old = half credit
       │      ●●●●●
   0.0 ┤           ●●●●●●●●●●●
       └──────┬──────┬──────┬──────▶  age
            today  180d    1y
   and it's multiplied by 0.1 so it NUDGES the ranking instead of hijacking it.
```

Calibrate the decay coefficient against your fused-score scale: RRF scores are small sums of `1/(k+rank)`, while a decay returns `[0,1]`, so an un-weighted decay term would swamp the ranking. A formula can't also be a fusion in the same (distributed) main query — for a formula rescore over fused results across shards, use a single shard.

### 8.4 Recommendation API
Search by **examples** (point ids or raw vectors), positive and negative:

```csharp
await client.QueryAsync(
    collectionName: "articles",
    query: new RecommendInput { Positive = { 100, 231 }, Negative = { 718 } },
    filter: MatchKeyword("tenant_id", "acme"),
    limit: 10);
```

```
        "more like THESE 👍 , less like THAT 👎"

           👍100   👍231                   results land in here ✨
              ●      ●                     ┌─────────┐
               ╲    ╱                      │ ● ● ●   │
                ╲  ╱                       │  ● ●    │
                 ▓▓  ◀── pulled toward ────┘         │
                                                     │
                 ░░  ◀── pushed away from ───────────┘
                  ●
                👎718
```

Strategies:
- **`average_vector`** (default) — averages positives/negatives into one vector; fast, search-speed parity.
- **`best_score`** (v1.6) — scores each candidate against every example; more embedding-agnostic but slower with more examples. Raise `ef` (>32) to recover accuracy.
- **`sum_scores`** — sums per-example scores; good for relevance-feedback loops.

`best_score` and `sum_scores` also accept **negative-only** input — a clean way to find *outliers* / most-dissimilar items for data cleaning. Use `usingVector` for named vectors and `lookupFrom` to pull example vectors from *another* collection (item↔user recommendation). `QueryBatchAsync` batches recommendations.

> 💡 `average_vector` blurs your examples into one blob — fine when they're similar, bad when they're "jazz" and "death metal" (you get… elevator music). `best_score` keeps them distinct.

### 8.5 Discovery & Context search (v1.7)
Constrain the *vector space itself* with positive-negative **context pairs**:
- **Discovery search** — a `target` plus context pairs: return points near the target but biased toward positive zones. Great for multimodal / constrained search. Raise `ef` (>64) since the space is hard-constrained.
- **Context search** — pairs only, no target: return a *diverse* set living in the least-negative zone.

```
   DISCOVERY = target + guardrails        CONTEXT = guardrails only
        👍 ▒▒▒▒▒▒│░░░░░░ 👎                   👍 ▒▒▒▒▒│░░░░ 👎
        👍 ▒▒▒🎯▒│░░░░░░ 👎                   👍 ▒▒▒▒▒│░░░░ 👎
              ▲  └ each pair splits                ▲
        "near the target, but                "anywhere in the least-negative
         only on the 👍 side of                zone" → a diverse spread,
         every dividing line"                  no single bullseye
```

```csharp
// Discovery
await client.QueryAsync("articles",
    query: new DiscoverInput
    {
        Target  = new float[] { 0.2f, 0.1f, 0.9f, 0.7f },
        Context = new ContextInput
        {
            Pairs =
            {
                new ContextInputPair { Positive = 100, Negative = 718 },
                new ContextInputPair { Positive = 200, Negative = 300 },
            }
        }
    },
    limit: 10);

// Context-only
await client.QueryAsync("articles",
    query: new ContextInput
    {
        Pairs =
        {
            new ContextInputPair { Positive = 100, Negative = 718 },
            new ContextInputPair { Positive = 200, Negative = 300 },
        }
    },
    limit: 10);
```

### 8.6 Distance Matrix (v1.12)
Sample the collection and compute pairwise distances — the input for clustering, graph visualization, or dimensionality reduction:

```csharp
await client.SearchMatrixPairsAsync("articles",           // list of {a, b, score}
    filter: MatchKeyword("color", "red"), sample: 100, limit: 10);

await client.SearchMatrixOffsetsAsync("articles",          // CSR-style arrays for scientific tooling
    filter: MatchKeyword("color", "red"), sample: 100, limit: 10);
```

```
        Pairs  → [{a:1,b:7,score:.91}, {a:1,b:3,score:.88}, …]   easy to eyeball
        Offsets→ CSR arrays  (row_ptr, col_idx, values)          feed to scipy/UMAP
        use it for: clustering your corpus, spotting duplicate clusters,
                    drawing the "map" of your own embedding space 🗺️
```

### 8.7 Sparse retrieval & BM25 (lexical matching)

Dense vectors capture *meaning* but blur *exact terms*. They struggle with rare tokens, product codes, acronyms, names, code identifiers, and any vocabulary the embedding model didn't see much in training. **Lexical (sparse) retrieval** is the complement: it scores on literal term overlap, so "error code E4012" or "sec. 12(3)(b)" match precisely. Serious RAG uses both and fuses them (§8.2).

```
   query: "error code E4012"

   DENSE only  🤖 "ah yes, an error code. here are 10 docs about errors."   ❌
   SPARSE      🔎 "E4012. this exact token. document #883, line 12."        ✅

   query: "why does my bread not rise"

   DENSE       🤖 "fermentation, proofing, yeast activity…"                 ✅
   SPARSE only 🔎 "documents containing 'bread' and 'rise'… literally"      ❌

   → this is why you run BOTH and fuse (§8.2). they fail in opposite directions.
```

**A sparse vector** represents a document as `{token_id: weight}` — a mostly-zero vector over the vocabulary. The interesting question is how you compute the weights. The classic answer is **BM25**.

**BM25, explained.** BM25 ("Best Match 25") is the workhorse ranking function behind Lucene/Elasticsearch. It refines TF-IDF with two corrections that matter a lot in practice:

- **Term-frequency saturation.** In raw TF, a word appearing 50 times looks 50× more relevant than appearing once. That's wrong — relevance flattens out. BM25 saturates TF through a parameter **`k1`** (typically 1.2–2.0): the 2nd occurrence adds a lot, the 20th almost nothing.
- **Document-length normalization.** Long documents accumulate term matches just by being long. BM25 divides by a length factor tuned by **`b`** (typically 0.75, where 0 = no normalization, 1 = full), so a short, on-topic doc isn't buried under a long, rambling one.
- **IDF weighting.** Rare terms are more discriminative than common ones, so each term is weighted by **inverse document frequency** — "quantization" counts for far more than "the".

```
   ① TF SATURATION (k1)                    ② LENGTH NORM (b)
   score                                   a 50-word note that says "E4012" once
     │        raw TF ╱                      beats a 5000-word novel that says it
     │              ╱                       twice — because BM25 divides by |D|/avgdl
     │     ▁▂▃▄▅▆▆▆▆▆ BM25 (saturates)
     │   ▂▄                                ③ IDF
     │ ▄                                   "the"          ▏      ~nothing
     └──────────────────────▶ term count   "quantization" ████████ a lot
        1  2  3 … 20 … 50                   rare = discriminative = worth points
     the 2nd mention: big. the 20th: meh.
```

The scoring intuition (per query term, summed):

```
score(D, q) = Σ  IDF(term) ·           tf · (k1 + 1)
              term            ─────────────────────────────────
                              tf + k1 · (1 − b + b · |D|/avgdl)
```

You don't implement this yourself in Qdrant — you either feed it BM25-weighted sparse vectors, or let Qdrant handle the IDF part. **Three ways to get BM25-style sparse retrieval in Qdrant:**

1. **IDF modifier on a sparse vector** — set `Modifier.Idf` on the sparse vector config (as DocMind does in §3.2). Qdrant then applies the IDF component **at query time** using its own corpus statistics, so your stored weights only need to carry term frequencies. This is the built-in, no-extra-model path and is confirmed in the collection/indexing docs.
2. **A sparse embedding model** — generate the sparse vectors client-side with a BM25 encoder (e.g. FastEmbed's `Qdrant/bm25`) and upsert them as the `sparse` named vector. *(FastEmbed model names are ecosystem-specific — verify the current model id.)*
3. **A learned sparse model** — **SPLADE** (neural term expansion: adds related terms the document didn't literally contain, improving recall) or Qdrant's **miniCOIL** (a compact learned sparse model that keeps BM25's exact-match strength while adding light semantics). Heavier than BM25 but often higher recall. *(Verify availability/version.)*

```
   effort ▲
          │  ③ SPLADE / miniCOIL   learned, term-expanding, best recall, heaviest
          │  ② client-side BM25    you own the encoder, full control
          │  ① Modifier.Idf        🎯 zero extra models, built in — start here
          └──────────────────────────────────────────────▶ recall
```

Server-side inference (Qdrant Cloud Inference / FastEmbed integration) can also produce these from raw text so you don't ship an encoder in your app. Whichever you choose, the sparse leg plugs straight into the RRF/DBSF hybrid query in §8.2.

**When sparse earns its keep:** keyword-heavy domains (legal, medical, code, finance), exact-match requirements, out-of-domain vocabulary, and as a cheap recall booster on top of dense. When it doesn't: purely conceptual/paraphrase queries where the user's words never match the document's.

### 8.8 Reranking — three kinds, and why cross-encoders win on precision

Retrieval and reranking optimize for different things. **Retrieval** casts a wide net for **recall** — get all the plausibly-relevant passages into a candidate set of, say, top-50, fast. **Reranking** reorders that set for **precision** — put the genuinely best few at the very top before they go to the LLM. This two-stage *retrieve-then-rerank* shape is the standard high-quality RAG pipeline, because the LLM only sees a handful of passages and their **order matters** (models attend more to the top).

```
   🕸️ RECALL stage                        🎯 PRECISION stage
   "don't lose the right answer"          "put it at position #1"
   ┌───────────────────────┐              ┌─────────────┐
   │ ░░░░░░░░░░░░░░░░░░░░░ │  top-50      │ ▓▓▓▓▓ │      │  top-6
   │ ░░░░░▓░░░░░░▓░░░░░░░░ │ ──────────▶  │ ▓▓▓   │      │ ──────▶ LLM
   │ ░░░░░░░░▓░░░░░░░░░░░░ │  rerank      │ ▓     │      │
   └───────────────────────┘              └─────────────┘
    fast, cheap, wide                      slow, smart, narrow
```

The key distinction is **bi-encoder vs cross-encoder**:

- A **bi-encoder** (what produces your embeddings) encodes the query and each document **separately** into vectors, then compares with cosine/dot. Because documents are encoded independently of the query, you can precompute and index them — that's what makes Qdrant fast. The cost is accuracy: query and document never "see" each other.
- A **cross-encoder** feeds the query and a passage **together** through a transformer, which lets every query token attend to every document token, and outputs a single relevance score. This joint attention is dramatically more accurate at judging relevance — but it **can't be precomputed or indexed** (the score depends on the specific query), so you can only afford to run it on a **small candidate set**: the top-N from retrieval.

```
   BI-ENCODER (indexable, fast)              CROSS-ENCODER (accurate, unindexable)
   ┌───────┐        ┌──────────┐             ┌────────────────────────────┐
   │ query │──▶[v]  │ document │──▶[v]       │  [query] [SEP] [passage]   │
   └───────┘   │    └──────────┘   │         │      ▼ joint attention ▼   │
               └── cos/dot ────────┘         │   every query token sees   │
   docs embedded ONCE, ahead of time         │   every document token     │
   → precompute → index → 1000s/ms           └──────────┬─────────────────┘
   ⚠️ the two never actually "meet"                     ▼  one relevance score
                                             must run per (query,passage) pair
                                             → N forward passes, at QUERY time
                                             → only affordable on top-N (≤50–100)
```

Qdrant offers **three reranking mechanisms**, in increasing cost and accuracy:

1. **Fusion reranking (RRF / DBSF)** — merges the rankings of multiple retrievers (dense + sparse). No model, negligible cost, runs inside Qdrant. Covered in §8.2. Always worth doing for hybrid.
2. **Late-interaction reranking (ColBERT / ColPali, MaxSim)** — stores a *matrix* of token-level vectors per passage and scores with MaxSim (each query token matched to its best document token). Token-level matching is far more precise than a single pooled vector, and it runs **inside Qdrant** as a multivector rescoring prefetch (§8.1). A middle ground: more accurate than bi-encoder, cheaper than a cross-encoder, no external service.
3. **Cross-encoder reranking** — the classic "add a reranker" step. Run an external cross-encoder over the (query, passage) pairs from your top-N and reorder. Highest accuracy, highest cost. Common models: `bge-reranker-v2-m3`, `mxbai-rerank-large`, Jina reranker, Cohere Rerank, Voyage rerank. Because latency scales with N (one forward pass per passage), keep **N ≤ ~50–100** and rerank down to your final k (~4–8).

```
   ┌────────────────────┬──────────┬───────────┬──────────────┬─────────────┐
   │ mechanism          │ accuracy │ cost      │ runs where   │ needs       │
   ├────────────────────┼──────────┼───────────┼──────────────┼─────────────┤
   │ 1 fusion RRF/DBSF  │ ★★☆☆☆    │ ~free     │ in Qdrant    │ nothing     │
   │ 2 ColBERT MaxSim   │ ★★★★☆    │ moderate  │ in Qdrant    │ multivector │
   │ 3 cross-encoder    │ ★★★★★    │ high      │ YOUR app     │ a model     │
   └────────────────────┴──────────┴───────────┴──────────────┴─────────────┘
```

**The reference two-stage pipeline:**

```
query ──▶ hybrid retrieve (dense + sparse, RRF) top-50
      ──▶ cross-encoder rerank (query,passage) pairs
      ──▶ keep top-6 ──▶ LLM
```

**Cross-encoder in C#.** Qdrant doesn't run cross-encoders (it's a vector DB, not a model server), so this is a stage in *your* pipeline. Two realistic .NET options: call a hosted rerank API, or run a local cross-encoder via ONNX Runtime. Abstract it:

```csharp
public interface ICrossEncoderReranker
{
    // returns the input passages reordered best-first, trimmed to topK
    Task<IReadOnlyList<Passage>> RerankAsync(
        string query, IReadOnlyList<Passage> candidates, int topK, CancellationToken ct = default);
}
```

**Option A — hosted rerank API** (Cohere/Jina/Voyage look similar; this is the shape):

```csharp
public sealed class HostedReranker(HttpClient http, IOptions<RerankOptions> opt) : ICrossEncoderReranker
{
    public async Task<IReadOnlyList<Passage>> RerankAsync(
        string query, IReadOnlyList<Passage> candidates, int topK, CancellationToken ct = default)
    {
        if (candidates.Count <= 1) return candidates;

        var req = new
        {
            model = opt.Value.Model,                 // e.g. "rerank-english-v3.0"
            query,
            documents = candidates.Select(c => c.Text).ToArray(),
            top_n = topK
        };
        using var resp = await http.PostAsJsonAsync(opt.Value.Endpoint, req, ct);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<RerankResponse>(cancellationToken: ct);

        // API returns indices into `documents` with relevance scores, best-first
        return body!.Results
            .OrderByDescending(r => r.RelevanceScore)
            .Take(topK)
            .Select(r => candidates[r.Index] with { Score = (float)r.RelevanceScore })
            .ToList();
    }

    private sealed record RerankResponse(RerankResult[] Results);
    private sealed record RerankResult(int Index, double RelevanceScore);
}
```

**Option B — local cross-encoder via ONNX Runtime** (no network, no per-call cost; export e.g. `bge-reranker-v2-m3` to ONNX). Sketch: tokenize each `[query, passage]` pair with a matching tokenizer, run `InferenceSession.Run`, read the logit as the relevance score, sort descending, take topK. Use `Microsoft.ML.OnnxRuntime` + a tokenizer library. This is the .NET-native, offline path and keeps data in-house — relevant for education/enterprise data-residency constraints.

```
   A) hosted API              B) local ONNX
   ☁️  no GPU to babysit      🖥️  no network hop
   💸 per-call cost           🆓 no per-call cost
   🌐 data leaves your box    🔒 data stays in-house (data residency ✅)
   ⏱️ ~100–300ms / batch      ⏱️ ~10s of ms, if you sized the box right
```

**Cross-encoder vs late-interaction — which to reach for?** If you want maximum precision and can tolerate an external model/latency, cross-encoder. If you want most of the gain with no external dependency and lower latency, ColBERT late-interaction inside Qdrant. Many production systems do fusion + cross-encoder; some do fusion + ColBERT and skip the external hop. Measure both on your eval set.

### 8.9 Top-k, oversampling & diversity (MMR)

**Two different k's.** There's the **retrieval k** (how wide you cast the net — the candidate set, N) and the **final k** (how many passages actually reach the LLM). Retrieve wide, keep narrow. Getting this wrong is a common, quiet cause of bad RAG:

- **Final k too small** → you miss context that was actually retrievable; the LLM can't answer from what it never saw.
- **Final k too large** → you dilute the good passages with marginal ones, blow the context-token budget, and trigger "lost in the middle" (LLMs under-attend to the middle of long contexts). More context is not more accuracy past a point.

```
   answer
   quality ▲
           │        ▄▄▄▄▄▄▄
           │      ▄▄       ▀▀▄▄▄▄
           │   ▄▄▀                ▀▀▀▄▄▄▄▄   ← "lost in the middle" + noise
           │ ▄▀                             
           └─┬────────┬─────────┬──────────────▶ final k
             1      4–8        20            50
          too few:  🎯 sweet    diluted, expensive, and the model
          missing   spot        starts ignoring the middle of your context
          context
```

A good default: retrieve N≈30–50 (hybrid), rerank, feed final k≈4–8. Tune k against answer quality, not vibes.

**Oversampling** is the same idea applied to quantization: pre-select more candidates on the cheap quantized index, then rescore to the top-k on full precision (§9). Retrieval k and oversampling stack.

**MMR (Maximal Marginal Relevance) — diversity.** Top-k by pure relevance often returns near-duplicates (three chunks that say the same thing). MMR reranks to balance relevance against novelty, iteratively picking the next passage that is relevant to the query **but** dissimilar to what's already selected:

```
MMR = argmax over unpicked d of:   λ · sim(d, query) − (1 − λ) · max sim(d, already_picked)
```

```
   pure relevance top-5              MMR (λ≈0.6) top-5
   ┌────────────────────┐            ┌────────────────────┐
   │ "pricing is $10"   │            │ "pricing is $10"   │
   │ "price: ten bucks" │ 🔁 dupe    │ "billing cycle"    │ ✨ new angle
   │ "costs $10/mo"     │ 🔁 dupe    │ "refund policy"    │ ✨ new angle
   │ "the price is $10" │ 🔁 dupe    │ "enterprise tier"  │ ✨ new angle
   │ "$10 monthly"      │ 🔁 dupe    │ "price: ten bucks" │
   └────────────────────┘            └────────────────────┘
   the LLM learned one fact          the LLM can actually answer follow-ups
   λ=1 → pure relevance · λ≈0.5–0.7 → healthy diversity · λ=0 → pure novelty chaos
```

`λ` (0–1) dials the trade-off: λ=1 is pure relevance, λ≈0.5–0.7 injects diversity. Qdrant has no native MMR operator (as of writing), so you implement it client-side over the retrieved candidate set — cheap, since you already have the vectors and it's a handful of passages. A lighter-weight alternative Qdrant *does* offer is **grouping** (§7): collapse multiple chunks of the same document into one result, which removes the most obvious redundancy without full MMR. Use grouping for "one hit per document"; use MMR when you need semantic diversity across documents.

### 8.10 Query-side techniques (before retrieval even runs)

Retrieval quality isn't only about the index — the *query* you send matters. These live in your app / Semantic Kernel layer, not in Qdrant, and they're some of the highest-ROI RAG improvements:

- **Query rewriting.** Use an LLM to clean the raw user turn before retrieval: resolve pronouns and follow-ups against chat history ("what about *its* pricing?" → "what about Qdrant Cloud's pricing?"), fix typos, expand abbreviations. Essential for multi-turn RAG.
- **Multi-query (query expansion).** Generate several paraphrases of the question, retrieve for each, and fuse the results with RRF. Boosts recall for ambiguous or under-specified queries, because different phrasings surface different passages.
- **HyDE (Hypothetical Document Embeddings).** Ask the LLM to draft a *hypothetical answer* to the question, embed that draft, and retrieve with it. Counter-intuitive but effective: a plausible answer often lands closer in embedding space to the real passages than the terse question does. Costs one extra LLM call; helps most when questions are short and documents are verbose.
- **Contextual retrieval.** Before embedding each chunk, prepend a short LLM-generated blurb situating it in its parent document ("This chunk is from the 2025 fees policy, section on late payment…"). The chunk then carries document-level context it would otherwise lose when split, which markedly improves retrieval — and pairs especially well with BM25 (the added context adds matchable terms). One-time ingest-side cost; recurring retrieval-quality gain.

```
 REWRITING     "what about its pricing?"  ──LLM+history──▶ "Qdrant Cloud pricing tiers"
               ✨ fixes multi-turn. cheapest win in the whole guide.

 MULTI-QUERY   q ──▶ q1 "how do I tune HNSW"     ──▶ retrieve ─┐
                 ──▶ q2 "improving ANN recall"   ──▶ retrieve ─┼─▶ RRF ─▶ better recall
                 ──▶ q3 "ef vs m parameters"     ──▶ retrieve ─┘

 HyDE          short question ──LLM──▶ a fake answer paragraph ──embed──▶ search
               ❓ "what is ef?"  →  📄 "ef controls the candidate list size during…"
               the fake answer looks like your documents. questions don't. 🤯

 CONTEXTUAL    chunk: "…must be paid within 14 days."          ← alone: unsearchable
   RETRIEVAL   ▶ "From the 2025 Fees Policy, late-payment
                  section: …must be paid within 14 days."      ← now it retrieves
```

None of these require Qdrant changes; they shape the vector/sparse query you ultimately send.

### 8.11 Putting it all together — the reference retrieval pipeline

End to end, a strong RAG retrieval path chains the pieces above:

```
1. Query understanding   →  rewrite / expand (multi-query, HyDE)          [app / SK]
2. Hybrid retrieval      →  dense (semantic) + sparse (BM25) , top-N=30–50 [Qdrant, §8.2]
3. Fuse                  →  RRF (or DBSF)                                   [Qdrant]
4. (optional) rescore    →  ColBERT late-interaction                       [Qdrant, §8.1]
5. Rerank                →  cross-encoder over (query, passage), top-N→k    [your reranker, §8.8]
6. Diversify             →  MMR / grouping to drop near-duplicates          [app / Qdrant, §8.9]
7. Assemble context      →  final k=4–8 passages, ordered best-first        [app]
8. Generate              →  LLM answers grounded on the passages, cites     [SK]
```

```
   🍜 the funnel, with the numbers on it:

   corpus  ████████████████████████████████████████  1,000,000 chunks
      │  ① rewrite / expand the query
      ▼
   hybrid  ██████                                    ~60 (30 dense + 30 sparse)
      │  ② + ③ retrieve both legs, fuse with RRF
      ▼
   fused   ████                                      50
      │  ④ optional ColBERT rescore
      ▼
   rerank  █                                         cross-encoder reorders
      │  ⑤ + ⑥ keep the best, drop near-duplicates
      ▼
   context ▌                                         4–8 passages → ⑧ the LLM
```

Not every system needs every stage. A minimal viable pipeline is dense-only + top-k. The next upgrade is hybrid + RRF (step 2–3). The single biggest precision jump after that is usually the cross-encoder rerank (step 5). Query rewriting (step 1) is the cheapest multi-turn fix. Add stages when your **measured** recall/precision says you need them — every stage adds latency, so earn each one against an eval set.

> 🎯 **Build order if you're starting today:** dense-only → **hybrid + RRF** → **cross-encoder rerank** → query rewriting → MMR/grouping → the exotic stuff. Stop whenever the eval numbers stop moving.

---

## 9. Quantization

Quantization compresses stored vectors to cut RAM and speed up scoring, at a small, tunable accuracy cost. It's the biggest memory/latency lever Qdrant offers. Quantized vectors are stored **alongside** the originals, so you can always rescore against full precision.

> 💡 **Mental model: JPEG for vectors.** You keep a compressed copy for the fast pass and the original for the final look. Search the thumbnails, confirm on the full-res.
>
> ```
>   float32  [0.05123, 0.61044, -0.12876, …]   4 bytes/dim   the original 📸
>   int8     [   6,      78,      -16,     …]   1 byte/dim    4×  smaller
>   4-bit    [   1,       5,       -1,     …]   ½ byte/dim    8×  smaller
>   1-bit    [   1,       1,        0,     …]   ⅛ byte/dim    32× smaller
>            └─ search on these, then rescore the top-k on the originals ─┘
> ```

Four methods:

| Method | Compression | Notes |
|---|---|---|
| **TurboQuant** (v1.18) | 8× (4-bit, default) → 32× (1-bit) | Google-derived; random rotation makes it work on *any* distribution. Asymmetric automatically (queries stay full precision). Strong recall; good default. |
| **Scalar (int8)** (v1.1) | 4× | `float32→uint8`. SIMD-accelerated, ~<1% error. Well-established. |
| **Binary** (v1.5) | up to 32× | 1 bit/dim; fastest (up to ~40× speedup). Needs high-dim, centered vectors (great on OpenAI/Cohere-scale). 1.5/2-bit variants (v1.15) improve small-dim recall; **asymmetric** query encoding (scalar8/4-bit) boosts precision cheaply. |
| **Product (PQ)** (v1.2) | up to 64× | k-means centroids. Max compression, but not SIMD-friendly (slower) and lossier. Use when RAM is the only priority. |

```
   6.1 GB of float32 vectors (1M × 1536), after:

   none        ████████████████████████████████  6.1 GB   baseline
   Scalar 4×   ████████                          1.5 GB   ~<1% error, SIMD fast
   Turbo 8×    ████                              0.8 GB   🎯 strong default
   Binary 32×  █                                 0.2 GB   ⚡ fastest, needs high-dim
   Product 64× ▌                                 0.1 GB   max squeeze, slower scoring
```

**Rule of thumb (compression → method):** 4× → Scalar (or 4-bit TurboQuant for 2× the compression at similar recall); 8× → 4-bit TurboQuant; 16/24/32× → TurboQuant vs binary at that bit-depth (TurboQuant = better recall, binary = faster); up to 64× → Product.

**Enable at collection creation** (examples):

```csharp
// TurboQuant, 2-bit, pinned to RAM
await client.CreateCollectionAsync("c",
    vectorsConfig: new VectorParams { Size = 1536, Distance = Distance.Cosine },
    quantizationConfig: new QuantizationConfig
    {
        Turboquant = new TurboQuantization { AlwaysRam = true, Bits = TurboQuantBitSize.Bits2 }
    });

// Scalar int8
quantizationConfig: new QuantizationConfig
{
    Scalar = new ScalarQuantization { Type = QuantizationType.Int8, Quantile = 0.99f, AlwaysRam = true }
};

// Binary, 2-bit, asymmetric queries
quantizationConfig: new QuantizationConfig
{
    Binary = new BinaryQuantization
    {
        Encoding = BinaryQuantizationEncoding.TwoBits,
        QueryEncoding = new BinaryQuantizationQueryEncoding
        {
            Setting = BinaryQuantizationQueryEncoding.Types.Setting.Scalar8Bits
        },
        AlwaysRam = true
    }
};

// Product, 16×
quantizationConfig: new QuantizationConfig
{
    Product = new ProductQuantization { Compression = CompressionRatio.X16, AlwaysRam = true }
};
```

Enable on an existing collection with `UpdateCollectionAsync` (omit vector config). **Disable:**

```csharp
await client.UpdateCollectionAsync("c",
    quantizationConfig: new QuantizationConfigDiff { Disabled = new Disabled() });
```

**Search-time controls** (`SearchParams.Quantization`):

```csharp
searchParams: new SearchParams
{
    Quantization = new QuantizationSearchParams
    {
        Ignore       = false,   // set true to A/B against full precision (recall check)
        Rescore      = true,    // re-rank top-k with original vectors
        Oversampling = 2.0      // pre-select 2× limit on quantized, then rescore
    }
};
```

```
   OVERSAMPLING + RESCORE, limit=100, oversampling=2.4

   quantized index  ░░░░░░░░░░░░░░░░░░░░░░░░  pre-select 240  (fast, slightly lossy)
                              │ rescore with the ORIGINAL vectors
                              ▼
   final answer     ▓▓▓▓▓▓▓▓▓▓                top 100          (accurate again)

   you spend a little extra scan to buy back almost all the lost recall. 🧠
```

`rescore` defaults **on** for binary and TurboQuant 1/1.5/2-bit (they need it), off for others. `oversampling` (v1.3) trades speed for quality at query time: with limit 100 and oversampling 2.4, Qdrant pre-selects 240 on the quantized index then rescores to the top 100.

**Three storage modes** (the memory/speed dial):
1. **All in RAM** — originals + quantized both in RAM. Fastest, heaviest. Default.
2. **Original on disk, quantized in RAM** — set vector `OnDisk = true` **and** quantization `AlwaysRam = true`. The sweet spot: big RAM savings, most speed retained. Watch disk-read latency on the rescore step; disable `rescore` if it bottlenecks.
3. **All on disk** — `OnDisk = true`, `AlwaysRam = false`. Smallest footprint; requires fast SSD/NVMe.

```
   ┌ MODE 1 ─ all RAM ───────┐ ┌ MODE 2 ─ the sweet spot ─┐ ┌ MODE 3 ─ all disk ───┐
   │ RAM  [orig][quant]      │ │ RAM  [quant]             │ │ RAM  ·               │
   │ DISK ·                  │ │ DISK [orig]              │ │ DISK [orig][quant]   │
   │ 💨 fastest  💰 heaviest │ │ 🎯 big savings, fast     │ │ 🪶 tiny  🐢 needs NVMe│
   └─────────────────────────┘ └──────────────────────────┘ └──────────────────────┘
                                OnDisk=true + AlwaysRam=true
```

**Accuracy tuning:** lower the scalar `Quantile` (e.g. 0.99) to clip outliers from quantization bounds; enable `rescore`; A/B with `Ignore = true` to see quantization's real impact on your data.

---

## 10. Production internals

### Sharding & replication
A collection is split into **shards** across cluster nodes; each shard can have **replicas** for HA. Set `shard_number` (parallelism/scale) and `replication_factor` (durability/availability) at creation. **Sharding methods:** automatic, or **custom sharding** where you route by a shard key (e.g. per-tenant or per-region shards) for data locality and blast-radius control.

```
   COLLECTION (shard_number = 3, replication_factor = 2)

        node-A            node-B            node-C
      ┌─────────┐       ┌─────────┐       ┌─────────┐
      │ S0 ⭐   │       │ S0 ♻️   │       │ S1 ♻️   │      ⭐ = primary copy
      │ S2 ♻️   │       │ S1 ⭐   │       │ S2 ⭐   │      ♻️ = replica
      └─────────┘       └─────────┘       └─────────┘
   shards  → scale + parallelism (split the data)
   replicas→ survive a node loss (copy the data)
   lose node-B? S0 and S1 still served from A and C. ✅ (run ≥3 nodes for quorum)
```

### Consistency & write ordering
Replication makes these tunable per request:
- **`write_ordering`** — `weak` (default, fastest), `medium`, or `strong` (ordered through the leader). Raise it when write order across replicas must be deterministic.
- **read `consistency`** — how many replicas must agree on a read (`1`, `majority`, `all`, or an int). Higher = more correct, more latency. This is the classic latency-vs-correctness dial for a replicated cluster.

```
   fast ◀──────────────────────────────────────────────────▶ correct
   write_ordering:  weak ──────── medium ──────── strong
   read consistency: 1 ────── majority ────────── all
                     ▲                              ▲
              "a stale answer is fine"      "a stale answer is a bug"
```

### Optimizers & storage
Background **optimizers** keep segments healthy: they vacuum deleted points, merge small segments, build/rebuild the HNSW index, and defragment. `optimizers_config` controls thresholds (e.g. `default_segment_number`, `memmap_threshold`, `indexing_threshold`). During heavy ingest you'll see collection status `yellow` while indexing catches up; `IndexedOnly = true` on queries keeps latency predictable meanwhile. Move cold data to mmap (`on_disk`) to shrink RAM.

```
   ingest burst ──▶ 🟡 many small segments, HNSW building, deleted points lingering
                          │  optimizer: merge · vacuum · build index · defragment
                          ▼
                    🟢 fewer, bigger, indexed, tidy segments
   meanwhile, queries with IndexedOnly=true skip the un-indexed ones → stable p99
```

### Snapshots & backups
Snapshots are point-in-time archives for backup, migration, and disaster recovery.

```csharp
// Per-collection
var snap  = await client.CreateSnapshotAsync("articles");
var snaps = await client.ListSnapshotsAsync("articles");
await client.DeleteSnapshotAsync("articles", snap.Name);

// Whole-storage (all collections)
var full  = await client.CreateFullSnapshotAsync();
var fulls = await client.ListFullSnapshotsAsync();
await client.DeleteFullSnapshotAsync(full.Name);
```

Snapshot files live under `/qdrant/snapshots` (mount this to a persistent volume). **Recovery** restores a collection from a snapshot file or URL — via the REST recover endpoint (`PUT /collections/{name}/snapshots/recover` with a `location`, or upload a `.snapshot` file). For real DR, ship snapshot files off-box to object storage (S3/GCS) on a schedule and test restores. Access to snapshot create/delete/recover is governed by the RBAC table below.

> ⚠️ A snapshot sitting on the same container you're trying to survive the loss of is **not a backup**. Off-box + a rehearsed restore, or it doesn't count.

### Security
Self-hosted Qdrant is **wide open by default** — no auth, all interfaces bound. Before production:

```
   🚨 fresh `docker run qdrant/qdrant` =
      no auth · no TLS · bound to 0.0.0.0 · anyone who can reach the port owns your data
      (yes, really. the defaults are for your laptop, not the internet.)
```

1. **API keys** (`api_key` config or `QDRANT__SERVICE__API_KEY` env). Optional **`read_only_api_key`** for query-only consumers. Rotate the admin key with no downtime via **`alt_api_key`** (v1.17) + rolling restart.
2. **Granular RBAC via JWT** (v1.9) — enable `jwt_rbac: true`. Sign HS256 tokens with the admin key (offline, client-side). Claims: `exp` (expiry), `access` (`r` global / `m` manage / per-collection `r`|`rw`), and `value_exists` (validate a token against a point in a collection — lets you *revoke* a token by changing that point, without rotating the api-key). Full allowed/denied matrix per level (manage / read-only / collection-rw / collection-ro) is published in the docs.
3. **Network bind** — `127.0.0.1` locally, a private interface in prod. With Docker: `-p 127.0.0.1:6333:6333`, or `QDRANT__SERVICE__HOST=127.0.0.1`.
4. **TLS** (v1.2) — `enable_tls: true` with `cert`/`key` `.pem`; p2p TLS for inter-node; optional client-cert verification; auto cert rotation (`cert_ttl`, default 1h).
5. **Audit logging** (v1.17) — `audit:` config writes every authn/authz'd op as JSON (daily/hourly rotation, retention). Attach a **tracing id** per request (`x-request-id` header → `RequestHeaders.Use("x-request-id", id)` in C#) to correlate client and server. Query it via `/audit/logs` (v1.18) with time-range and field filters.
6. **Hardening** — run the `-unprivileged` image / `--user`, `--read-only` root FS (data stays on mounted `/qdrant/storage` + `/qdrant/snapshots`), and block external egress (internal Docker network / k8s NetworkPolicy); multi-node clusters only need TCP 6333/6334/6335 between peers.

```
   the layers, outside-in:
   🌐 network bind + NetworkPolicy   ← can they even reach the port?
   🔐 TLS                            ← can they read the wire?
   🔑 API key / read-only key        ← are they anyone at all?
   🎫 JWT RBAC (+ value_exists)      ← what exactly may they touch? revoke instantly
   📝 audit log + x-request-id       ← what did they do, and can you trace it?
   📦 unprivileged, read-only FS     ← how bad is it if they get in anyway?
```

**Auth in the C# client:**

```csharp
var client = new QdrantClient("host", 6334, https: true, apiKey: "…");
// or Bearer token via headers:
var client2 = new QdrantClient("host", 6334, https: true, apiKey: null,
    grpcTimeout: default, loggerFactory: null,
    headers: new Dictionary<string, string> { ["authorization"] = "Bearer <jwt>" });
```

### Observability
- **`/metrics`** — Prometheus endpoint (request rates, latencies, collection sizes, optimizer status). Scrape it; alert on it.
- **`/healthz`, `/livez`, `/readyz`** — health/liveness/readiness probes for k8s and load balancers.
- **`/telemetry`** — detailed internal state (segments, config, cluster). `GetCollectionInfoAsync` surfaces per-collection status and counts for app-level checks.

Baseline production alerts: collection status stuck `red`; `yellow` for longer than your ingest window; p99 query latency regression; RAM near limit; replica down; snapshot job failure.

```
   📟 the pager-worthy six
   ┌────────────────────────────────┬───────────────────────────────────┐
   │ collection status 🔴 red       │ something is actually broken      │
   │ 🟡 yellow past the ingest window│ optimizer stuck / starved        │
   │ p99 query latency regression   │ ef too high? filters unindexed?   │
   │ RAM near limit                 │ time for quantization / on-disk   │
   │ replica down                   │ quorum at risk                    │
   │ snapshot job failed            │ your DR plan is currently fiction │
   └────────────────────────────────┴───────────────────────────────────┘
```

---

# PART 2 — Quick Hands-On with C# ⚡

Goal: from nothing to a working semantic + filtered search in about ten minutes. Every step lists what you should **see**. This uses deterministic fake vectors so it runs with zero API keys or models — swap in a real embedder later (Part 3 shows how).

```
   ⏱️  10 minutes, 7 steps, 0 API keys

   [1] docker run ──▶ [2] dotnet new ──▶ [3] create collection
                                              │
   [7] filtered search ◀── [6] search ◀── [5] upsert ◀── [4] payload index
        │
        └──▶ you now understand §3 §4 §5 §6 §7 by having typed them 💪
```

### Prerequisites
- Docker
- .NET 8 or 9 SDK (`dotnet --version`)

### Step 1 — Run Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

**You should see** log lines ending with something like `Qdrant HTTP listening on 6333` and `gRPC listening on 6334`. Open **http://localhost:6333/dashboard** — the web UI loads with no collections yet. Leave this running.

### Step 2 — Create the project

```bash
dotnet new console -n QdrantHandsOn
cd QdrantHandsOn
dotnet add package Qdrant.Client
```

**You should see** `PackageReference for package 'Qdrant.Client' … added`.

### Step 3–7 — The program

Replace `Program.cs` with the full listing below, then:

```bash
dotnet run
```

```csharp
using Qdrant.Client;
using Qdrant.Client.Grpc;
using static Qdrant.Client.Grpc.Conditions;

var client = new QdrantClient("localhost", 6334);
const string Collection = "handson";
const ulong Dim = 8; // tiny for demo

// --- Step 3: (re)create a collection ---
if (await client.CollectionExistsAsync(Collection))
    await client.DeleteCollectionAsync(Collection);

await client.CreateCollectionAsync(
    collectionName: Collection,
    vectorsConfig: new VectorParams { Size = Dim, Distance = Distance.Cosine });

Console.WriteLine($"[3] Collection '{Collection}' created (dim={Dim}, cosine).");

// --- Step 4: a payload index so filters are fast ---
await client.CreatePayloadIndexAsync(Collection, "category", PayloadSchemaType.Keyword);
Console.WriteLine("[4] Payload index on 'category' created.");

// --- Step 5: upsert a handful of points (deterministic fake embeddings) ---
static float[] FakeEmbed(string s)
{
    // stable pseudo-embedding: hash chars into 8 dims, then L2-normalize
    var v = new float[8];
    foreach (var c in s) v[c % 8] += 1f;
    var norm = MathF.Sqrt(v.Sum(x => x * x));
    if (norm > 0) for (int i = 0; i < v.Length; i++) v[i] /= norm;
    return v;
}

var docs = new (ulong Id, string Title, string Category)[]
{
    (1, "Introduction to vector databases", "database"),
    (2, "Tuning HNSW for recall and speed", "database"),
    (3, "A beginner's guide to sourdough",  "cooking"),
    (4, "Hybrid search with dense + sparse", "database"),
    (5, "Slow-roast lamb techniques",        "cooking"),
};

await client.UpsertAsync(Collection, docs.Select(d => new PointStruct
{
    Id = d.Id,
    Vectors = FakeEmbed(d.Title),
    Payload = { ["title"] = d.Title, ["category"] = d.Category }
}).ToList());

Console.WriteLine($"[5] Upserted {docs.Length} points.");
Console.WriteLine($"    Count = {await client.CountAsync(Collection)}");

// --- Step 6: semantic search ---
Console.WriteLine("\n[6] Search: 'approximate nearest neighbour indexing'");
var q = FakeEmbed("approximate nearest neighbour indexing");
foreach (var h in await client.QueryAsync(Collection, query: q, limit: 3, payloadSelector: true))
    Console.WriteLine($"    #{h.Id}  score={h.Score:F3}  {h.Payload["title"].StringValue}");

// --- Step 7: the same search, filtered to one category ---
Console.WriteLine("\n[7] Same query, filtered to category = 'cooking'");
foreach (var h in await client.QueryAsync(Collection,
             query: q, limit: 3,
             filter: MatchKeyword("category", "cooking"),
             payloadSelector: true))
    Console.WriteLine($"    #{h.Id}  score={h.Score:F3}  {h.Payload["title"].StringValue}");

Console.WriteLine("\nDone.");
```

### What you should see

```
[3] Collection 'handson' created (dim=8, cosine).
[4] Payload index on 'category' created.
[5] Upserted 5 points.
    Count = 5

[6] Search: 'approximate nearest neighbour indexing'
    #2  score=0.9xx  Tuning HNSW for recall and speed
    #1  score=0.8xx  Introduction to vector databases
    #4  score=0.8xx  Hybrid search with dense + sparse

[7] Same query, filtered to category = 'cooking'
    #3  score=0.xxx  A beginner's guide to sourdough
    #5  score=0.xxx  Slow-roast lamb techniques

Done.
```

Exact scores will vary (the fake embedder is a toy), but the **shape** is the point: step 6 surfaces the database-related titles first; step 7 restricts the *same* vector query to the `cooking` category using the payload index. Refresh the dashboard and you'll see the `handson` collection with 5 points.

```
   the exact same query vector, two different worlds:

   [6] no filter                        [7] filter: category = cooking
   ┌────────────────────────────┐       ┌────────────────────────────┐
   │ 🥇 #2 HNSW tuning          │       │ 🥇 #3 sourdough            │
   │ 🥈 #1 vector databases     │       │ 🥈 #5 slow-roast lamb      │
   │ 🥉 #4 hybrid dense+sparse  │       │    (the db docs are gone   │
   └────────────────────────────┘       │     — not reranked, but    │
    the semantics won                   │     filtered OUT entirely) │
                                        └────────────────────────────┘
   💡 the filter doesn't reorder the ranking. it changes who was ever eligible.
```

### What just happened (mapping to Part 1)
- **Step 3** = §3 create collection (cosine, fixed dim).
- **Step 4** = §5 payload index → §6 filters are fast.
- **Step 5** = §4 batch upsert with payload.
- **Step 6** = §7 `QueryAsync` nearest-neighbour.
- **Step 7** = §7 + §6 filtered vector search.

### Next moves
- Swap `FakeEmbed` for a real model: Ollama (`all-minilm`) locally, or OpenAI/Azure OpenAI. Set `Dim` to the model's dimension (384, 768, 1536…).
- Add a `sparse` named vector and try **RRF hybrid** (§8.2).
- Turn on **scalar quantization** (§9) and compare `Ignore=true` vs `false` to feel the recall/speed trade.

That progression is exactly what Part 3 builds into a real service.

```
   🪜 hands-on ──▶ real embedder ──▶ hybrid + RRF ──▶ quantization ──▶ PART 3
      toy vectors    384/768/1536     dense+sparse      RAM sanity      DocMind
```

---

# PART 3 — Production Project: **DocMind** 🏗️

A .NET RAG service that answers questions over a multi-tenant document corpus (a natural fit for an education-sector platform: per-institute knowledge bases). It grounds a **Semantic Kernel** agent on Qdrant using **hybrid retrieval**, and exposes its capabilities as **MCP tools** so any MCP client (Claude Desktop, an orchestrator, another agent) can call them.

This section is deliberately concrete — deployment, resilience, security, backup, and observability are built out as code and config, not left as "TODO: add retries".

```
   what "production" means in this part, concretely:
   ✅ retries + circuit breaker      ✅ deterministic ids (safe re-ingest)
   ✅ tenant isolation, defence-in-depth  ✅ snapshots shipped off-box
   ✅ health checks + metrics        ✅ TLS + API key + (optional) JWT RBAC
   ✅ an ops runbook you can hand to the on-call person at 3am
```

## 3.1 Architecture

```
                 ┌─────────────────────────────────────────────┐
   MCP client ──▶│  DocMind.Mcp   (MCP server: tools)           │
 (Claude etc.)   │    • ingest_document                         │
                 │    • search_documents                        │
                 │    • answer_question (RAG)                    │
                 └───────────────┬─────────────────────────────┘
                                 │ calls
                 ┌───────────────▼─────────────────────────────┐
                 │  DocMind.Core                                │
                 │   IngestionService   RagService              │
                 │   IEmbeddingClient (dense + sparse)          │
                 │   SemanticKernel (chat + prompt)             │
                 └───────┬───────────────────────┬─────────────┘
                         │ gRPC (Polly retry)     │
                 ┌───────▼────────┐       ┌───────▼───────────┐
                 │ Qdrant cluster │       │ Embedding models  │
                 │  chunks +      │       │ (Azure OpenAI /   │
                 │  documents     │       │  Ollama)          │
                 └────────────────┘       └───────────────────┘
```

Two projects: `DocMind.Core` (retrieval + RAG logic, reusable) and `DocMind.Mcp` (the MCP host). Keeping Core free of MCP concerns means the same logic can also sit behind an ASP.NET API or a background worker.

> 💡 **Why the split matters.** Core knows nothing about MCP. Tomorrow you bolt on an ASP.NET controller, a Hangfire job, or a Slack bot — same Core, new shell:
> ```
>            ┌── DocMind.Mcp ──┐
>   shells ──┼── ASP.NET API ──┼──▶  DocMind.Core  ──▶ Qdrant + models
>            └── worker/cron ──┘     (the only place logic lives)
> ```

## 3.2 Collection design

Two collections, tenant-isolated, with **named dense + sparse vectors for hybrid**:

- **`docmind_chunks`** — one point per chunk. Dense (768-d, e.g. `bge-base` / Azure `text-embedding-3-small` at 1536) + sparse (BM25/SPLADE). Payload: `tenant_id`, `document_id`, `chunk_index`, `text`, `heading`, `source_uri`, `published_at`.
- **`docmind_documents`** — one point per source document (a lightweight vector or a single zero-vector placeholder is fine; this collection exists mostly for lookup-in-groups enrichment). Payload: `tenant_id`, `title`, `source_uri`, `author`, `updated_at`.

```
   docmind_chunks                        docmind_documents
   ┌──────────────────────────────┐      ┌───────────────────────────┐
   │ point = one CHUNK            │      │ point = one DOCUMENT      │
   │ dense + sparse vectors       │      │ placeholder vector        │
   │ tenant_id, document_id ──────┼─────▶│ tenant_id, title,         │
   │ chunk_index, text, heading   │ join │ source_uri, author,       │
   │ source_uri, published_at     │      │ updated_at                │
   └──────────────────────────────┘      └───────────────────────────┘
    drives RETRIEVAL                      drives ENRICHMENT
    (searched, ranked, fused)             (grouping + lookup-in-groups, §7)
```

Why two collections: chunks drive retrieval; grouping by `document_id` (§7) collapses many chunk hits into per-document results, and **lookup-in-groups** enriches each group with the parent document's metadata in one round-trip.

```csharp
public static class Schema
{
    public const string Chunks = "docmind_chunks";
    public const string Documents = "docmind_documents";
    public const uint DenseDim = 768;

    public static async Task EnsureAsync(QdrantClient c)
    {
        if (!await c.CollectionExistsAsync(Chunks))
        {
            await c.CreateCollectionAsync(Chunks,
                vectorsConfig: new VectorParamsMap
                {
                    Map = { ["dense"] = new VectorParams
                    {
                        Size = DenseDim, Distance = Distance.Cosine, OnDisk = true // originals on disk…
                    }}
                },
                sparseVectorsConfig: new SparseVectorConfig(
                    ("sparse", new SparseVectorParams { Modifier = Modifier.Idf })), // IDF for BM25-like scoring
                quantizationConfig: new QuantizationConfig
                {
                    // …quantized vectors pinned to RAM: the memory sweet spot (§9 mode 2)
                    Scalar = new ScalarQuantization { Type = QuantizationType.Int8, AlwaysRam = true }
                });

            // Filterable + tenant-aware indexes — created BEFORE ingest (§5)
            await c.CreatePayloadIndexAsync(Chunks, "tenant_id", PayloadSchemaType.Keyword);
            await c.CreatePayloadIndexAsync(Chunks, "document_id", PayloadSchemaType.Keyword);
            await c.CreatePayloadIndexAsync(Chunks, "published_at", PayloadSchemaType.Datetime);
            // mark tenant field so Qdrant co-locates each tenant's vectors
            // (set IsTenant via the index params overload in your client version)
        }

        if (!await c.CollectionExistsAsync(Documents))
        {
            await c.CreateCollectionAsync(Documents,
                vectorsConfig: new VectorParams { Size = DenseDim, Distance = Distance.Cosine, OnDisk = true });
            await c.CreatePayloadIndexAsync(Documents, "tenant_id", PayloadSchemaType.Keyword);
        }
    }
}
```

> 🔍 **Every line of that schema is a §-reference cashing in:**
> `OnDisk = true` + `AlwaysRam = true` → §9 storage mode 2 · `Modifier.Idf` → §8.7 path ① · payload indexes before ingest → §5 order-matters · `tenant_id` keyword index → §3 multitenancy + §6 filters.

## 3.3 Embeddings abstraction

Isolate the model behind an interface so you can run Ollama locally and Azure OpenAI in prod without touching retrieval code.

```csharp
public interface IEmbeddingClient
{
    Task<float[]> EmbedDenseAsync(string text, CancellationToken ct = default);
    Task<(float[] values, uint[] indices)> EmbedSparseAsync(string text, CancellationToken ct = default);
    Task<IReadOnlyList<float[]>> EmbedDenseBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
}
```

Dense from your embedding model; sparse from a BM25/SPLADE encoder (e.g. FastEmbed's SPLADE, or a BM25 term-weighting you compute yourself). Batch the dense path — it's the ingest bottleneck.

```
   dev laptop ──▶ OllamaEmbeddingClient      ┐
   prod       ──▶ AzureOpenAIEmbeddingClient ├──▶ IEmbeddingClient ──▶ ingestion + retrieval
   tests      ──▶ FakeEmbeddingClient        ┘     (nothing downstream knows or cares)
   ⚠️ the one thing that must match across all of them: DenseDim.
```

## 3.4 Ingestion pipeline — chunk, embed, upsert (with real robustness)

Production ingestion needs: structure-aware chunking, batched embedding, **idempotent** upserts (deterministic ids so re-ingest overwrites cleanly), batched writes, and **retries** on transient gRPC failures.

```
   raw document
       │  Chunker.Split(512 tokens, 64 overlap)
       ▼
   [chunk 0][chunk 1]…[chunk N]
       │  EmbedDenseBatchAsync × 64   +   EmbedSparseAsync
       ▼
   points with StableId(tenant, "doc:idx")   ← deterministic = re-ingest OVERWRITES
       │  UpsertAsync × 256, wrapped in Polly retry
       ▼
   Qdrant 🟡 indexing… ──▶ 🟢
```

```csharp
public sealed class IngestionService(
    QdrantClient qdrant,
    IEmbeddingClient embed,
    ResiliencePipeline retry,          // Polly v8 pipeline (see 3.8)
    ILogger<IngestionService> log)
{
    private const int EmbedBatch = 64;
    private const int UpsertBatch = 256;

    public async Task<int> IngestAsync(
        string tenantId, string documentId, string title, string sourceUri,
        string fullText, DateTimeOffset publishedAt, CancellationToken ct = default)
    {
        var chunks = Chunker.Split(fullText, targetTokens: 512, overlapTokens: 64).ToList();
        log.LogInformation("Ingest {Doc} for {Tenant}: {N} chunks", documentId, tenantId, chunks.Count);

        // 1) upsert the parent document record (idempotent id = stable hash)
        await retry.ExecuteAsync(async token => await qdrant.UpsertAsync(Schema.Documents, new[]
        {
            new PointStruct
            {
                Id = StableId(tenantId, documentId),
                Vectors = new float[Schema.DenseDim],           // placeholder vector
                Payload =
                {
                    ["tenant_id"] = tenantId, ["title"] = title,
                    ["source_uri"] = sourceUri, ["updated_at"] = publishedAt.ToString("o")
                }
            }
        }, cancellationToken: token), ct);

        // 2) embed + upsert chunks in batches
        var upserted = 0;
        foreach (var slice in chunks.Chunk(UpsertBatch))
        {
            var points = new List<PointStruct>(slice.Length);

            foreach (var embBatch in slice.Chunk(EmbedBatch))
            {
                var texts  = embBatch.Select(x => x.Text).ToList();
                var dense  = await embed.EmbedDenseBatchAsync(texts, ct);
                for (int i = 0; i < embBatch.Length; i++)
                {
                    var chunk  = embBatch[i];
                    var sparse = await embed.EmbedSparseAsync(chunk.Text, ct);
                    points.Add(new PointStruct
                    {
                        Id = StableId(tenantId, $"{documentId}:{chunk.Index}"),  // deterministic → idempotent
                        Vectors = new Dictionary<string, Vector>
                        {
                            ["dense"]  = dense[i],
                            ["sparse"] = new (sparse.values, sparse.indices),
                        },
                        Payload =
                        {
                            ["tenant_id"] = tenantId, ["document_id"] = documentId,
                            ["chunk_index"] = chunk.Index, ["text"] = chunk.Text,
                            ["heading"] = chunk.Heading ?? "", ["source_uri"] = sourceUri,
                            ["published_at"] = publishedAt.ToString("o")
                        }
                    });
                }
            }

            await retry.ExecuteAsync(
                async token => await qdrant.UpsertAsync(Schema.Chunks, points, cancellationToken: token), ct);
            upserted += points.Count;
        }

        log.LogInformation("Ingest {Doc}: {N} chunks upserted", documentId, upserted);
        return upserted;
    }

    // deterministic UUIDv5-style id from tenant + key → safe re-ingest (overwrite, not duplicate)
    private static Guid StableId(string tenant, string key)
    {
        using var md5 = System.Security.Cryptography.MD5.Create();
        var bytes = md5.ComputeHash(System.Text.Encoding.UTF8.GetBytes($"{tenant}/{key}"));
        return new Guid(bytes);
    }
}
```

> 💡 **Why `StableId` is the hero of this file.** Random ids + a retried upsert = duplicates forever. Deterministic ids make the write *idempotent*, which is what makes the retry in §3.8 safe:
> ```
>   random id:  ingest → timeout → retry → 2 copies of the same chunk 👻👻
>   stable id:  ingest → timeout → retry → 1 chunk, overwritten. ✅
> ```

A minimal structure-aware chunker (real systems split on headings/sentences; this is the shape):

```csharp
public static class Chunker
{
    public record Chunk(int Index, string Text, string? Heading);

    public static IEnumerable<Chunk> Split(string text, int targetTokens, int overlapTokens)
    {
        // approximate tokens as words * 1.3; split on paragraphs, pack to target, carry overlap
        var paras = text.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        var buf = new List<string>(); var idx = 0; var approx = 0;
        foreach (var p in paras)
        {
            var t = (int)(p.Split(' ').Length * 1.3);
            if (approx + t > targetTokens && buf.Count > 0)
            {
                yield return new Chunk(idx++, string.Join("\n\n", buf), null);
                var carry = Math.Max(0, buf.Count - overlapTokens / 20);
                buf = buf.Skip(carry).ToList();
                approx = buf.Sum(x => (int)(x.Split(' ').Length * 1.3));
            }
            buf.Add(p); approx += t;
        }
        if (buf.Count > 0) yield return new Chunk(idx, string.Join("\n\n", buf), null);
    }
}
```

## 3.5 Retrieval — hybrid RRF + grouping, tenant-scoped

The retrieval core: fuse sparse+dense with RRF, restrict to the tenant, group by document, and (optionally) rescore with recency.

```csharp
public sealed record Passage(string DocumentId, string Text, string SourceUri, float Score);

public sealed class RagRetriever(QdrantClient qdrant, IEmbeddingClient embed, ResiliencePipeline retry)
{
    public async Task<IReadOnlyList<Passage>> RetrieveAsync(
        string tenantId, string question, int k = 6, CancellationToken ct = default)
    {
        var dense  = await embed.EmbedDenseAsync(question, ct);
        var sparse = await embed.EmbedSparseAsync(question, ct);
        var tenant = MatchKeyword("tenant_id", tenantId);

        var results = await retry.ExecuteAsync(async token =>
            await qdrant.QueryAsync(
                collectionName: Schema.Chunks,
                prefetch: new List<PrefetchQuery>
                {
                    new() { Query = new (sparse.values, sparse.indices), Using = "sparse",
                            Filter = tenant, Limit = 30 },
                    new() { Query = dense, Using = "dense",
                            Filter = tenant, Limit = 30 },
                },
                query: new Rrf(),                 // hybrid fusion (§8.2)
                filter: tenant,                   // safety: enforce tenant on the fused stage too
                limit: (ulong)k,
                payloadSelector: true,
                searchParams: new SearchParams { HnswEf = 128 },
                cancellationToken: token), ct);

        return results.Select(r => new Passage(
            r.Payload["document_id"].StringValue,
            r.Payload["text"].StringValue,
            r.Payload.TryGetValue("source_uri", out var u) ? u.StringValue : "",
            r.Score)).ToList();
    }
}
```

Tenant isolation is enforced on **every** prefetch and on the fused query — defence in depth, so a bug in one place can't leak another tenant's data.

```
   🔒 three gates, same tenant filter, on purpose:

   sparse prefetch ──[tenant]──┐
                               ├──▶ RRF fuse ──[tenant]──▶ results
   dense  prefetch ──[tenant]──┘

   redundant? yes. that's the point — a cross-tenant leak needs THREE
   independent bugs, not one. this is the cheapest insurance in the file.
```

## 3.5.1 Adding a cross-encoder rerank stage

The `RetrieveAsync` above returns fused (RRF) hits directly. For production precision, insert a **cross-encoder rerank** between retrieval and generation: retrieve *wider* (top-N), rerank, keep the final k. This is the single biggest quality lever after hybrid (§8.8).

```csharp
public sealed class RerankingRetriever(
    RagRetriever inner,                 // the hybrid RRF retriever from 3.5
    ICrossEncoderReranker reranker,     // §8.8 — hosted API or local ONNX
    ILogger<RerankingRetriever> log)
{
    public async Task<IReadOnlyList<Passage>> RetrieveAsync(
        string tenantId, string question, int finalK = 6, CancellationToken ct = default)
    {
        // 1) retrieve WIDE — recall stage
        var candidates = await inner.RetrieveAsync(tenantId, question, k: 40, ct: ct);
        if (candidates.Count <= finalK) return candidates;

        // 2) rerank — precision stage (query attends to each passage jointly)
        var reranked = await reranker.RerankAsync(question, candidates, finalK, ct);

        log.LogInformation("Rerank {Tenant}: {N} candidates → top {K}", tenantId, candidates.Count, finalK);
        return reranked;   // (optionally) run MMR here to drop near-duplicates — §8.9
    }
}
```

Wire it into DI and have `RagService` depend on `RerankingRetriever` instead of `RagRetriever`:

```csharp
builder.Services.AddSingleton<RagRetriever>();
builder.Services.AddSingleton<ICrossEncoderReranker, HostedReranker>();   // or a local OnnxReranker
builder.Services.AddSingleton<RerankingRetriever>();
builder.Services.Configure<RerankOptions>(builder.Configuration.GetSection("Rerank"));
builder.Services.AddHttpClient<HostedReranker>(c =>
    c.DefaultRequestHeaders.Authorization =
        new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer",
            builder.Configuration["Rerank:ApiKey"]));
```

Cost/latency note: the rerank call adds one model pass over ~40 passages, typically tens of milliseconds (local ONNX) to a couple hundred (hosted API). Keep the candidate N bounded, cache reranks for identical (query, tenant) pairs if traffic is repetitive, and fall back to the un-reranked RRF order if the reranker is unavailable (wrap it in the same Polly pipeline as §3.8). If you'd rather avoid an external model entirely, swap this stage for ColBERT late-interaction rescoring inside Qdrant (§8.1, §8.8) — no extra service, most of the gain.

> ⚠️ **Fail open, not closed.** If the reranker times out, return the RRF order — slightly worse ranking beats no answer. Degradation, not an outage.

## 3.6 RAG answer with Semantic Kernel

Ground the LLM strictly on retrieved passages, with citations and an explicit "I don't know" path.

```csharp
public sealed class RagService(Kernel kernel, RagRetriever retriever, ILogger<RagService> log)
{
    public async Task<string> AnswerAsync(string tenantId, string question, CancellationToken ct = default)
    {
        var passages = await retriever.RetrieveAsync(tenantId, question, k: 6, ct: ct);
        if (passages.Count == 0)
            return "I don't have any documents that cover that.";

        var context = string.Join("\n\n---\n\n",
            passages.Select((p, i) => $"[{i + 1}] (source: {p.SourceUri})\n{p.Text}"));

        var prompt = """
            You are DocMind, answering strictly from the CONTEXT.
            If the answer is not in the context, say you don't know.
            Cite sources inline as [1], [2] matching the context blocks.

            QUESTION: {{$question}}

            CONTEXT:
            {{$context}}
            """;

        var fn = kernel.CreateFunctionFromPrompt(prompt, new OpenAIPromptExecutionSettings
        {
            Temperature = 0.1, MaxTokens = 700
        });

        var result = await kernel.InvokeAsync(fn,
            new() { ["question"] = question, ["context"] = context }, ct);

        log.LogInformation("Answered for {Tenant}: {Chars} chars, {P} passages",
            tenantId, result.ToString().Length, passages.Count);
        return result.ToString();
    }
}
```

> 💡 **The three anti-hallucination levers in that prompt**, in order of importance:
> ```
>   1. "strictly from the CONTEXT"        → no outside knowledge
>   2. "if not in the context, say so"    → an explicit escape hatch, so the model
>                                            doesn't have to invent one
>   3. "cite as [1], [2]"                 → citations make a hallucination VISIBLE
>   + Temperature = 0.1                   → this is retrieval, not poetry ✍️
>   + passages.Count == 0 → short-circuit  → never hand an LLM an empty context
> ```

## 3.7 MCP server — expose the capability

Expose ingest / search / answer as MCP tools (using the C# MCP SDK's attribute style):

```csharp
[McpServerToolType]
public sealed class DocMindTools(IngestionService ingest, RagRetriever retriever, RagService rag)
{
    [McpServerTool, Description("Ingest a document into a tenant's knowledge base.")]
    public async Task<string> IngestDocument(
        string tenantId, string documentId, string title, string sourceUri, string text)
    {
        var n = await ingest.IngestAsync(tenantId, documentId, title, sourceUri, text, DateTimeOffset.UtcNow);
        return $"Ingested {n} chunks for document '{documentId}'.";
    }

    [McpServerTool, Description("Hybrid-search a tenant's documents; returns ranked passages with sources.")]
    public async Task<IReadOnlyList<Passage>> SearchDocuments(string tenantId, string query, int k = 6)
        => await retriever.RetrieveAsync(tenantId, query, k);

    [McpServerTool, Description("Answer a question using RAG over a tenant's documents, with citations.")]
    public async Task<string> AnswerQuestion(string tenantId, string question)
        => await rag.AnswerAsync(tenantId, question);
}
```

```
   any MCP client  ──▶  ingest_document    ──▶ IngestionService  ──▶ Qdrant
   (Claude Desktop,     search_documents   ──▶ RagRetriever      ──▶ Qdrant
    an orchestrator,    answer_question    ──▶ RagService        ──▶ Qdrant + LLM
    another agent)
   💡 that `[Description]` string is the tool's UX — it's what the model reads
      to decide whether to call you. write it like a docstring for a colleague.
```

## 3.8 Composition, config & resilience (DI + Polly)

Wire everything with typed config, a resilient gRPC client, and health checks.

```csharp
// appsettings.json (bind to options; secrets via env / Key Vault, never in source)
// {
//   "Qdrant": { "Host": "qdrant", "Port": 6334, "UseTls": true, "ApiKey": "" },
//   "Embedding": { "Provider": "AzureOpenAI", "DenseDim": 768 }
// }

public sealed class QdrantOptions
{
    public string Host { get; set; } = "localhost";
    public int Port { get; set; } = 6334;
    public bool UseTls { get; set; }
    public string? ApiKey { get; set; }
}

var builder = Host.CreateApplicationBuilder(args);
builder.Services.Configure<QdrantOptions>(builder.Configuration.GetSection("Qdrant"));

// Polly v8 resilience pipeline: retry transient gRPC + timeout + circuit breaker
builder.Services.AddSingleton(sp =>
    new ResiliencePipelineBuilder()
        .AddRetry(new RetryStrategyOptions
        {
            MaxRetryAttempts = 4,
            Delay = TimeSpan.FromMilliseconds(200),
            BackoffType = DelayBackoffType.Exponential,
            UseJitter = true,
            ShouldHandle = new PredicateBuilder().Handle<Grpc.Core.RpcException>(e =>
                e.StatusCode is Grpc.Core.StatusCode.Unavailable
                             or Grpc.Core.StatusCode.DeadlineExceeded
                             or Grpc.Core.StatusCode.ResourceExhausted)
        })
        .AddTimeout(TimeSpan.FromSeconds(10))
        .AddCircuitBreaker(new CircuitBreakerStrategyOptions
        {
            FailureRatio = 0.5, SamplingDuration = TimeSpan.FromSeconds(30),
            MinimumThroughput = 10, BreakDuration = TimeSpan.FromSeconds(15)
        })
        .Build());

// A single, long-lived QdrantClient (it manages the gRPC channel/pool)
builder.Services.AddSingleton(sp =>
{
    var o = sp.GetRequiredService<IOptions<QdrantOptions>>().Value;
    return new QdrantClient(o.Host, o.Port, https: o.UseTls, apiKey: o.ApiKey);
});

builder.Services.AddSingleton<IEmbeddingClient, AzureOpenAIEmbeddingClient>();
builder.Services.AddSingleton<IngestionService>();
builder.Services.AddSingleton<RagRetriever>();
builder.Services.AddSingleton<RagService>();

// Semantic Kernel
builder.Services.AddKernel().AddAzureOpenAIChatCompletion(
    deploymentName: builder.Configuration["SK:ChatDeployment"]!,
    endpoint: builder.Configuration["SK:Endpoint"]!,
    apiKey: builder.Configuration["SK:ApiKey"]!);

// Health check that actually pings Qdrant
builder.Services.AddHealthChecks().AddCheck<QdrantHealthCheck>("qdrant");

var app = builder.Build();
await Schema.EnsureAsync(app.Services.GetRequiredService<QdrantClient>()); // create collections/indexes on boot
```

```csharp
public sealed class QdrantHealthCheck(QdrantClient c) : IHealthCheck
{
    public async Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext _, CancellationToken ct = default)
    {
        try { await c.ListCollectionsAsync(ct); return HealthCheckResult.Healthy(); }
        catch (Exception e) { return HealthCheckResult.Unhealthy("Qdrant unreachable", e); }
    }
}
```

> 💡 **What that Polly pipeline is actually doing**, from the outside in:
> ```
>   call ──▶ [ retry ×4, exp backoff + jitter ] ──▶ [ timeout 10s ] ──▶ [ breaker ] ──▶ Qdrant
>              ⤷ only Unavailable / DeadlineExceeded / ResourceExhausted
>
>   breaker: >50% failures over 30s (min 10 calls) ──▶ OPEN for 15s
>            ┌─ CLOSED ──(too many fails)──▶ OPEN ──(15s)──▶ HALF-OPEN ─┐
>            └────────────────(a probe succeeds)◀────────────────────────┘
>   why a breaker at all: when Qdrant is down, retrying 4× just makes YOU slow too.
>   fail fast, recover politely. 🧘
> ```

Notes that matter in prod:
- **One `QdrantClient` singleton.** It owns the gRPC channel; creating one per request exhausts connections.
- **Config, not constants.** Host/keys/dims come from config + environment; secrets from Key Vault or Docker/K8s secrets — never committed.
- **Retry only transient codes** (`Unavailable`, `DeadlineExceeded`, `ResourceExhausted`); retrying a `InvalidArgument` just wastes time.
- **Idempotent writes** (deterministic ids, §3.4) make retries safe — a re-sent upsert overwrites rather than duplicates.

```
   ✅ retry these        ❌ never retry these
   Unavailable           InvalidArgument   ← your payload is wrong. 4 more tries
   DeadlineExceeded      NotFound            won't make it right.
   ResourceExhausted     PermissionDenied
```

## 3.9 Backup & disaster recovery

A scheduled job takes per-collection snapshots and ships them off-box; document the restore.

```csharp
public sealed class SnapshotBackupJob(QdrantClient qdrant, IObjectStorage store, ILogger<SnapshotBackupJob> log)
{
    public async Task RunAsync(CancellationToken ct = default)
    {
        foreach (var col in new[] { Schema.Chunks, Schema.Documents })
        {
            var snap = await qdrant.CreateSnapshotAsync(col, ct);
            log.LogInformation("Snapshot {Name} for {Col}", snap.Name, col);

            // stream /qdrant/snapshots/{col}/{snap.Name} to S3/Blob (the file is on the mounted volume)
            await store.UploadAsync($"qdrant/{col}/{snap.Name}", SnapshotPath(col, snap.Name), ct);

            // retention: keep last N in Qdrant, rest live in object storage
            var all = await qdrant.ListSnapshotsAsync(col, ct);
            foreach (var old in all.OrderByDescending(s => s.CreationTime).Skip(3))
                await qdrant.DeleteSnapshotAsync(col, old.Name, ct);
        }
    }

    static string SnapshotPath(string col, string name) => $"/qdrant/snapshots/{col}/{name}";
}
```

```
   nightly:  CreateSnapshot ──▶ /qdrant/snapshots (mounted volume)
                              ──▶ upload to S3/Blob  ☁️  ← the part that makes it a backup
                              ──▶ keep last 3 locally, prune the rest

   restore:  S3 ──▶ PUT /collections/{name}/snapshots/recover { location }
                ──▶ verify counts  ──▶ repoint the alias (§3)  ──▶ 🎉
   📅 rehearse this on a calendar invite. an untested restore is a rumour.
```

Schedule it (Hangfire/Quartz/K8s CronJob) nightly. **Restore** (rehearse it — an untested backup isn't a backup): recover a collection from a snapshot file/URL via the REST recover endpoint — `PUT /collections/{name}/snapshots/recover` with `{ "location": "file:///qdrant/snapshots/…"}` (or upload the `.snapshot`). Mount `/qdrant/snapshots` (and `/qdrant/storage`) to durable volumes so snapshots survive container replacement.

## 3.10 Security wiring

- Qdrant behind an **API key** (`QDRANT__SERVICE__API_KEY`) + **TLS**; DocMind connects with `https: true, apiKey: …` from config.
- If DocMind is multi-tenant SaaS, consider **JWT RBAC** (§10): mint per-tenant read-write tokens scoped to that tenant's use, so even a compromised DocMind node can't touch other collections. Use the `value_exists` claim against a `tenants` collection for instant revocation without rotating the master key.
- **Never** interpolate `tenant_id` into anything but a Qdrant filter value; it's data, not a query fragment.
- Run Qdrant `-unprivileged`, read-only root FS, internal network (§10 hardening).

```
   🧅 blast-radius thinking:
   a leaked app key  ──▶ scoped JWT means the attacker gets ONE tenant, not all of them
   a compromised pod ──▶ read-only FS + no egress means they can't persist or exfiltrate
   tenant_id as data ──▶ never string-built into a query = no injection surface
```

## 3.11 Observability & HA

**Observability:** scrape Qdrant's `/metrics` (Prometheus) alongside DocMind's own OpenTelemetry traces/metrics; propagate a request/trace id into Qdrant via `RequestHeaders.Use("x-request-id", traceId)` so a slow answer can be traced from MCP call → retrieval → Qdrant audit log. Alert on: collection status `red`; `yellow` beyond the ingest window; p99 retrieval latency; RAM headroom; replica down; snapshot-job failure; circuit breaker open.

```
   one trace id, end to end — this is how you answer "why was that answer slow?"

   MCP call ──[x-request-id: 7f3a]──▶ RagService ──▶ RagRetriever ──▶ Qdrant audit log
      120ms                              40ms          embed 300ms      query 25ms
                                                        ▲
                                     turns out it was never the vector DB 🕵️
```

**High availability:** create the collections with `replication_factor: 2+` and enough `shard_number` for your corpus/throughput; use **custom sharding** by `tenant_id` (or region) for locality and blast-radius control. Choose consistency per call: `write_ordering: strong` for ingest that must be ordered, read `consistency: majority` for answers that must not read stale replicas — trading latency for correctness (§10). Run ≥3 nodes so a single loss keeps quorum.

## 3.12 Ops runbook (condensed) 🚑

| Situation | Action |
|---|---|
| Ingest slow / status stuck `yellow` | Expected during bulk load; queries use `IndexedOnly=true`. If persistent, raise optimizer `indexing_threshold`, check RAM/CPU. |
| Recall too low | Raise query `HnswEf`; verify correct distance metric; if quantized, enable `Rescore` + `Oversampling`; A/B with `Ignore=true`. |
| Latency too high | Lower `HnswEf`; add scalar/TurboQuant quantization (RAM-pinned); ensure payload indexes exist for every filter. |
| RAM pressure | Move originals `OnDisk=true` + quantized `AlwaysRam=true`; shard out; move cold tenants to separate shards. |
| Tenant onboarding | No schema change — same collection, new `tenant_id`; (optional) new custom shard key. |
| Add sparse to a dense-only collection | `CreateVectorNameAsync` (v1.18) migration, then backfill sparse vectors. |
| DR restore | Pull latest snapshot from object storage → recover endpoint → verify counts vs. expected → repoint alias. |
| Key rotation | Set `alt_api_key`, rolling restart, switch clients, promote + drop old key (§10). |

```
   🧭 the 3am decision tree

   "results are bad"  ──▶ bad RECALL?  ──▶ ef ↑ · metric? · rescore+oversample (§9)
                      └─▶ bad ORDER?   ──▶ add rerank (§8.8) / hybrid (§8.2)
   "it's slow"        ──▶ ef ↓ · quantize (§9) · check every filter has an index (§5)
   "RAM is on fire"   ──▶ OnDisk originals + AlwaysRam quantized (§9 mode 2)
   "it's 🔴 red"      ──▶ /telemetry · disk full? · restore from snapshot (§3.9)
```

## 3.13 Build order (suggested)

1. `Schema.EnsureAsync` + hands-on-style smoke test (Part 2) against local Docker.
2. `IEmbeddingClient` (Ollama first, Azure later) — verify dims match `DenseDim`.
3. Ingestion with deterministic ids; ingest a few real docs; check counts + dashboard.
4. `RagRetriever` hybrid RRF; eyeball passages; measure recall vs `Exact=true` on a query sample.
5. `RagService` + SK prompt; confirm citations and the "I don't know" path.
6. MCP host; call the tools from an MCP client.
7. Harden: quantization, Polly, health checks, snapshots job, TLS+API key, metrics, replication.

```
   ①schema ─▶ ②embed ─▶ ③ingest ─▶ ④retrieve ─▶ ⑤answer ─▶ ⑥MCP ─▶ ⑦harden
   ▏         ▏          ▏           ▏            ▏          ▏        ▏
   docker    dims       counts      recall vs    citations  a client  it survives
   works     match      + dashboard  Exact=true   + "idk"    calls it  the on-call
   ↑ each step is independently verifiable. never build two steps before testing one.
```

This is the same arc as Parts 1→2, just built for keeps.

---

## 🃏 The whole guide, on one card

```
 EMBED ──▶ STORE ──▶ INDEX ──▶ FILTER ──▶ SEARCH ──▶ FUSE ──▶ RERANK ──▶ ANSWER
   §1       §2–4      §5        §6         §7        §8.2     §8.8       Part 3

 the five numbers you will actually tune:
   d          embedding dimension       → RAM bill (§1)
   m / ef     HNSW breadth              → recall ⟷ latency (§5, §7)
   N          retrieval candidates      → 30–50, then rerank (§8.9)
   k          passages to the LLM       → 4–8, no more (§8.9)
   λ          MMR diversity             → 0.5–0.7 (§8.9)

 the five mistakes that cost the most:
   ❌ wrong distance metric for the model            → silent recall loss (§1)
   ❌ filtering on an unindexed payload field        → filters become scans (§5)
   ❌ dense-only retrieval on keyword-heavy data     → add sparse + RRF (§8.2, §8.7)
   ❌ shipping ANN without ever measuring recall     → Exact=true exists (§7)
   ❌ random point ids + retries                     → duplicate chunks (§3.4)
```

---

## Source map

Compiled from the official Qdrant documentation at qdrant.tech, consolidated to the C# (`Qdrant.Client`) client:

- Overview, Quickstart — core concepts, setup, first collection/search
- Concepts → Collections — creation, named/sparse vectors, on-disk, aliases, multitenancy, status
- Concepts → Points — upsert, payload/vector ops, scroll, count, schema migration *(concepts/points page returned a redirect; compiled from quickstart + Query API + client reference)*
- Concepts → Indexing — payload/full-text/HNSW/sparse indexes, tenant/principal, tokenizers, ACORN
- Concepts → Filtering — must/should/must_not, match/range/geo/datetime/nested/full-text
- Search → Search — Query API, params, selectors, pagination, grouping, lookup-in-groups
- Search → Hybrid Queries — prefetch, multi-stage, RRF (+weights, k), DBSF, formula queries, distributed fusion
- Search → Explore — recommendation (avg/best-score/sum-scores), discovery, context, distance matrix
- Manage Data → Quantization — TurboQuant, scalar, binary (1.5/2-bit, asymmetric), product, search params, storage modes
- Security — API keys, read-only key, JWT RBAC, access table, audit logging, tracing ids, network bind, TLS, hardening
- Snapshots — *compiled from established Qdrant snapshot API + the snapshot operations confirmed in the Security access-control table; the concepts/snapshots page did not return content on crawl. Verify exact `CreateSnapshotAsync`/recover signatures against your installed `Qdrant.Client` version.*
- Retrieval quality (§8.7–8.11) — the **BM25** algorithm, **bi-encoder vs cross-encoder**, reranking taxonomy, **MMR**, and query-side techniques (**multi-query, HyDE, contextual retrieval**) are established information-retrieval / RAG knowledge, not a single Qdrant page. The Qdrant-specific hooks used (sparse vectors, `Modifier.Idf`, ColBERT multivector rescoring) come from the crawled Collections/Indexing/Hybrid-Queries pages. *Ecosystem model names (FastEmbed `Qdrant/bm25`, miniCOIL, SPLADE, and specific cross-encoder rerankers like `bge-reranker-v2-m3`/Cohere/Jina) and the C# reranker/ONNX code are illustrative — verify current model ids and the hosted-API request shape against each provider's docs. Cross-encoder reranking runs in your app, not in Qdrant.*

> 🏷️ **Version note:** features are labelled with the Qdrant version that introduced them (e.g. TurboQuant v1.18, weighted RRF v1.17, formula queries v1.14). Confirm your server and `Qdrant.Client` versions support a feature before relying on it.
