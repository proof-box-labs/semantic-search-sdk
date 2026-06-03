# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build entire solution
dotnet build

# Run the console app
dotnet run --project src/SemanticSearchSdk.App

# Run all tests
dotnet test

# Run a single test by name
dotnet test --filter "FullyQualifiedName~TestMethodName"

# Run tests in a specific project
dotnet test tests/SemanticSearchSdk.Tests

# Restore packages
dotnet restore
```

## Architecture

The solution has three projects:

```
SemanticSearchSdk.sln
├── src/SemanticSearchSdk/          # Core SDK class library — the primary deliverable
├── src/SemanticSearchSdk.App/      # Console app — wires up and exercises the SDK
└── tests/SemanticSearchSdk.Tests/  # xUnit tests against the SDK library
```

**Dependency flow:** `App` → `SemanticSearchSdk` ← `Tests`

Application logic belongs in the library, not in `Program.cs`. The console app and tests both depend on the library; tests do not go through the console app.

## Semantic Search Design Intent

The SDK is built around the **embed → store → query** pattern:

1. **Embed** — text is converted to dense `float` vectors via `IEmbeddingGenerator<string, Embedding<float>>` (from `Microsoft.Extensions.AI`), backed at runtime by an `OllamaApiClient` pointed at `http://localhost:11434` using the `all-minilm` model.
2. **Store** — embeddings are held in memory alongside their source documents.
3. **Query** — a query string is embedded the same way, then all stored vectors are ranked by cosine similarity using `TensorPrimitives.CosineSimilarity()` from `System.Numerics.Tensors`.

The reference implementation of this full flow lives in `src/SemanticSearchSdk.App/Program.cs`. As the library grows, that logic should be extracted into the SDK as reusable abstractions.

## Core SDK API

**`SemanticSearchEngine`** — constructor-injected with `IEmbeddingGenerator<string, Embedding<float>>`:

- `IndexAsync(IEnumerable<(string Label, string Content)> documents)` — embeds all documents and replaces the entire in-memory index. **Calling it again discards the previous index entirely** — there is no incremental append.
- `IndexAsync(IEnumerable<string> documents)` — convenience overload; uses each document as its own label.
- `SearchAsync(string query, int topK = 3)` — embeds the query and returns up to `topK` results sorted by descending cosine similarity.

**`SearchResult`** — `record(string Label, string Content, float Similarity)`

**`DocumentLoader`** — static utility; reads `.txt` files (plain text) and `.pdf` files (via `UglyToad.PdfPig`) from a directory, returning `IEnumerable<(string Label, string Content)>` where Label is the filename.

## Testing Pattern

Tests use a `FakeEmbeddingGenerator` (file-scoped class in `SemanticSearchEngineTests.cs`) that maps known input strings to deterministic orthogonal unit vectors. This makes similarity scores predictable without a live Ollama instance. When adding new tests, follow the same pattern: call `BuildGenerator(params string[] knownInputs)` to create a fake, then pass it to `SemanticSearchEngine`.

## Key Constraints

- **Always code against `IEmbeddingGenerator<string, Embedding<float>>`**, never against `OllamaApiClient` directly. This is what allows tests to inject a fake without a live Ollama process.
- **Runtime dependency:** Ollama must be installed and running on `http://localhost:11434` before the app or any integration tests execute. This is not enforced at build time.
- **Similarity math:** Use `TensorPrimitives` (span-based) for vector operations — avoid pulling in external ML frameworks.

## Tech Stack

- **.NET 8.0**, C# 12, nullable reference types and implicit usings enabled across all projects
- **OllamaSharp** (v5.x) — C# client for Ollama; used only to construct the `IEmbeddingGenerator` implementation
- **Microsoft.Extensions.AI** (v10.x) — provider-agnostic AI abstractions; the interfaces the SDK and tests code against
- **System.Numerics.Tensors** (v10.x) — span-based tensor/vector math for similarity scoring
- **UglyToad.PdfPig** (v1.7.x) — PDF text extraction; used only in `DocumentLoader`
- **xUnit** (v2.5) + **coverlet** — test framework and coverage collection
