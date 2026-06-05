# semantic-search-sdk

A .NET 8 class library for local semantic search over documents. Drop in `.txt` or `.pdf` files, and query them with natural language — no cloud API required.

Embeddings are generated locally via [Ollama](https://ollama.com) and ranked by dot product using `System.Numerics.Tensors` (equivalent to cosine similarity since `all-minilm` outputs unit-normalized vectors).

## Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download)
- [Ollama](https://ollama.com) running locally on `http://localhost:11434`
- The `all-minilm` model pulled:

  ```bash
  ollama pull all-minilm
  ```

## Quick Start

```bash
# Clone and build
git clone <repo-url>
cd semantic-search-sdk
dotnet build

# Run the interactive console app from a directory containing .txt or .pdf files
cd path/to/your/documents
dotnet run --project /path/to/src/SemanticSearchSdk.App

# Or run from the app directory (a sample.txt is included)
cd src/SemanticSearchSdk.App
dotnet run
```

The app indexes all `.txt` and `.pdf` files in the **current working directory**, then enters an interactive query loop:

```
Indexing 1 document(s) from C:\...\SemanticSearchSdk.App...
Ready.

Query: what is deep learning?

[92.3%] sample.txt:1
Deep learning is a type of machine learning that uses neural networks with many
layers to analyze data. It has been responsible for breakthroughs in image...
```

## Using the SDK

Add a reference to `src/SemanticSearchSdk` in your project and wire up the embedding generator:

```csharp
using Microsoft.Extensions.AI;
using OllamaSharp;
using SemanticSearchSdk;

IEmbeddingGenerator<string, Embedding<float>> generator =
    new OllamaApiClient(new Uri("http://localhost:11434"), "all-minilm");

var engine = new SemanticSearchEngine(generator);

// Index documents (replaces any previous index entirely)
await engine.IndexAsync(new[]
{
    ("doc1.txt", "The quick brown fox jumps over the lazy dog."),
    ("doc2.txt", "Vector embeddings capture semantic meaning in high-dimensional space."),
});

// Or load from a directory
var docs = DocumentLoader.LoadFromDirectory("/path/to/docs");
await engine.IndexAsync(docs);

// Query
var results = await engine.SearchAsync("semantic similarity", topK: 3);
foreach (var r in results)
    Console.WriteLine($"[{r.Similarity:P1}] {r.Label}");
```

### API Reference

**`SemanticSearchEngine(IEmbeddingGenerator<string, Embedding<float>> generator)`**

| Method | Description |
|---|---|
| `IndexAsync(IEnumerable<(string Label, string Content)>)` | Embeds all documents in one batch and builds the index. Calling again replaces the entire index. |
| `IndexAsync(IEnumerable<string>)` | Convenience overload — uses each string as both label and content. |
| `SearchAsync(string query, int topK = 3)` | Returns up to `topK` results ordered by descending similarity. |

**`SearchResult`** — `record(string Label, string Content, float Similarity)`

**`DocumentLoader.LoadFromDirectory(string directory)`** — Reads `.txt` and `.pdf` files (top-level only) from a directory. Each file is split on double newlines into paragraphs; labels are `filename:paragraphIndex` (e.g. `sample.txt:3`). PDFs with no blank lines between sections will produce a single chunk.

## Configuration

The console app reads its Ollama endpoint and model from `src/SemanticSearchSdk.App/appsettings.json`:

```json
{
  "Ollama": {
    "Url": "http://localhost:11434",
    "Model": "all-minilm"
  }
}
```

To switch models or point to a remote Ollama instance, edit this file.

## Running Tests

Tests use a deterministic `FakeEmbeddingGenerator` and **do not require Ollama**.

```bash
dotnet test
dotnet test --filter "FullyQualifiedName~SearchAsync_ReturnsTopKResults"
```

## Project Structure

```
SemanticSearchSdk.sln
├── src/SemanticSearchSdk/        # Core SDK — the primary deliverable
├── src/SemanticSearchSdk.App/    # Interactive console app (sample.txt included)
└── tests/SemanticSearchSdk.Tests # xUnit unit tests
```

## Tech Stack

| Package | Role |
|---|---|
| OllamaSharp v5.x | Ollama HTTP client — constructs the `IEmbeddingGenerator` |
| Microsoft.Extensions.AI v10.x | Provider-agnostic embedding abstraction |
| System.Numerics.Tensors v10.x | Span-based dot product math for ranking |
| UglyToad.PdfPig v1.7.x | PDF text extraction in `DocumentLoader` |
| xUnit v2.5 + coverlet | Test framework and coverage |
