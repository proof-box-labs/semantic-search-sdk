using Microsoft.Extensions.AI;
using System.Numerics.Tensors;

namespace SemanticSearchSdk;

public class SemanticSearchEngine(IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator)
{
    private List<(string Label, string Content, Embedding<float> Embedding)> _index = [];

    public async Task IndexAsync(IEnumerable<(string Label, string Content)> documents)
    {
        var docs = documents.ToList();
        var embeddings = (await embeddingGenerator.GenerateAndZipAsync(docs.Select(d => d.Content))).ToList();
        _index = [.. docs.Zip(embeddings, (doc, emb) => (doc.Label, doc.Content, emb.Embedding))];
    }

    public Task IndexAsync(IEnumerable<string> documents) =>
        IndexAsync(documents.Select(d => (d, d)));

    public async Task<IEnumerable<SearchResult>> SearchAsync(string query, int topK = 3)
    {
        var queryEmbedding = await embeddingGenerator.GenerateAsync(query);
        return _index
            .Select(item => new SearchResult(
                item.Label,
                item.Content,
                TensorPrimitives.CosineSimilarity(item.Embedding.Vector.Span, queryEmbedding.Vector.Span)))
            .OrderByDescending(r => r.Similarity)
            .Take(topK);
    }
}

public record SearchResult(string Label, string Content, float Similarity);
