using Microsoft.Extensions.AI;

namespace SemanticSearchSdk.Tests;

public class SemanticSearchEngineTests
{
    // Returns a fixed embedding for each input string based on a simple deterministic mapping.
    // Each document gets a unique unit vector so cosine similarity is predictable in tests.
    private static IEmbeddingGenerator<string, Embedding<float>> BuildGenerator(params string[] knownInputs) =>
        new FakeEmbeddingGenerator(knownInputs);

    [Fact]
    public async Task SearchAsync_ReturnsTopKResults()
    {
        var docs = new[] { "cats", "dogs", "birds" };
        var engine = new SemanticSearchEngine(BuildGenerator(docs));
        await engine.IndexAsync(docs);

        // query matches "cats" — fake generator returns the same vector for identical input
        var results = (await engine.SearchAsync("cats", topK: 2)).ToList();

        Assert.Equal(2, results.Count);
        Assert.Equal("cats", results[0].Label);  // exact match should rank first
    }

    [Fact]
    public async Task SearchAsync_OrdersByDescendingSimilarity()
    {
        var docs = new[] { "cats", "dogs", "birds" };
        var engine = new SemanticSearchEngine(BuildGenerator(docs));
        await engine.IndexAsync(docs);

        var results = (await engine.SearchAsync("cats")).ToList();

        for (int i = 0; i < results.Count - 1; i++)
            Assert.True(results[i].Similarity >= results[i + 1].Similarity);
    }

    [Fact]
    public async Task SearchAsync_RespectsTopKLimit()
    {
        var docs = new[] { "a", "b", "c", "d", "e" };
        var engine = new SemanticSearchEngine(BuildGenerator(docs));
        await engine.IndexAsync(docs);

        var results = (await engine.SearchAsync("a", topK: 3)).ToList();

        Assert.Equal(3, results.Count);
    }

    [Fact]
    public async Task IndexAsync_LabeledOverload_PreservesLabels()
    {
        var docs = new (string Label, string Content)[]
        {
            ("file1.txt", "the cat sat on the mat"),
            ("file2.txt", "dogs love to fetch balls"),
        };
        var engine = new SemanticSearchEngine(BuildGenerator(docs.Select(d => d.Content).ToArray()));
        await engine.IndexAsync(docs);

        var results = (await engine.SearchAsync("the cat sat on the mat")).ToList();

        Assert.Equal("file1.txt", results[0].Label);
    }

    [Fact]
    public async Task IndexAsync_ReplacesExistingIndex()
    {
        var engine = new SemanticSearchEngine(BuildGenerator("first", "second", "third"));
        await engine.IndexAsync(new[] { "first", "second" });
        await engine.IndexAsync(new[] { "third" });

        var results = (await engine.SearchAsync("third")).ToList();

        Assert.Single(results);
        Assert.Equal("third", results[0].Label);
    }
}

/// <summary>
/// Deterministic fake: assigns each known input a unique orthogonal unit vector.
/// Unknown inputs (e.g. the query) get the vector of the first known input whose
/// string value equals the query, or a zero-like fallback.
/// </summary>
file sealed class FakeEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly Dictionary<string, float[]> _vectors;
    private readonly int _dims;

    public FakeEmbeddingGenerator(string[] knownInputs)
    {
        _dims = Math.Max(knownInputs.Length, 1);
        _vectors = new Dictionary<string, float[]>();
        for (int i = 0; i < knownInputs.Length; i++)
        {
            var v = new float[_dims];
            v[i] = 1f;
            _vectors[knownInputs[i]] = v;
        }
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var embeddings = values.Select(v =>
        {
            var vec = _vectors.TryGetValue(v, out var found) ? found : new float[_dims];
            return new Embedding<float>(vec);
        });
        return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(embeddings));
    }

    public EmbeddingGeneratorMetadata Metadata => new("fake", null, null);

    public TService? GetService<TService>(object? key = null) where TService : class => null;

    public object? GetService(Type serviceType, object? key = null) => null;

    public void Dispose() { }
}
