using Microsoft.Extensions.AI;
using OllamaSharp;
using SemanticSearchSdk;

IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
    new OllamaApiClient(new Uri("http://localhost:11434"), "all-minilm");

var directory = Directory.GetCurrentDirectory();
var documents = DocumentLoader.LoadFromDirectory(directory).ToList();

if (documents.Count == 0)
{
    Console.WriteLine($"No .txt or .pdf files found in: {directory}");
    return;
}

Console.WriteLine($"Indexing {documents.Count} document(s) from {directory}...");
var engine = new SemanticSearchEngine(embeddingGenerator);
await engine.IndexAsync(documents);
Console.WriteLine("Ready.\n");

while (true)
{
    Console.Write("Query: ");
    var userInput = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(userInput)) continue;

    var results = await engine.SearchAsync(userInput);
    foreach (var result in results)
    {
        var snippet = result.Content.Length > 200 ? result.Content[..200] + "…" : result.Content;
        Console.WriteLine($"\n[{result.Similarity:P1}] {result.Label}");
        Console.WriteLine(snippet);
    }
    Console.WriteLine();
}
