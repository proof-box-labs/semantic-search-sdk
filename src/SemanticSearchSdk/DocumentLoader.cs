using UglyToad.PdfPig;

namespace SemanticSearchSdk;

public static class DocumentLoader
{
    public static IEnumerable<(string Label, string Content)> LoadFromDirectory(string directory)
    {
        foreach (var file in Directory.EnumerateFiles(directory, "*.txt"))
            yield return (Path.GetFileName(file), File.ReadAllText(file));

        foreach (var file in Directory.EnumerateFiles(directory, "*.pdf"))
            yield return (Path.GetFileName(file), ReadPdf(file));
    }

    private static string ReadPdf(string path)
    {
        using var doc = PdfDocument.Open(path);
        return string.Join("\n", doc.GetPages().Select(p => p.Text));
    }
}
