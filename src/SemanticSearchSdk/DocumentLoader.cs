using UglyToad.PdfPig;

namespace SemanticSearchSdk;

public static class DocumentLoader
{
    public static IEnumerable<(string Label, string Content)> LoadFromDirectory(string directory)
    {
        foreach (var file in Directory.EnumerateFiles(directory, "*.txt"))
            foreach (var chunk in ChunkByParagraph(File.ReadAllText(file), Path.GetFileName(file)))
                yield return chunk;

        foreach (var file in Directory.EnumerateFiles(directory, "*.pdf"))
            foreach (var chunk in ChunkByParagraph(ReadPdf(file), Path.GetFileName(file)))
                yield return chunk;
    }

    private static IEnumerable<(string Label, string Content)> ChunkByParagraph(string text, string fileName)
    {
        var paragraphs = text.Split(["\r\n\r\n", "\n\n"], StringSplitOptions.RemoveEmptyEntries);
        var index = 0;
        foreach (var paragraph in paragraphs)
        {
            var trimmed = paragraph.Trim();
            if (!string.IsNullOrWhiteSpace(trimmed))
                yield return ($"{fileName}:{++index}", trimmed);
        }
    }

    private static string ReadPdf(string path)
    {
        using var doc = PdfDocument.Open(path);
        return string.Join("\n", doc.GetPages().Select(p => p.Text));
    }
}
