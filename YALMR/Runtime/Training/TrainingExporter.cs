using System.Text.Json;
using System.Text.Json.Serialization;

namespace YALMR.Runtime;

/// <summary>
/// Provides factory methods, batch export utilities, and shared helpers for training data export.
/// Use <see cref="GetExporter"/> to obtain a built-in <see cref="ITrainingExporter"/>,
/// or pass your own implementation to the <see cref="Export(Conversation, ITrainingExporter, TrainingExportOptions?)"/> overloads.
/// </summary>
public static class TrainingExporter
{
    internal static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    /// <summary>
    /// Returns the built-in <see cref="ITrainingExporter"/> for the specified format.
    /// </summary>
    public static ITrainingExporter GetExporter(TrainingFormat format) => format switch
    {
        TrainingFormat.OpenAI => OpenAITrainingExporter.Instance,
        TrainingFormat.ShareGPT => ShareGPTTrainingExporter.Instance,
        TrainingFormat.ChatML => ChatMLTrainingExporter.Instance,
        TrainingFormat.Alpaca => AlpacaTrainingExporter.Instance,
        _ => throw new ArgumentOutOfRangeException(nameof(format), $"Unsupported format: {format}")
    };

    // ── Export (string) ─────────────────────────────────────────────

    /// <summary>
    /// Exports a conversation using the format specified in <paramref name="options"/>.
    /// </summary>
    public static string Export(Conversation conversation, TrainingExportOptions? options = null)
    {
        options ??= new TrainingExportOptions();
        return GetExporter(options.Format).Export(conversation, options);
    }

    /// <summary>
    /// Exports a conversation using a custom <see cref="ITrainingExporter"/>.
    /// </summary>
    public static string Export(Conversation conversation, ITrainingExporter exporter, TrainingExportOptions? options = null)
    {
        options ??= new TrainingExportOptions();
        return exporter.Export(conversation, options);
    }

    /// <summary>
    /// Exports multiple conversations as a JSONL string (one or more lines per conversation).
    /// </summary>
    public static string Export(IEnumerable<Conversation> conversations, TrainingExportOptions? options = null)
    {
        options ??= new TrainingExportOptions();
        var exporter = GetExporter(options.Format);
        return string.Join("\n", conversations.Select(c => exporter.Export(c, options)));
    }

    /// <summary>
    /// Exports multiple conversations using a custom <see cref="ITrainingExporter"/>.
    /// </summary>
    public static string Export(IEnumerable<Conversation> conversations, ITrainingExporter exporter, TrainingExportOptions? options = null)
    {
        options ??= new TrainingExportOptions();
        return string.Join("\n", conversations.Select(c => exporter.Export(c, options)));
    }

    // ── Export (file) ───────────────────────────────────────────────

    /// <summary>
    /// Writes a conversation to a JSONL file and, when <see cref="ImageExportMode.FileReference"/>
    /// is active, saves referenced images alongside it.
    /// </summary>
    public static async Task ExportAsync(
        Conversation conversation, string path,
        TrainingExportOptions? options = null, CancellationToken ct = default)
    {
        options ??= new TrainingExportOptions();
        string jsonl = GetExporter(options.Format).Export(conversation, options);
        if (options.ImageMode == ImageExportMode.FileReference)
            await SaveReferencedImagesAsync(conversation, path, options, ct).ConfigureAwait(false);
        await File.WriteAllTextAsync(path, jsonl, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Writes multiple conversations to a single JSONL file and saves any referenced images.
    /// </summary>
    public static async Task ExportAsync(
        IEnumerable<Conversation> conversations, string path,
        TrainingExportOptions? options = null, CancellationToken ct = default)
    {
        options ??= new TrainingExportOptions();
        var exporter = GetExporter(options.Format);
        var lines = new List<string>();
        foreach (var conv in conversations)
        {
            lines.Add(exporter.Export(conv, options));
            if (options.ImageMode == ImageExportMode.FileReference)
                await SaveReferencedImagesAsync(conv, path, options, ct).ConfigureAwait(false);
        }
        await File.WriteAllTextAsync(path, string.Join("\n", lines), ct).ConfigureAwait(false);
    }

    // ── Shared helpers used by built-in exporters ───────────────────

    internal static bool ShouldIncludeMessage(ChatMessage msg, TrainingExportOptions options)
    {
        if (options.MessageFilter is not null && !options.MessageFilter(msg))
            return false;
        if (msg.Role == "system" && !options.IncludeSystemMessages)
            return false;
        if (msg.Role == "tool" && options.ToolCallMode == ToolCallExportMode.Omit)
            return false;
        // Skip assistant-only-tool-call messages when tool calls are omitted
        if (msg.Role == "assistant" && msg.ToolCalls is { Count: > 0 }
            && string.IsNullOrWhiteSpace(msg.Content)
            && msg.ReasoningContent is not { Length: > 0 }
            && options.ToolCallMode == ToolCallExportMode.Omit)
            return false;
        return true;
    }

    /// <summary>
    /// Extracts text content from a message, applying reasoning inline if configured.
    /// </summary>
    internal static string? GetTextContent(ChatMessage msg, TrainingExportOptions options)
    {
        string? content = msg.Content;

        if (string.IsNullOrWhiteSpace(content) && msg.Parts is { Count: > 0 })
            content = string.Join("\n", msg.Parts.OfType<TextPart>().Select(p => p.Text));

        if (msg.Role == "assistant"
            && msg.ReasoningContent is { Length: > 0 } reasoning
            && options.ReasoningMode == ReasoningExportMode.Inline)
        {
            content = $"<think>{reasoning}</think>{content}";
        }

        return content;
    }

    internal static string FormatToolCallsAsText(IReadOnlyList<ToolCall> toolCalls)
    {
        return string.Join("\n", toolCalls.Select(tc =>
            $"<tool_call>\n{{\"name\": \"{tc.Name}\", \"arguments\": {JsonSerializer.Serialize(tc.Arguments, JsonOptions)}}}\n</tool_call>"));
    }

    private static async Task SaveReferencedImagesAsync(
        Conversation conversation, string basePath, TrainingExportOptions options, CancellationToken ct)
    {
        string baseDir = Path.GetDirectoryName(basePath) ?? ".";
        string imageDir = Path.Combine(baseDir, options.ImageOutputDirectory);
        Directory.CreateDirectory(imageDir);

        int idx = 0;
        foreach (var msg in conversation)
        {
            if (!ShouldIncludeMessage(msg, options)) continue;
            if (msg.Parts is not { Count: > 0 }) continue;
            foreach (var part in msg.Parts)
            {
                if (part is ImagePart { Base64: { Length: > 0 } b64 })
                {
                    string imgPath = Path.Combine(imageDir, $"{conversation.Id}_img{idx++}.png");
                    await File.WriteAllBytesAsync(imgPath, Convert.FromBase64String(b64), ct).ConfigureAwait(false);
                }
            }
        }
    }
}
