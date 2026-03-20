using System.IO;
using YALMR.Diagnostics;
using YALMR.LlamaCpp;
using YALMR.Utils;

namespace YALMR.Runtime;

// ── Multimodal content ──────────────────────────────────────────────

public abstract record ContentPart;
public sealed record TextPart(string Text) : ContentPart;

public sealed record ImagePart(string? Base64 = null) : ContentPart
{
    public static ImagePart FromFile(string path) => new(Convert.ToBase64String(File.ReadAllBytes(path)));
    public static ImagePart FromBytes(byte[] data) => new(Convert.ToBase64String(data));
}

// ── Core message types ──────────────────────────────────────────────

public sealed record InferenceUsage(int PromptTokens, int CompletionTokens, int TotalTokens)
{
    public InferenceUsage(int PromptTokens, int CompletionTokens)
        : this(PromptTokens, CompletionTokens, PromptTokens + CompletionTokens) { }
}

public sealed record ChatMessage(
    string Role,
    string? Content = null,
    IReadOnlyList<ContentPart>? Parts = null,
    string? ReasoningContent = null,
    IReadOnlyList<ToolCall>? ToolCalls = null,
    string? RawContent = null,
    InferenceUsage? Usage = null,
    string? ToolCallId = null
);

public sealed record ToolCall(
    string Name,
    IReadOnlyDictionary<string, object?> Arguments,
    string CallId
)
{
    public ToolCall(string Name, IReadOnlyDictionary<string, object?> Arguments)
        : this(Name, Arguments, $"call_{Guid.NewGuid():N}") { }
}

public sealed record ToolCallResult(
    string CallId,
    string Name,
    IReadOnlyDictionary<string, object?> Arguments,
    string Result
);

// ── Tool definitions ────────────────────────────────────────────────

public sealed record ToolDefinition(
    string Type,
    string Name,
    object? Parameters = null,
    string? Description = null,
    string? ServerLabel = null,
    string? ServerUrl = null,
    IReadOnlyList<string>? AllowedTools = null,
    IReadOnlyDictionary<string, string>? Headers = null
);

// ── Inference options (sampling + prompt rendering) ─────────────────

public sealed record InferenceOptions
{
    public IReadOnlyList<ToolDefinition>? Tools { get; init; }
    public float? Temperature { get; init; } = 0.0f;
    public float? TopP { get; init; } = 1.0f;
    public int? TopK { get; init; } = 0;
    public float? PresencePenalty { get; init; } = 0.0f;
    public float? FrequencyPenalty { get; init; } = 0.0f;
    public float? RepetitionPenalty { get; init; } = 1.0f;
    public bool? EnableThinking { get; init; } = true;
    public string? ReasoningEffort { get; init; }
    public int? MaxOutputTokens { get; init; } = 512;
    public int? Seed { get; init; }
    public bool AddVisionId { get; init; } = false;
}

// ── Request / Response ──────────────────────────────────────────────

public sealed record ResponseRequest
{
    public string Model { get; init; } = string.Empty;
    public IReadOnlyList<ChatMessage> Input { get; init; } = [];
    public string? PreviousResponseId { get; init; }
    public InferenceOptions Inference { get; init; } = new();
}

public sealed record ResponseObject(
    string Id,
    long CreatedAt,
    string Model,
    IReadOnlyList<ChatMessage> Output,
    InferenceUsage Usage,
    string? PreviousResponseId = null
);

// ── Enums colocated with their primary consumer ─────────────────────

public enum ImageRetentionPolicy { KeepAllImages, KeepLatestImage }

// ── Session configuration ───────────────────────────────────────────

public sealed record SessionOptions
{
    public string BackendDirectory { get; init; } = string.Empty;
    public required string ModelPath { get; init; }
    public required ToolRegistry ToolRegistry { get; init; }
    public required ConversationCompactionOptions Compaction { get; init; }
    public InferenceOptions? DefaultInference { get; init; }
    public IConversationCompactor? ConversationCompactor { get; init; }
    public IToolExecutionEngine? ToolExecutionEngine { get; init; }
    public ILogger? Logger { get; init; }
    public int ContextTokens { get; init; } = 8192;
    public int ResetContextTokens { get; init; } = 2048;
    public int BatchTokens { get; init; } = 1024;
    public int MicroBatchTokens { get; init; } = 1024;
    public int? Threads { get; init; }
    public int MaxToolRounds { get; init; } = 10;
    public string? MmprojPath { get; init; }
    public ImageRetentionPolicy ImageRetentionPolicy { get; init; } = ImageRetentionPolicy.KeepAllImages;
    public bool UseGpuForVision { get; init; } = true;
    public int VisionThreads { get; init; } = 0;
    public int VisionImageMinTokens { get; init; } = 1024;
    public int VisionImageMaxTokens { get; init; } = 1024;
    public bool UnifiedKvCache { get; init; } = false;
    public float? RopeFrequencyBase { get; init; }
    public float? RopeFrequencyScale { get; init; }
    public bool OffloadKvCacheToGpu { get; init; } = true;
    public int GpuLayers { get; init; } = 0;
    public bool UseMmap { get; init; } = true;
    public bool UseMlock { get; init; } = false;
    public bool CheckTensors { get; init; } = false;
    public bool FlashAttention { get; init; } = false;
    public KvCacheQuantization? KvCacheTypeK { get; init; }
    public KvCacheQuantization? KvCacheTypeV { get; init; }
}
