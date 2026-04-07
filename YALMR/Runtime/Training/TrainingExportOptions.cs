namespace YALMR.Runtime;

// ── Training export configuration ───────────────────────────────────

/// <summary>Training data output format.</summary>
public enum TrainingFormat
{
    /// <summary>{"messages": [{"role": "...", "content": "..."}]} — OpenAI, Unsloth, TRL, most tools.</summary>
    OpenAI,
    /// <summary>{"conversations": [{"from": "human", "value": "..."}]} — Axolotl, LLaMA-Factory, FastChat.</summary>
    ShareGPT,
    /// <summary>{"text": "&lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;"} — raw ChatML for custom pipelines.</summary>
    ChatML,
    /// <summary>{"instruction": "...", "input": "...", "output": "..."} — single-turn, splits multi-turn into pairs.</summary>
    Alpaca
}

/// <summary>How model reasoning / thinking content is exported.</summary>
public enum ReasoningExportMode
{
    /// <summary>Strip reasoning content entirely (default).</summary>
    Omit,
    /// <summary>Wrap reasoning in &lt;think&gt; tags inline with content — DeepSeek R1 / Qwen3 convention.</summary>
    Inline,
    /// <summary>Emit as a separate "reasoning" JSON field (OpenAI format only).</summary>
    SeparateField
}

/// <summary>How tool-call rounds are exported.</summary>
public enum ToolCallExportMode
{
    /// <summary>Drop all tool-call and tool-result messages (default).</summary>
    Omit,
    /// <summary>Keep structured tool_calls arrays and tool role messages — OpenAI finetuning format.</summary>
    Structured,
    /// <summary>Flatten tool calls / results into text with &lt;tool_call&gt; / &lt;tool_response&gt; tags.</summary>
    Inline
}

/// <summary>How images in multimodal messages are exported.</summary>
public enum ImageExportMode
{
    /// <summary>Strip all image content (default — text-only training).</summary>
    Omit,
    /// <summary>Embed as data:image/png;base64,... URLs in OpenAI vision content-parts format.</summary>
    InlineBase64,
    /// <summary>Save to files and reference by relative path. Use with <see cref="TrainingExporter.ExportAsync"/>.</summary>
    FileReference,
    /// <summary>Replace each image with a [image] text placeholder.</summary>
    Placeholder
}

/// <summary>
/// Controls how conversations are exported for training / finetuning.
/// </summary>
public sealed record TrainingExportOptions
{
    /// <summary>Output format. Default: <see cref="TrainingFormat.OpenAI"/>.</summary>
    public TrainingFormat Format { get; init; } = TrainingFormat.OpenAI;

    /// <summary>How to handle model reasoning / thinking content.</summary>
    public ReasoningExportMode ReasoningMode { get; init; } = ReasoningExportMode.Omit;

    /// <summary>How to handle tool-call rounds.</summary>
    public ToolCallExportMode ToolCallMode { get; init; } = ToolCallExportMode.Omit;

    /// <summary>How to handle images in multimodal messages.</summary>
    public ImageExportMode ImageMode { get; init; } = ImageExportMode.Omit;

    /// <summary>
    /// Relative directory for saved images when <see cref="ImageMode"/> is
    /// <see cref="ImageExportMode.FileReference"/>. Default: <c>"images"</c>.
    /// </summary>
    public string ImageOutputDirectory { get; init; } = "images";

    /// <summary>Whether to include system-role messages in the output. Default: <c>true</c>.</summary>
    public bool IncludeSystemMessages { get; init; } = true;

    /// <summary>Optional per-message filter. Return <c>false</c> to exclude a message.</summary>
    public Func<ChatMessage, bool>? MessageFilter { get; init; }
}
