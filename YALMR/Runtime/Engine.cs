using System.Runtime.CompilerServices;
using System.Text;
using YALMR.Diagnostics;
using YALMR.LlamaCpp;

namespace YALMR.Runtime;

/// <summary>
/// A single token produced during inference.
/// </summary>
public readonly record struct InferenceToken(int Id, string Text, bool IsEndOfGeneration);

/// <summary>
/// Shared model host that owns the native llama model and optional vision context.
/// Per-session inference state is managed by <see cref="InferenceContext"/> instances
/// created via <see cref="CreateInferenceContext"/>.
/// </summary>
public sealed class Engine : IAsyncDisposable, IDisposable
{
    private readonly SessionOptions _options;
    private readonly SemaphoreSlim _visionLock = new(1, 1);
    private readonly int _nBatch;

    private readonly string? _template;
    private readonly string? _bosToken;
    private readonly string _imageToken;

    private Llama.Model _model;
    private Llama.Vision.Context _visionContext;
    private bool _disposed;

    public bool VisionEnabled => !_visionContext.IsNull;
    public string? VisionDisabledReason { get; }
    public string ImageToken => _imageToken;

    /// <summary>
    /// True when the model's chat template contains thinking/reasoning markers
    /// (<c>&lt;think&gt;</c> tags or <c>enable_thinking</c> variable).
    /// </summary>
    public bool ThinkingEnabled => DetectThinkingSupport(_template);

    /// <summary>
    /// True when the loaded model has no chat template and can only produce embeddings.
    /// </summary>
    public bool IsEmbeddingOnly => _template is null;

    private Engine(
        SessionOptions options,
        Llama.Model model,
        Llama.Vision.Context visionContext,
        string? template,
        string? bosToken,
        string? visionDisabledReason,
        int nBatch)
    {
        _options = options;
        _model = model;
        _visionContext = visionContext;
        _template = template;
        _bosToken = bosToken;
        _imageToken = InferImageToken(template);
        _nBatch = nBatch;
        VisionDisabledReason = visionDisabledReason;
    }

    /// <summary>
    /// Creates and initializes an engine from the provided options.
    /// </summary>
    public static Task<Engine> CreateAsync(SessionOptions options, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        string path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        if (!path.Contains(options.BackendDirectory))
            Environment.SetEnvironmentVariable("PATH", $"{path};{options.BackendDirectory}");

        return Task.Run(() =>
        {
            Llama.Init(options.BackendDirectory, options.Logger);

            var metadata = GgufReader.ReadMetadata(options.ModelPath);
            string? template = GgufReader.GetString(metadata, "tokenizer.chat_template");
            string? bosToken = GgufReader.ResolveTokenById(metadata, "tokenizer.ggml.bos_token_id");

            var model = Llama.LoadModel(
                options.ModelPath,
                useMmap: options.UseMmap,
                useMlock: options.UseMlock,
                checkTensors: options.CheckTensors,
                nGpuLayers: options.GpuLayers);

            // Resolve vision projector path before context creation so we can
            // size n_batch for the full multimodal prompt. When several images
            // are attached to one request, mtmd can produce far more positions
            // than a single-image token budget, so using the full context budget
            // avoids later images being dropped during prompt evaluation.
            // Embedding-only models (no chat template) never use vision.
            string mmprojPath = template is not null
                ? ResolveMmprojPath(options.ModelPath, options.MmprojPath)
                : string.Empty;
            int nBatch = !string.IsNullOrEmpty(mmprojPath)
                ? Math.Max(options.BatchTokens, options.VisionImageMaxTokens)
                : options.BatchTokens;

            Llama.Vision.Context visionCtx = default;
            string? visionDisabledReason = null;

            if (!string.IsNullOrEmpty(mmprojPath))
            {
                try
                {
                    visionCtx = Llama.Vision.Load(
                        model,
                        mmprojPath,
                        useGpu: options.UseGpuForVision,
                        nThreads: options.VisionThreads > 0 ? options.VisionThreads : Environment.ProcessorCount,
                        mediaMarker: Llama.Vision.DefaultMarker,
                        warmup: false,
                        imageMinTokens: options.VisionImageMinTokens,
                        imageMaxTokens: options.VisionImageMaxTokens);
                }
                catch (Exception ex)
                {
                    visionDisabledReason = ex.Message;
                }
            }

            return new Engine(options, model, visionCtx, template, bosToken, visionDisabledReason, nBatch);
        }, ct);
    }

    /// <summary>
    /// Tokenizes text using the loaded model vocabulary.
    /// </summary>
    public int[] Tokenize(string text) => Llama.Tokenize(_model, text);

    /// <summary>
    /// Converts a token ID to its string representation.
    /// </summary>
    public string TokenToString(int token) => Llama.TokenToString(_model, token);

    /// <summary>
    /// Writes the raw UTF-8 bytes for a token into <paramref name="buf"/> and returns the byte count.
    /// </summary>
    public int TokenToBytes(int token, byte[] buf) => Llama.TokenToBytes(_model, token, buf);

    /// <summary>
    /// Checks whether a token signals end of generation.
    /// </summary>
    public bool IsEndOfGeneration(int token) => Llama.IsEndOfGeneration(_model, token);

    /// <summary>
    /// Renders a chat prompt from messages and request settings using the model's Jinja template.
    /// This method is stateless and does not acquire the engine lock.
    /// </summary>
    public string RenderPrompt(IReadOnlyList<ChatMessage> messages, InferenceOptions options)
    {
        if (_template is null)
            throw new InvalidOperationException("Cannot render a chat prompt: the loaded model has no chat template (embedding-only model).");

        ValidateMessages(messages);
        var ctx = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["messages"] = messages.Select(BuildTemplateMessage).Cast<object?>().ToList(),
            ["add_generation_prompt"] = true,
            ["enable_thinking"] = options.EnableThinking ?? true,
            ["add_vision_id"] = options.AddVisionId,
            ["tools"] = options.Tools?.Select(BuildPromptTool).Cast<object?>().ToList()
        };
        if (!string.IsNullOrEmpty(_bosToken)) ctx["bos_token"] = _bosToken;

        return MiniJinjaChatTemplate.Render(_template, ctx);
    }

    /// <summary>
    /// Gets the native model handle for direct llama.cpp interop.
    /// </summary>
    internal Llama.Model NativeModel => _model;

    /// <summary>
    /// Creates a new per-session inference context with its own native llama context and KV cache.
    /// </summary>
    public InferenceContext CreateInferenceContext(SessionOptions options)
    {
        var context = Llama.CreateContext(
            _model,
            nCtx: options.ContextTokens,
            nBatch: _nBatch,
            nUbatch: options.MicroBatchTokens,
            nThreads: options.Threads,
            embeddings: true,
            unifiedKvCache: options.UnifiedKvCache,
            ropeFreqBase: options.RopeFrequencyBase,
            ropeFreqScale: options.RopeFrequencyScale,
            offloadKvCacheToGpu: options.OffloadKvCacheToGpu,
            flashAttention: options.FlashAttention,
            kvCacheTypeK: options.KvCacheTypeK,
            kvCacheTypeV: options.KvCacheTypeV);

        var defaultInference = options.DefaultInference ?? new InferenceOptions();
        var random = defaultInference.Seed is int seed ? new Random(seed) : Random.Shared;

        return new InferenceContext(this, options, context, _nBatch, random);
    }

    /// <summary>
    /// Evaluates a multimodal prompt through the shared vision context and returns the new KV
    /// cache position (<c>nPast</c>) after evaluation.
    /// Pass <paramref name="startNPast"/> greater than zero to append to an existing KV cache
    /// (incremental mode); pass <paramref name="addSpecial"/><c>=false</c> when the prompt is
    /// a continuation that must not receive a BOS token.
    /// Serialized with an internal lock so concurrent sessions can safely share the vision pipeline.
    /// </summary>
    internal async Task<int> EvalVisionPromptAsync(
        Llama.Context llamaCtx,
        string mtmdPrompt,
        IReadOnlyList<string> imageBase64s,
        int nBatch,
        int startNPast = 0,
        bool addSpecial = true,
        CancellationToken ct = default)
    {
        await _visionLock.WaitAsync(ct);
        try
        {
            int newNPast = startNPast;
            await Task.Run(() =>
            {
                int nPast = startNPast;
                Llama.Vision.EvalPromptWithBase64Images(
                    _visionContext, llamaCtx, mtmdPrompt, imageBase64s,
                    ref nPast, nBatch: nBatch, addSpecial: addSpecial);
                newNPast = nPast;
            }, ct);
            return newNPast;
        }
        finally
        {
            _visionLock.Release();
        }
    }

    private static string ResolveMmprojPath(string modelPath, string? mmprojPath)
    {
        if (!string.IsNullOrWhiteSpace(mmprojPath)) return mmprojPath;
        return Directory.GetFiles(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", "*.gguf")
            .FirstOrDefault(f => Path.GetFileName(f).Contains("mmproj", StringComparison.OrdinalIgnoreCase)) ?? string.Empty;
    }

    private static string InferImageToken(string? template)
    {
        if (template is null) return string.Empty;

        const string qwenVisionToken = "<|vision_start|><|image_pad|><|vision_end|>";
        if (template.Contains(qwenVisionToken, StringComparison.Ordinal))
            return qwenVisionToken;

        // LightOnOCR-2 and similar models use a standalone <|image_pad|> token.
        const string imagePadToken = "<|image_pad|>";
        if (template.Contains(imagePadToken, StringComparison.Ordinal))
            return imagePadToken;

        // InternVL, LLaVA, InternLM, LFS-VL and similar models use a plain <image> tag.
        if (template.Contains("<image>", StringComparison.Ordinal))
            return "<image>";

        // Phi-3/4 vision models use <|image|> as the placeholder.
        if (template.Contains("<|image|>", StringComparison.Ordinal))
            return "<|image|>";

        return string.Empty;
    }

    /// <summary>
    /// Heuristic: a template supports thinking when it references &lt;think&gt; tags or
    /// the <c>enable_thinking</c> variable used by Qwen3, QwQ, DeepSeek-R1, etc.
    /// </summary>
    private static bool DetectThinkingSupport(string? template) =>
        template is not null &&
        (template.Contains("<think>", StringComparison.Ordinal) ||
         template.Contains("enable_thinking", StringComparison.Ordinal));

    private static void ValidateMessages(IReadOnlyList<ChatMessage> messages)
    {
        for (int i = 1; i < messages.Count; i++)
            if (messages[i].Role == "system") throw new InvalidOperationException("System message must be first.");
    }

    internal static object BuildPromptTool(ToolDefinition tool) => new
    {
        name = tool.Name,
        description = tool.Description,
        parameters = tool.Parameters ?? new { type = "object" }
    };

    internal static Dictionary<string, object?> BuildTemplateMessage(ChatMessage message)
    {
        var result = new Dictionary<string, object?>(StringComparer.Ordinal) { ["role"] = message.Role };
        object? content = BuildTemplateContent(message);
        if (content is not null) result["content"] = content;
        if (!string.IsNullOrWhiteSpace(message.ToolCallId)) result["call_id"] = message.ToolCallId;
        if (!string.IsNullOrWhiteSpace(message.ReasoningContent)) result["reasoning_content"] = message.ReasoningContent;
        if (message.ToolCalls is { Count: > 0 })
            result["tool_calls"] = message.ToolCalls.Select(BuildTemplateToolCall).Cast<object?>().ToList();
        return result;
    }

    private static object? BuildTemplateContent(ChatMessage message)
    {
        if (message.Parts is not { Count: > 0 }) return message.Content;
        return message.Parts.Select(p => p switch
        {
            TextPart t => new Dictionary<string, object?>(StringComparer.Ordinal) { ["type"] = "text", ["text"] = t.Text },
            ImagePart => new Dictionary<string, object?>(StringComparer.Ordinal) { ["type"] = "image", ["image"] = true },
            _ => (object?)null
        }).Where(x => x != null).ToList();
    }

    private static Dictionary<string, object?> BuildTemplateToolCall(ToolCall toolCall)
    {
        object? args = ConvertTemplateValue(toolCall.Arguments);
        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["id"] = toolCall.CallId,
            ["call_id"] = toolCall.CallId,
            ["name"] = toolCall.Name,
            ["arguments"] = args,
            ["function"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["name"] = toolCall.Name,
                ["arguments"] = args,
                ["id"] = toolCall.CallId,
                ["call_id"] = toolCall.CallId
            }
        };
    }

    private static object? ConvertTemplateValue(object? value)
    {
        if (value is null) return null;
        if (value is System.Text.Json.JsonElement element) return element.Clone();
        if (value is IDictionary<string, object?> dict)
        {
            var d = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var kv in dict) d[kv.Key] = ConvertTemplateValue(kv.Value);
            return d;
        }
        if (value is System.Collections.IEnumerable enumerable && value is not string)
        {
            return enumerable.Cast<object>().Select(ConvertTemplateValue).ToList();
        }
        return value;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        await _visionLock.WaitAsync();
        try
        {
            if (_disposed) return;
            await Task.Run(() =>
            {
                if (!_visionContext.IsNull) Llama.Vision.Free(_visionContext);
                if (!_model.IsNull) Llama.FreeModel(_model);
                Llama.Shutdown();
            });
            _disposed = true;
        }
        finally { _visionLock.Release(); _visionLock.Dispose(); }
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();
}
