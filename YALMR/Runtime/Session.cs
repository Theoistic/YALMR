using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using YALMR.Diagnostics;
using YALMR.LlamaCpp;
using YALMR.Mcp;
using YALMR.Transport;
using YALMR.Utils;

namespace YALMR.Runtime;

/// <summary>
/// One streamed chunk of assistant output.
/// </summary>
public sealed record ChatResponseChunk(
    string? Text = null,
    string? ReasoningText = null,
    IReadOnlyList<ToolCall>? ToolCalls = null,
    InferenceUsage? Usage = null
);

/// <summary>
/// Captures the prompt-state view seen by the session for debugging.
/// </summary>
public sealed record SessionDebugView(
    IReadOnlyList<ChatMessage> History,
    IReadOnlyList<ChatMessage> PromptMessages,
    string RenderedPrompt,
    IReadOnlyList<object?>? Tools,
    int PromptTokens
);

/// <summary>
/// Maintains conversation state, prompt compaction, tool execution, and generation.
/// Each session owns a dedicated <see cref="InferenceContext"/> for parallel execution.
/// </summary>
public sealed class Session : IAsyncDisposable, IDisposable
{
    private sealed record ResponseExecutionContext(
        InferenceOptions Inference,
        IReadOnlyDictionary<string, RemoteToolBinding> RemoteTools);

    private readonly SessionOptions _options;
    private readonly Engine _engine;
    private readonly InferenceContext _inferenceContext;
    private readonly List<ChatMessage> _history = [];
    private readonly Dictionary<string, IReadOnlyList<ChatMessage>> _responseHistories = new(StringComparer.Ordinal);

    private readonly InferenceOptions _defaultInference;
    private readonly ConversationCompactionOptions _compaction;
    private readonly IConversationCompactor _conversationCompactor;
    private readonly IToolExecutionEngine _toolExecutionEngine;
    private readonly bool _ownsEngine;

    private bool _disposed;

    // Tracks which session is active on the current async call chain so that
    // nested tool calls (e.g. ScanPdfPage → lm.RespondAsync) can detect
    // re-entrancy and save/restore history instead of corrupting the outer call.
    private static readonly AsyncLocal<Session?> s_reentrancyOwner = new();

    public bool VisionEnabled => _engine.VisionEnabled;
    public string? VisionDisabledReason => _engine.VisionDisabledReason;
    public IReadOnlyList<ChatMessage> History => _history;
    public ResponseObject? LastResponse { get; private set; }

    /// <summary>
    /// Gets the underlying inference engine used by this session.
    /// </summary>
    public Engine Engine => _engine;

    /// <summary>
    /// Raised whenever the session prepares a prompt for model execution.
    /// </summary>
    public event EventHandler<SessionDebugView>? DebugViewCreated;

    private Session(SessionOptions options, Engine engine, bool ownsEngine)
    {
        _options = options;
        _engine = engine;
        _inferenceContext = engine.CreateInferenceContext(options);
        _ownsEngine = ownsEngine;
        _defaultInference = options.DefaultInference ?? new InferenceOptions();
        _compaction = options.Compaction;
        _conversationCompactor = options.ConversationCompactor ?? new TokenWindowConversationCompactor();
        _toolExecutionEngine = options.ToolExecutionEngine ?? new DefaultToolExecutionEngine([new McpRemoteToolExecutor()]);
    }

    /// <summary>
    /// Creates and initializes a session from the provided options.
    /// </summary>
    public static async Task<Session> CreateAsync(SessionOptions options, CancellationToken ct = default)
    {
        using var activity = RuntimeTelemetry.StartActivity("runtime.session.create");
        RuntimeTelemetry.SessionsCreated.Add(1);
        try
        {
            activity?.SetTag("yalmr.model_path", options.ModelPath);
            var engine = await Engine.CreateAsync(options, ct);
            return new Session(options, engine, ownsEngine: true);
        }
        catch (Exception ex)
        {
            RuntimeTelemetry.RecordException(activity, ex);
            throw;
        }
    }

    /// <summary>
    /// Creates a session that shares an existing engine instance.
    /// </summary>
    public static Session Create(SessionOptions options, Engine engine)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(engine);
        return new Session(options, engine, ownsEngine: false);
    }

    /// <summary>
    /// Clears session history and response state.
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();
        ResetCore(clearResponses: true);
    }

    /// <summary>
    /// Adds a message directly to the session history without triggering generation.
    /// Useful for seeding the conversation with a system prompt or prior context.
    /// </summary>
    public void PrimeHistory(ChatMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        ThrowIfDisposed();
        _history.Add(message);
    }

    private void ResetCore(bool clearResponses)
    {
        _history.Clear();
        if (clearResponses)
        {
            _responseHistories.Clear();
            LastResponse = null;
        }
    }

    /// <summary>
    /// Sends a single message and returns the final assistant message.
    /// </summary>
    public async Task<ChatMessage> SendAsync(ChatMessage message, CancellationToken ct = default)
    {
        await foreach (var _ in GenerateAsync(message, ct).ConfigureAwait(false))
        {
        }

        for (int i = _history.Count - 1; i >= 0; i--)
            if (_history[i].Role == "assistant")
                return _history[i];

        throw new InvalidOperationException("No assistant message was generated.");
    }

    /// <summary>
    /// Handles a response-style request and returns a response-style object.
    /// </summary>
    public async Task<ResponseObject> CreateResponseAsync(ResponseRequest request, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Input is not { Count: > 0 })
            throw new InvalidOperationException("Response requests must contain at least one input item.");

        ThrowIfDisposed();

        using var activity = RuntimeTelemetry.StartActivity("runtime.session.create_response");
        RuntimeTelemetry.InferenceCalls.Add(1);
        var sw = Stopwatch.StartNew();
        try
        {
            activity?.SetTag("gen_ai.request.model", request.Model);

            bool isReentrant = s_reentrancyOwner.Value == this;
            List<ChatMessage>? savedHistory = null;

            if (isReentrant)
                savedHistory = SaveHistorySnapshot();
            else
                s_reentrancyOwner.Value = this;

            try
            {
                var responseHistory = BuildResponseHistory(request);
                if (responseHistory.Count == 0)
                    throw new InvalidOperationException("Response request did not produce any renderable history.");

                int historyPrefixCount = Math.Max(0, responseHistory.Count - 1);

                ResetCore(clearResponses: false);
                _history.AddRange(responseHistory.Take(historyPrefixCount));

                var executionContext = await CreateResponseExecutionContextAsync(request.Inference, ct).ConfigureAwait(false);

                await foreach (var _ in GenerateCoreAsync(responseHistory[^1], executionContext, ct).ConfigureAwait(false))
                {
                }

                var result = FinalizeResponse(request.Model, request.PreviousResponseId, historyPrefixCount);
                activity?.SetTag("gen_ai.usage.input_tokens", result.Usage.PromptTokens);
                activity?.SetTag("gen_ai.usage.output_tokens", result.Usage.CompletionTokens);
                return result;
            }
            finally
            {
                if (isReentrant)
                    RestoreAfterReentrantCall(savedHistory!);
                else
                    s_reentrancyOwner.Value = null;
            }
        }
        catch (Exception ex)
        {
            RuntimeTelemetry.RecordException(activity, ex);
            throw;
        }
        finally
        {
            RuntimeTelemetry.InferenceDuration.Record(sw.Elapsed.TotalMilliseconds);
        }
    }

    /// <summary>
    /// Streams a response-style request while preserving response history state.
    /// </summary>
    public async IAsyncEnumerable<ChatResponseChunk> GenerateResponseAsync(
        ResponseRequest request,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Input is not { Count: > 0 })
            throw new InvalidOperationException("Response requests must contain at least one input item.");

        ThrowIfDisposed();

        bool isReentrant = s_reentrancyOwner.Value == this;
        List<ChatMessage>? savedHistory = null;

        if (isReentrant)
            savedHistory = SaveHistorySnapshot();
        else
            s_reentrancyOwner.Value = this;

        try
        {
            var responseHistory = BuildResponseHistory(request);
            if (responseHistory.Count == 0)
                throw new InvalidOperationException("Response request did not produce any renderable history.");

            int historyPrefixCount = Math.Max(0, responseHistory.Count - 1);

            ResetCore(clearResponses: false);
            _history.AddRange(responseHistory.Take(historyPrefixCount));

            var executionContext = await CreateResponseExecutionContextAsync(request.Inference, ct).ConfigureAwait(false);

            await foreach (var chunk in GenerateCoreAsync(responseHistory[^1], executionContext, ct).ConfigureAwait(false))
                yield return chunk;

            FinalizeResponse(request.Model, request.PreviousResponseId, historyPrefixCount);
        }
        finally
        {
            if (isReentrant)
                RestoreAfterReentrantCall(savedHistory!);
            else
                s_reentrancyOwner.Value = null;
        }
    }

    /// <summary>
    /// Generates an embedding vector as <see cref="float"/> values.
    /// </summary>
    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        ThrowIfDisposed();
        using var activity = RuntimeTelemetry.StartActivity("runtime.session.embed");
        RuntimeTelemetry.EmbeddingCalls.Add(1);
        try
        {
            return await _inferenceContext.EmbedAsync(text, ct);
        }
        catch (Exception ex)
        {
            RuntimeTelemetry.RecordException(activity, ex);
            throw;
        }
    }

    /// <summary>
    /// Generates an embedding vector as <see cref="double"/> values.
    /// </summary>
    public async Task<double[]> EmbedAsDoubleAsync(string text, CancellationToken ct = default)
    {
        ThrowIfDisposed();
        using var activity = RuntimeTelemetry.StartActivity("runtime.session.embed");
        RuntimeTelemetry.EmbeddingCalls.Add(1);
        try
        {
            return await _inferenceContext.EmbedAsDoubleAsync(text, ct);
        }
        catch (Exception ex)
        {
            RuntimeTelemetry.RecordException(activity, ex);
            throw;
        }
    }

    /// <summary>
    /// Streams assistant output for a user message.
    /// </summary>
    public async IAsyncEnumerable<ChatResponseChunk> GenerateAsync(
        ChatMessage message,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ThrowIfDisposed();

        bool isReentrant = s_reentrancyOwner.Value == this;
        List<ChatMessage>? savedHistory = null;

        if (isReentrant)
            savedHistory = SaveHistorySnapshot();
        else
            s_reentrancyOwner.Value = this;

        try
        {
            await foreach (var chunk in GenerateCoreAsync(message, new ResponseExecutionContext(_defaultInference, new Dictionary<string, RemoteToolBinding>(StringComparer.Ordinal)), ct).ConfigureAwait(false))
            {
                yield return chunk;
            }
        }
        finally
        {
            if (isReentrant)
                RestoreAfterReentrantCall(savedHistory!);
            else
                s_reentrancyOwner.Value = null;
        }
    }

    private async IAsyncEnumerable<ChatResponseChunk> GenerateCoreAsync(
        ChatMessage message,
        ResponseExecutionContext executionContext,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        _history.Add(message);
        var inference = executionContext.Inference;

        for (int round = 0; round < _options.MaxToolRounds; round++)
        {
            ct.ThrowIfCancellationRequested();

            var fitted = CompactHistory(_history);
            var promptMessages = SessionParsing.ApplyImageRetentionPolicy(fitted, _options.ImageRetentionPolicy);
            string prompt = _engine.RenderPrompt(promptMessages, inference);
            int[] promptTokens = _engine.Tokenize(prompt);

            DebugViewCreated?.Invoke(this, new SessionDebugView(
                History: [.. _history],
                PromptMessages: [.. promptMessages],
                RenderedPrompt: prompt,
                Tools: inference.Tools?.Select(Engine.BuildPromptTool).Cast<object?>().ToList(),
                PromptTokens: promptTokens.Length));

            List<string> imageBase64s = SessionParsing.ExtractImageBase64s(promptMessages);

            var output = new StringBuilder();
            List<ToolCall>? toolCalls = null;
            int completionTokens = 0;
            int emittedContentLength = 0;
            int emittedReasoningLength = 0;
            int maxOutputTokens = inference.MaxOutputTokens.GetValueOrDefault() > 0 ? inference.MaxOutputTokens.GetValueOrDefault() : _options.ContextTokens;

            await _inferenceContext.EncodePromptAsync(prompt, promptTokens, imageBase64s, ct);

            await foreach (var token in _inferenceContext.GenerateTokensAsync(inference, maxOutputTokens, ct).ConfigureAwait(false))
            {
                output.Append(token.Text);
                completionTokens++;

                var parsedOutput = SessionParsing.ParseAssistantOutput(output.ToString(), inference.EnableThinking ?? true);
                string visibleContent = SessionParsing.GetStreamingVisibleContent(parsedOutput.Content);
                string? contentDelta = SessionParsing.GetDelta(visibleContent, ref emittedContentLength);
                string? reasoningDelta = SessionParsing.GetDelta(parsedOutput.ReasoningContent, ref emittedReasoningLength);

                if (contentDelta is not null || reasoningDelta is not null)
                    yield return new ChatResponseChunk(Text: contentDelta, ReasoningText: reasoningDelta);

                if (MiniJinjaChatTemplate.TryParseToolCalls(output.ToString(), out toolCalls))
                    break;
            }

            string outputText = output.ToString();
            var usage = new InferenceUsage(promptTokens.Length, completionTokens);

            if (toolCalls is { Count: > 0 } || MiniJinjaChatTemplate.TryParseToolCalls(outputText, out toolCalls))
            {
                yield return new ChatResponseChunk(ToolCalls: toolCalls);

                var assistantMessage = SessionParsing.CreateAssistantMessage(outputText, inference.EnableThinking ?? true, toolCalls, usage);

                _history.Clear();
                _history.AddRange(fitted);
                _history.Add(assistantMessage);
                yield return new ChatResponseChunk(Usage: usage);

                // Tool execution happens outside the inference loop, so nested
                // calls (e.g. tools that invoke embeddings or sub-generation)
                // can proceed without contention.
                foreach (var call in toolCalls!)
                {
                    string result = await ExecuteToolAsync(call, executionContext.RemoteTools, ct);
                    _history.Add(new ChatMessage("tool", result, ToolCallId: call.CallId));
                }
                continue;
            }

            _history.Clear();
            _history.AddRange(fitted);
            _history.Add(SessionParsing.CreateAssistantMessage(outputText, inference.EnableThinking ?? true, usage: usage));
            yield return new ChatResponseChunk(Usage: usage);
            yield break;
        }
    }

    private async Task<ResponseExecutionContext> CreateResponseExecutionContextAsync(InferenceOptions inference, CancellationToken ct)
    {
        var remoteTools = new Dictionary<string, RemoteToolBinding>(StringComparer.Ordinal);
        List<ToolDefinition> promptTools = inference.Tools is { Count: > 0 }
            ? []
            : [.. _defaultInference.Tools ?? []];

        if (inference.Tools is { Count: > 0 })
        {
            foreach (var tool in inference.Tools)
            {
                if (string.Equals(tool.Type, "mcp", StringComparison.OrdinalIgnoreCase))
                {
                    foreach (var remoteTool in await McpHttpClient.ListToolsAsync(tool, ct).ConfigureAwait(false))
                    {
                        promptTools.Add(new ToolDefinition(
                            Type: "function",
                            Name: remoteTool.Name,
                            Parameters: remoteTool.InputSchema,
                            Description: remoteTool.Description));
                        remoteTools[remoteTool.Name] = remoteTool.Binding;
                    }

                    continue;
                }

                promptTools.Add(new ToolDefinition(
                    Type: "function",
                    Name: tool.Name,
                    Parameters: tool.Parameters ?? new { type = "object" },
                    Description: tool.Description));
            }
        }

        return new ResponseExecutionContext(MergeDefaults(inference, promptTools), remoteTools);
    }

    private InferenceOptions MergeDefaults(InferenceOptions inference, IReadOnlyList<ToolDefinition> promptTools)
    {
        bool enableThinking = inference.EnableThinking
            ?? (inference.ReasoningEffort is { Length: > 0 } reasoningEffort
                ? !string.Equals(reasoningEffort, "none", StringComparison.OrdinalIgnoreCase)
                : _defaultInference.EnableThinking ?? true);

        return inference with
        {
            Temperature = inference.Temperature ?? _defaultInference.Temperature,
            TopP = inference.TopP ?? _defaultInference.TopP,
            TopK = inference.TopK ?? _defaultInference.TopK,
            PresencePenalty = inference.PresencePenalty ?? _defaultInference.PresencePenalty,
            FrequencyPenalty = inference.FrequencyPenalty ?? _defaultInference.FrequencyPenalty,
            RepetitionPenalty = inference.RepetitionPenalty ?? _defaultInference.RepetitionPenalty,
            MaxOutputTokens = inference.MaxOutputTokens ?? _defaultInference.MaxOutputTokens,
            EnableThinking = enableThinking,
            Tools = promptTools.Count > 0 ? [.. promptTools] : null,
            AddVisionId = inference.AddVisionId || _defaultInference.AddVisionId,
            Seed = inference.Seed ?? _defaultInference.Seed
        };
    }

    private List<ChatMessage> BuildResponseHistory(ResponseRequest request)
    {
        List<ChatMessage> history = request.PreviousResponseId is { Length: > 0 } previousResponseId
            ? [.. GetResponseHistory(previousResponseId)]
            : [];

        history.AddRange(request.Input);

        return history;
    }

    private IReadOnlyList<ChatMessage> GetResponseHistory(string responseId)
    {
        if (!_responseHistories.TryGetValue(responseId, out var history))
            throw new InvalidOperationException($"Unknown previous_response_id '{responseId}'.");

        return [.. history];
    }

    private ResponseObject FinalizeResponse(string model, string? previousResponseId, int historyPrefixCount)
    {
        string responseId = $"resp_{Guid.NewGuid():N}";
        var newMessages = _history.Skip(historyPrefixCount).ToList();
        var usage = SumUsage(newMessages);
        var response = new ResponseObject(
            Id: responseId,
            CreatedAt: DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            Model: model,
            Output: newMessages,
            Usage: usage,
            PreviousResponseId: previousResponseId);

        _responseHistories[responseId] = [.. _history];
        LastResponse = response;
        return response;
    }

    private async Task<string> ExecuteToolAsync(
        ToolCall call,
        IReadOnlyDictionary<string, RemoteToolBinding> remoteTools,
        CancellationToken ct)
    {
        if (remoteTools.TryGetValue(call.Name, out var remoteTool))
            return await McpHttpClient.CallToolAsync(remoteTool, call.Arguments, ct).ConfigureAwait(false);

        if (!_options.ToolRegistry.TryGet(call.Name, out var tool))
            return $"Error: tool '{call.Name}' is not registered.";

        return await _toolExecutionEngine.ExecuteAsync(new ToolExecutionRequest(call, tool, _options.ToolRegistry), ct);
    }

    private IReadOnlyList<ChatMessage> CompactHistory(IReadOnlyList<ChatMessage> messages)
        => _conversationCompactor.Compact(messages, new ConversationCompactionContext(_compaction, CountTokens, SessionParsing.HasRenderableUserQuery));

    private int CountTokens(IReadOnlyList<ChatMessage> messages)
        => _engine.Tokenize(_engine.RenderPrompt(messages, _defaultInference)).Length;

    private static InferenceUsage SumUsage(IReadOnlyList<ChatMessage> messages)
    {
        int promptTokens = 0;
        int completionTokens = 0;

        foreach (var message in messages)
        {
            if (message.Usage is { } usage)
            {
                promptTokens += usage.PromptTokens;
                completionTokens += usage.CompletionTokens;
            }
        }

        return new InferenceUsage(promptTokens, completionTokens);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        await _inferenceContext.DisposeAsync();
        if (_ownsEngine)
            await _engine.DisposeAsync();
    }

    /// <summary>
    /// Disposes the session synchronously.
    /// </summary>
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    private List<ChatMessage> SaveHistorySnapshot() => [.. _history];

    private void RestoreAfterReentrantCall(List<ChatMessage> savedHistory)
    {
        _history.Clear();
        _history.AddRange(savedHistory);
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}