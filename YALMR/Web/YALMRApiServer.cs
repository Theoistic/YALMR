using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using YALMR.Runtime;

namespace YALMR.Web;

/// <summary>
/// Hosts the YALMR inference HTTP API on top of a <see cref="YALMRServer"/>.
/// <para>
/// Self-hosted usage: call <see cref="StartAsync"/> to spin up a Kestrel listener.
/// </para>
/// <para>
/// Integration usage: call <see cref="MapEndpoints"/> to attach routes to an existing
/// <see cref="IEndpointRouteBuilder"/> (e.g. an ASP.NET Core app you already own).
/// </para>
/// </summary>
public sealed class YALMRApiServer : IAsyncDisposable
{
    private readonly YALMRServer _server;
    private WebApplication? _app;
    private bool _disposed;

    public YALMRApiServer(YALMRServer server)
    {
        ArgumentNullException.ThrowIfNull(server);
        _server = server;
    }

    // ─── Self-hosted lifecycle ────────────────────────────────────────────────

    /// <summary>
    /// Builds and starts a self-hosted Kestrel server listening on <paramref name="url"/>.
    /// </summary>
    public async Task StartAsync(string url = "http://localhost:5000", CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var builder = WebApplication.CreateSlimBuilder();
        _app = builder.Build();
        _app.Urls.Clear();
        _app.Urls.Add(url);
        MapEndpoints(_app);

        await _app.StartAsync(ct);
    }

    /// <summary>
    /// Stops the self-hosted server.
    /// </summary>
    public async Task StopAsync(CancellationToken ct = default)
    {
        if (_app is not null)
            await _app.StopAsync(ct);
    }

    // ─── Route registration ───────────────────────────────────────────────────

    /// <summary>
    /// Maps all YALMR API routes onto the provided <paramref name="routes"/>.
    /// </summary>
    public void MapEndpoints(IEndpointRouteBuilder routes)
    {
        routes.MapGet("/chat", () => ServeEmbeddedResource("YALMR.Web.Chat.index.html", "text/html; charset=utf-8"));
        routes.MapGet("/chat/inference.js", () => ServeEmbeddedResource("YALMR.Web.Chat.inference.js", "application/javascript; charset=utf-8"));
        routes.MapGet("/v1/health", HandleHealth);
        routes.MapGet("/v1/models", HandleModels);
        routes.MapPost("/v1/generate", HandleGenerateAsync);
        routes.MapPost("/v1/chat", HandleChatAsync);
        routes.MapPost("/v1/sessions", HandleCreateSession);
        routes.MapPost("/v1/sessions/{id}/chat", HandleSessionChatAsync);
        routes.MapDelete("/v1/sessions/{id}", HandleDeleteSessionAsync);
    }

    // ─── GET /v1/health ───────────────────────────────────────────────────────

    private IResult HandleHealth() =>
        Results.Ok(new HealthResponse(
            Ok: true,
            Engine: "llama.cpp",
            Model: _server.ModelIds.FirstOrDefault()));

    // ─── GET /v1/models ───────────────────────────────────────────────────────

    private IResult HandleModels()
    {
        var models = _server.ModelIds
            .Select(id =>
            {
                int ctx = _server.TryGetModelOptions(id, out var opts) ? opts.ContextTokens : 0;
                return new ModelInfo(id, "model", Loaded: true, ContextLength: ctx);
            })
            .ToList();

        return Results.Ok(new ModelsResponse(models));
    }

    // ─── POST /v1/generate ───────────────────────────────────────────────────

    private async Task<IResult> HandleGenerateAsync(GenerateRequest req, HttpContext ctx, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Model))
            return Error(400, "invalid_request", "model is required", "missing_model");
        if (string.IsNullOrWhiteSpace(req.Prompt))
            return Error(400, "invalid_request", "prompt is required", "missing_prompt");
        if (!_server.IsModelLoaded(req.Model))
            return Error(404, "not_found", $"model '{req.Model}' is not loaded", "model_not_found");

        string genId = $"gen_{Guid.NewGuid():N}";
        var inference = BuildInference(req.MaxTokens, req.Temperature, req.TopP);
        IReadOnlyList<ChatMessage> input = [new ChatMessage("user", req.Prompt)];

        if (req.Stream)
        {
            var sessionId = _server.CreateSession(req.Model);
            try
            {
                var session = _server.GetSession(sessionId);
                var request = new ResponseRequest { Model = req.Model, Input = input, Inference = inference };
                await WriteSseAsync(ctx.Response, session.GenerateResponseAsync(request, ct), genId, req.Model, ct);
            }
            finally
            {
                await _server.RemoveSessionAsync(sessionId);
            }
            return Results.Empty;
        }

        return await RunStatelessAsync(req.Model, input, inference, ct, response =>
        {
            var msg = response.Output.LastOrDefault(m => m.Role == "assistant");
            return Results.Ok(new GenerateResponse(
                Id: genId,
                Object: "text_completion",
                Model: req.Model,
                Text: msg?.Content ?? string.Empty,
                FinishReason: "stop",
                Usage: ToUsage(response.Usage)));
        });
    }

    // ─── POST /v1/chat ────────────────────────────────────────────────────────

    private async Task<IResult> HandleChatAsync(ChatRequest req, HttpContext ctx, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Model))
            return Error(400, "invalid_request", "model is required", "missing_model");
        if (req.Messages is not { Count: > 0 })
            return Error(400, "invalid_request", "messages is required", "missing_messages");
        if (!_server.IsModelLoaded(req.Model))
            return Error(404, "not_found", $"model '{req.Model}' is not loaded", "model_not_found");

        string chatId = $"chat_{Guid.NewGuid():N}";
        var inference = BuildInference(req.MaxTokens, req.Temperature, req.TopP);
        IReadOnlyList<ChatMessage> input = req.Messages
            .Select(ToRuntimeMessage)
            .ToList();

        if (req.Stream)
        {
            var sessionId = _server.CreateSession(req.Model);
            try
            {
                var session = _server.GetSession(sessionId);
                var request = new ResponseRequest { Model = req.Model, Input = input, Inference = inference };
                await WriteSseAsync(ctx.Response, session.GenerateResponseAsync(request, ct), chatId, req.Model, ct);
            }
            finally
            {
                await _server.RemoveSessionAsync(sessionId);
            }
            return Results.Empty;
        }

        return await RunStatelessAsync(req.Model, input, inference, ct, response =>
        {
            var msg = response.Output.LastOrDefault(m => m.Role == "assistant");
            return Results.Ok(new ChatResponse(
                Id: chatId,
                Object: "chat.completion",
                Model: req.Model,
                Message: new ApiMessage("assistant", msg?.Content ?? string.Empty),
                FinishReason: "stop",
                Usage: ToUsage(response.Usage)));
        });
    }

    // ─── POST /v1/sessions ────────────────────────────────────────────────────

    private IResult HandleCreateSession(CreateSessionRequest req)
    {
        if (string.IsNullOrWhiteSpace(req.Model))
            return Error(400, "invalid_request", "model is required", "missing_model");
        if (!_server.IsModelLoaded(req.Model))
            return Error(404, "not_found", $"model '{req.Model}' is not loaded", "model_not_found");

        var sessionId = _server.CreateSession(req.Model);

        if (!string.IsNullOrWhiteSpace(req.System))
            _server.GetSession(sessionId).PrimeHistory(new ChatMessage("system", req.System));

        return Results.Ok(new CreateSessionResponse(sessionId, "session", req.Model));
    }

    // ─── POST /v1/sessions/{id}/chat ─────────────────────────────────────────

    private async Task<IResult> HandleSessionChatAsync(string id, SessionChatRequest req, HttpContext ctx, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Message) && req.Parts is not { Count: > 0 })
            return Error(400, "invalid_request", "message or parts is required", "missing_message");
        if (!_server.TryGetSession(id, out var session))
            return Error(404, "not_found", $"session '{id}' does not exist", "session_not_found");

        string chatId = $"chat_{Guid.NewGuid():N}";
        var userMessage = req.Parts is { Count: > 0 } parts
            ? new ChatMessage("user", Parts: ToRuntimeParts(parts))
            : new ChatMessage("user", req.Message);

        if (req.Stream)
        {
            await WriteSseAsync(ctx.Response, session.GenerateAsync(userMessage, ct), chatId, string.Empty, ct);
            return Results.Empty;
        }

        var reply = await session.SendAsync(userMessage, ct);
        return Results.Ok(new SessionChatResponse(
            Id: chatId,
            SessionId: id,
            Message: new ApiMessage("assistant", reply.Content ?? string.Empty),
            FinishReason: "stop"));
    }

    // ─── DELETE /v1/sessions/{id} ─────────────────────────────────────────────

    private async Task<IResult> HandleDeleteSessionAsync(string id)
    {
        if (!await _server.RemoveSessionAsync(id))
            return Error(404, "not_found", $"session '{id}' does not exist", "session_not_found");

        return Results.Ok(new { ok = true });
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private static IResult ServeEmbeddedResource(string resourceName, string contentType)
    {
        var stream = typeof(YALMRApiServer).Assembly.GetManifestResourceStream(resourceName);
        return stream is null ? Results.NotFound() : Results.Stream(stream, contentType);
    }

    private async Task<IResult> RunStatelessAsync(
        string modelId,
        IReadOnlyList<ChatMessage> input,
        InferenceOptions inference,
        CancellationToken ct,
        Func<ResponseObject, IResult> buildResult)
    {
        var sessionId = _server.CreateSession(modelId);
        try
        {
            var session = _server.GetSession(sessionId);
            var request = new ResponseRequest { Model = modelId, Input = input, Inference = inference };
            var response = await session.CreateResponseAsync(request, ct);
            return buildResult(response);
        }
        catch (Exception ex)
        {
            return Error(500, "engine_failure", ex.Message, "inference_error");
        }
        finally
        {
            await _server.RemoveSessionAsync(sessionId);
        }
    }

    private static async Task WriteSseAsync(
        HttpResponse response,
        IAsyncEnumerable<ChatResponseChunk> stream,
        string id,
        string model,
        CancellationToken ct)
    {
        response.ContentType = "text/event-stream";
        response.Headers["Cache-Control"] = "no-cache";
        response.Headers["X-Accel-Buffering"] = "no";

        await response.WriteAsync($"event: start\ndata: {Json(new SseStartEvent(id, model))}\n\n", ct);
        await response.BodyWriter.FlushAsync(ct);

        InferenceUsage? lastUsage = null;

        await foreach (var chunk in stream.WithCancellation(ct))
        {
            if (chunk.ReasoningText is { Length: > 0 } rDelta)
            {
                await response.WriteAsync($"event: thinking\ndata: {Json(new SseThinkingEvent(rDelta))}\n\n", ct);
                await response.BodyWriter.FlushAsync(ct);
            }

            if (chunk.Text is { Length: > 0 } delta)
            {
                await response.WriteAsync($"event: token\ndata: {Json(new SseTokenEvent(delta))}\n\n", ct);
                await response.BodyWriter.FlushAsync(ct);
            }

            if (chunk.ToolCalls is { Count: > 0 } toolCalls)
            {
                foreach (var call in toolCalls)
                {
                    await response.WriteAsync($"event: tool_call\ndata: {Json(new SseToolCallEvent(call.CallId, call.Name, call.Arguments))}\n\n", ct);
                    await response.BodyWriter.FlushAsync(ct);
                }
            }

            if (chunk.ToolResults is { Count: > 0 } toolResults)
            {
                foreach (var res in toolResults)
                {
                    await response.WriteAsync($"event: tool_result\ndata: {Json(new SseToolResultEvent(res.CallId, res.Name, res.Result))}\n\n", ct);
                    await response.BodyWriter.FlushAsync(ct);
                }
            }

            if (chunk.Usage is not null)
                lastUsage = chunk.Usage;
        }

        var endEvent = new SseEndEvent("stop", lastUsage is null ? null : ToUsage(lastUsage));
        await response.WriteAsync($"event: end\ndata: {Json(endEvent)}\n\n", ct);
        await response.BodyWriter.FlushAsync(ct);
    }

    private static ChatMessage ToRuntimeMessage(ApiMessage msg) =>
        msg.Content?.Parts is { Count: > 0 } parts
            ? new ChatMessage(msg.Role, Parts: ToRuntimeParts(parts))
            : new ChatMessage(msg.Role, msg.Content?.Text ?? string.Empty);

    private static IReadOnlyList<ContentPart> ToRuntimeParts(IReadOnlyList<ApiContentPart> parts)
    {
        var result = new List<ContentPart>(parts.Count);
        foreach (var part in parts)
        {
            switch (part)
            {
                case ApiTextPart t:
                    result.Add(new TextPart(t.Text));
                    break;
                case ApiImagePart img:
                    result.Add(new ImagePart(ExtractBase64(img.ImageUrl.Url)));
                    break;
                case ApiFilePart f:
                    result.Add(new ImagePart(f.File.Data));
                    break;
            }
        }
        return result;
    }

    // Strips the data URI prefix from an image_url value.
    // "data:image/png;base64,xxx" → "xxx"; plain base64 strings pass through unchanged.
    private static string ExtractBase64(string url)
    {
        int idx = url.IndexOf("base64,", StringComparison.Ordinal);
        return idx >= 0 ? url[(idx + 7)..] : url;
    }

    private static InferenceOptions BuildInference(int? maxTokens, float? temperature, float? topP) =>
        new() { MaxOutputTokens = maxTokens, Temperature = temperature, TopP = topP };

    private static ApiUsage ToUsage(InferenceUsage u) =>
        new(u.PromptTokens, u.CompletionTokens, u.TotalTokens);

    private static string Json<T>(T value) => JsonSerializer.Serialize(value);

    private static IResult Error(int status, string type, string message, string code) =>
        Results.Json(new ApiErrorResponse(new ApiErrorDetail(type, message, code)), statusCode: status);

    // ─── SSE event shapes ─────────────────────────────────────────────────────

    private sealed record SseStartEvent(
        [property: JsonPropertyName("id")] string Id,
        [property: JsonPropertyName("model")] string Model);

    private sealed record SseTokenEvent(
        [property: JsonPropertyName("delta")] string Delta);

    private sealed record SseThinkingEvent(
        [property: JsonPropertyName("delta")] string Delta);

    private sealed record SseToolCallEvent(
        [property: JsonPropertyName("call_id")] string CallId,
        [property: JsonPropertyName("name")] string Name,
        [property: JsonPropertyName("arguments")] object Arguments);

    private sealed record SseToolResultEvent(
        [property: JsonPropertyName("call_id")] string CallId,
        [property: JsonPropertyName("name")] string Name,
        [property: JsonPropertyName("result")] string Result);

    private sealed record SseEndEvent(
        [property: JsonPropertyName("finish_reason")] string FinishReason,
        [property: JsonPropertyName("usage")] ApiUsage? Usage);

    // ─── Disposal ─────────────────────────────────────────────────────────────

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        if (_app is not null)
            await _app.DisposeAsync();
    }
}
