using System.Text.Json;
using System.Text.Json.Serialization;

namespace YALMR.Web;

// ── Health ──────────────────────────────────────────────────────────────────

public sealed record HealthResponse(
    [property: JsonPropertyName("ok")] bool Ok,
    [property: JsonPropertyName("engine")] string Engine,
    [property: JsonPropertyName("model")] string? Model);

// ── Models ──────────────────────────────────────────────────────────────────

public sealed record ModelInfo(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("loaded")] bool Loaded,
    [property: JsonPropertyName("context_length")] int ContextLength);

public sealed record ModelsResponse(
    [property: JsonPropertyName("data")] IReadOnlyList<ModelInfo> Data);

// ── Content parts ───────────────────────────────────────────────────────────

[JsonPolymorphic(TypeDiscriminatorPropertyName = "type")]
[JsonDerivedType(typeof(ApiTextPart), "text")]
[JsonDerivedType(typeof(ApiImagePart), "image_url")]
[JsonDerivedType(typeof(ApiFilePart), "file")]
public abstract record ApiContentPart;

public sealed record ApiTextPart(
    [property: JsonPropertyName("text")] string Text) : ApiContentPart;

public sealed record ApiImageUrl(
    [property: JsonPropertyName("url")] string Url,
    [property: JsonPropertyName("detail")] string? Detail = null);

public sealed record ApiImagePart(
    [property: JsonPropertyName("image_url")] ApiImageUrl ImageUrl) : ApiContentPart;

public sealed record ApiFileData(
    [property: JsonPropertyName("filename")] string Filename,
    [property: JsonPropertyName("data")] string Data,
    [property: JsonPropertyName("mime_type")] string? MimeType = null);

public sealed record ApiFilePart(
    [property: JsonPropertyName("file")] ApiFileData File) : ApiContentPart;

/// <summary>
/// A message's content: either a plain text string or a list of typed content parts.
/// Serialises as a JSON string when text-only, or as a JSON array when parts are present.
/// </summary>
[JsonConverter(typeof(ApiContentConverter))]
public sealed class ApiContent
{
    public string? Text { get; init; }
    public IReadOnlyList<ApiContentPart>? Parts { get; init; }

    public static implicit operator ApiContent(string text) => new() { Text = text };
}

internal sealed class ApiContentConverter : JsonConverter<ApiContent>
{
    public override ApiContent Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
        reader.TokenType switch
        {
            JsonTokenType.String => new ApiContent { Text = reader.GetString() },
            JsonTokenType.StartArray => new ApiContent { Parts = JsonSerializer.Deserialize<List<ApiContentPart>>(ref reader, options) },
            JsonTokenType.Null => new ApiContent { Text = string.Empty },
            _ => throw new JsonException($"Unexpected token type {reader.TokenType} for message content.")
        };

    public override void Write(Utf8JsonWriter writer, ApiContent value, JsonSerializerOptions options)
    {
        if (value.Parts is not null)
            JsonSerializer.Serialize(writer, value.Parts, options);
        else
            writer.WriteStringValue(value.Text ?? string.Empty);
    }
}

// ── Generate ────────────────────────────────────────────────────────────────

public sealed record GenerateRequest
{
    [JsonPropertyName("model")] public string Model { get; init; } = string.Empty;
    [JsonPropertyName("prompt")] public string Prompt { get; init; } = string.Empty;
    [JsonPropertyName("max_tokens")] public int? MaxTokens { get; init; }
    [JsonPropertyName("temperature")] public float? Temperature { get; init; }
    [JsonPropertyName("top_p")] public float? TopP { get; init; }
    [JsonPropertyName("stop")] public IReadOnlyList<string>? Stop { get; init; }
    [JsonPropertyName("stream")] public bool Stream { get; init; }
}

public sealed record ApiUsage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens);

public sealed record GenerateResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("finish_reason")] string FinishReason,
    [property: JsonPropertyName("usage")] ApiUsage Usage);

// ── Chat ────────────────────────────────────────────────────────────────────

public sealed record ApiMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] ApiContent Content);

public sealed record ChatRequest
{
    [JsonPropertyName("model")] public string Model { get; init; } = string.Empty;
    [JsonPropertyName("messages")] public IReadOnlyList<ApiMessage> Messages { get; init; } = [];
    [JsonPropertyName("max_tokens")] public int? MaxTokens { get; init; }
    [JsonPropertyName("temperature")] public float? Temperature { get; init; }
    [JsonPropertyName("top_p")] public float? TopP { get; init; }
    [JsonPropertyName("stop")] public IReadOnlyList<string>? Stop { get; init; }
    [JsonPropertyName("stream")] public bool Stream { get; init; }
}

public sealed record ChatResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("message")] ApiMessage Message,
    [property: JsonPropertyName("finish_reason")] string FinishReason,
    [property: JsonPropertyName("usage")] ApiUsage Usage);

// ── Sessions ─────────────────────────────────────────────────────────────────

public sealed record CreateSessionRequest
{
    [JsonPropertyName("model")] public string Model { get; init; } = string.Empty;
    [JsonPropertyName("system")] public string? System { get; init; }
}

public sealed record CreateSessionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("model")] string Model);

public sealed record SessionChatRequest
{
    [JsonPropertyName("message")] public string Message { get; init; } = string.Empty;
    [JsonPropertyName("parts")] public IReadOnlyList<ApiContentPart>? Parts { get; init; }
    [JsonPropertyName("stream")] public bool Stream { get; init; }
}

public sealed record SessionChatResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("session_id")] string SessionId,
    [property: JsonPropertyName("message")] ApiMessage Message,
    [property: JsonPropertyName("finish_reason")] string FinishReason);

// ── Errors ───────────────────────────────────────────────────────────────────

public sealed record ApiErrorDetail(
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("message")] string Message,
    [property: JsonPropertyName("code")] string Code);

public sealed record ApiErrorResponse(
    [property: JsonPropertyName("error")] ApiErrorDetail Error);
