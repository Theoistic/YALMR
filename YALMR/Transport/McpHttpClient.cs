using YALMR.Runtime;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace YALMR.Transport;

internal sealed record RemoteToolBinding(
    string ServerUrl,
    string ToolName,
    IReadOnlyDictionary<string, string>? Headers = null);

internal sealed record RemoteToolListing(
    string Name,
    string? Description,
    JsonElement InputSchema,
    RemoteToolBinding Binding);

internal static class McpHttpClient
{
    private static readonly HttpClient s_http = new();
    private static readonly JsonSerializerOptions s_json = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public static async Task<IReadOnlyList<RemoteToolListing>> ListToolsAsync(ToolDefinition definition, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(definition);

        if (string.IsNullOrWhiteSpace(definition.ServerUrl))
            throw new InvalidOperationException("MCP tool definition is missing a server URL.");

        _ = await SendRequestAsync(definition.ServerUrl, definition.Headers, 1, "initialize", new
        {
            protocolVersion = "2025-03-26",
            capabilities = new { },
            clientInfo = new
            {
                name = "YALMR",
                version = "1.0.0"
            }
        }, ct);

        JsonElement result = await SendRequestAsync(definition.ServerUrl, definition.Headers, 2, "tools/list", new { }, ct);
        var tools = new List<RemoteToolListing>();
        var allowed = definition.AllowedTools is { Count: > 0 }
            ? new HashSet<string>(definition.AllowedTools, StringComparer.Ordinal)
            : null;

        if (result.TryGetProperty("tools", out var toolsElement) && toolsElement.ValueKind == JsonValueKind.Array)
        {
            foreach (var toolElement in toolsElement.EnumerateArray())
            {
                if (!toolElement.TryGetProperty("name", out var nameElement) || nameElement.GetString() is not { Length: > 0 } name)
                    continue;

                if (allowed is not null && !allowed.Contains(name))
                    continue;

                JsonElement inputSchema = toolElement.TryGetProperty("inputSchema", out var schemaElement)
                    ? schemaElement.Clone()
                    : JsonDocument.Parse("{\"type\":\"object\"}").RootElement.Clone();

                string? description = toolElement.TryGetProperty("description", out var descriptionElement)
                    ? descriptionElement.GetString()
                    : null;

                tools.Add(new RemoteToolListing(
                    Name: name,
                    Description: description,
                    InputSchema: inputSchema,
                    Binding: new RemoteToolBinding(definition.ServerUrl, name, definition.Headers)));
            }
        }

        return tools;
    }

    public static async Task<string> CallToolAsync(RemoteToolBinding binding, IReadOnlyDictionary<string, object?> arguments, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(binding);

        JsonElement result = await SendRequestAsync(binding.ServerUrl, binding.Headers, 3, "tools/call", new
        {
            name = binding.ToolName,
            arguments
        }, ct);

        return ExtractToolResult(result);
    }

    private static async Task<JsonElement> SendRequestAsync(
        string serverUrl,
        IReadOnlyDictionary<string, string>? headers,
        int requestId,
        string method,
        object? parameters,
        CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, serverUrl)
        {
            Content = new StringContent(JsonSerializer.Serialize(new
            {
                jsonrpc = "2.0",
                id = requestId,
                method,
                @params = parameters
            }, s_json), Encoding.UTF8, "application/json")
        };

        if (headers is not null)
        {
            foreach (var pair in headers)
            {
                if (!request.Headers.TryAddWithoutValidation(pair.Key, pair.Value))
                    request.Content.Headers.TryAddWithoutValidation(pair.Key, pair.Value);
            }
        }

        using var response = await s_http.SendAsync(request, ct);
        string body = await response.Content.ReadAsStringAsync(ct);

        if (!response.IsSuccessStatusCode)
            throw new InvalidOperationException($"MCP request '{method}' failed ({(int)response.StatusCode}): {body}");

        using JsonDocument document = JsonDocument.Parse(body);
        JsonElement root = document.RootElement;

        if (root.TryGetProperty("error", out var errorElement))
            throw new InvalidOperationException($"MCP request '{method}' failed: {errorElement.GetRawText()}");

        if (!root.TryGetProperty("result", out var resultElement))
            return JsonDocument.Parse("null").RootElement.Clone();

        return resultElement.Clone();
    }

    private static string ExtractToolResult(JsonElement result)
    {
        if (result.TryGetProperty("isError", out var isErrorElement) && isErrorElement.ValueKind == JsonValueKind.True)
            return $"Error: {result.GetRawText()}";

        if (result.TryGetProperty("content", out var contentElement) && contentElement.ValueKind == JsonValueKind.Array)
        {
            var parts = new List<string>();
            foreach (var item in contentElement.EnumerateArray())
            {
                if (item.TryGetProperty("text", out var textElement) && textElement.ValueKind == JsonValueKind.String)
                    parts.Add(textElement.GetString() ?? string.Empty);
                else
                    parts.Add(item.GetRawText());
            }

            return string.Join(Environment.NewLine, parts);
        }

        return result.GetRawText();
    }
}
