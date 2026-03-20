using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using YALMR.Runtime;

namespace YALMR.Mcp;

/// <summary>
/// Executes remote tools against MCP servers over stdio using JSON-RPC framing.
/// </summary>
public sealed class McpRemoteToolExecutor : IRemoteToolExecutor, IAsyncDisposable, IDisposable
{
    private readonly ConcurrentDictionary<string, McpClientSession> _sessions = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Returns <c>true</c> for remote tool definitions that use the MCP transport.
    /// </summary>
    public bool CanExecute(AgentToolRemoteDefinition definition)
        => string.Equals(definition.Transport, "mcp", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Executes a remote tool call against an MCP server.
    /// </summary>
    public async Task<string> ExecuteAsync(AgentToolRemoteDefinition definition, ToolCall call, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(definition);

        if (!TryResolveServerOptions(definition, out var serverOptions))
            return $"Error: MCP tool '{definition.ToolName}' on '{definition.Server}' is missing valid McpServerProcessOptions metadata.";

        var session = _sessions.GetOrAdd(definition.Server, _ => new McpClientSession(serverOptions));
        return await session.CallToolAsync(definition.ToolName, call.Arguments, ct);
    }

    public async ValueTask DisposeAsync()
    {
        foreach (var session in _sessions.Values)
            await session.DisposeAsync();

        _sessions.Clear();
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    private static bool TryResolveServerOptions(AgentToolRemoteDefinition definition, out McpServerProcessOptions options)
    {
        if (definition.Metadata is McpServerProcessOptions typed)
        {
            options = typed;
            return true;
        }

        if (definition.Metadata is IReadOnlyDictionary<string, object?> readonlyDict)
            return TryResolveServerOptions(readonlyDict, out options);

        if (definition.Metadata is IDictionary<string, object?> dict)
            return TryResolveServerOptions(dict.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal), out options);

        options = default!;
        return false;
    }

    private static bool TryResolveServerOptions(IReadOnlyDictionary<string, object?> metadata, out McpServerProcessOptions options)
    {
        if (!metadata.TryGetValue("command", out var commandValue) || commandValue?.ToString() is not { Length: > 0 } command)
        {
            options = default!;
            return false;
        }

        IReadOnlyList<string>? arguments = metadata.TryGetValue("arguments", out var argumentsValue)
            ? ConvertToStringList(argumentsValue)
            : null;

        IReadOnlyDictionary<string, string>? environment = metadata.TryGetValue("environmentVariables", out var environmentValue)
            ? ConvertToStringDictionary(environmentValue)
            : null;

        options = new McpServerProcessOptions(
            Command: command,
            Arguments: arguments,
            WorkingDirectory: metadata.TryGetValue("workingDirectory", out var wd) ? wd?.ToString() : null,
            EnvironmentVariables: environment,
            ProtocolVersion: metadata.TryGetValue("protocolVersion", out var pv) && pv?.ToString() is { Length: > 0 } protocolVersion ? protocolVersion : "2024-11-05",
            ClientName: metadata.TryGetValue("clientName", out var cn) && cn?.ToString() is { Length: > 0 } clientName ? clientName : "YALMR",
            ClientVersion: metadata.TryGetValue("clientVersion", out var cv) && cv?.ToString() is { Length: > 0 } clientVersion ? clientVersion : "1.0.0");
        return true;
    }

    private static IReadOnlyList<string>? ConvertToStringList(object? value)
    {
        return value switch
        {
            null => null,
            IReadOnlyList<string> strings => strings,
            IEnumerable<string> strings => strings.ToArray(),
            JsonElement { ValueKind: JsonValueKind.Array } jsonArray => [.. jsonArray.EnumerateArray().Select(item => item.ToString())],
            IEnumerable<object?> objects => [.. objects.Select(item => item?.ToString() ?? string.Empty)],
            _ => null
        };
    }

    private static IReadOnlyDictionary<string, string>? ConvertToStringDictionary(object? value)
    {
        return value switch
        {
            null => null,
            IReadOnlyDictionary<string, string> dict => dict,
            IDictionary<string, string> dict => new Dictionary<string, string>(dict, StringComparer.Ordinal),
            IReadOnlyDictionary<string, object?> dict => dict.ToDictionary(kv => kv.Key, kv => kv.Value?.ToString() ?? string.Empty, StringComparer.Ordinal),
            IDictionary<string, object?> dict => dict.ToDictionary(kv => kv.Key, kv => kv.Value?.ToString() ?? string.Empty, StringComparer.Ordinal),
            JsonElement { ValueKind: JsonValueKind.Object } jsonObject => jsonObject.EnumerateObject().ToDictionary(property => property.Name, property => property.Value.ToString(), StringComparer.Ordinal),
            _ => null
        };
    }

    private sealed class McpClientSession : IAsyncDisposable, IDisposable
    {
        private static readonly JsonSerializerOptions s_jsonOptions = new(JsonSerializerDefaults.Web)
        {
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };

        private readonly McpServerProcessOptions _options;
        private readonly SemaphoreSlim _lock = new(1, 1);
        private readonly Process _process;
        private readonly Stream _stdin;
        private readonly Stream _stdout;
        private int _nextRequestId;
        private bool _initialized;
        private bool _disposed;

        public McpClientSession(McpServerProcessOptions options)
        {
            _options = options;

            var startInfo = new ProcessStartInfo
            {
                FileName = options.Command,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            if (!string.IsNullOrWhiteSpace(options.WorkingDirectory))
                startInfo.WorkingDirectory = options.WorkingDirectory;

            if (options.Arguments is { Count: > 0 })
            {
                foreach (string argument in options.Arguments)
                    startInfo.ArgumentList.Add(argument);
            }

            if (options.EnvironmentVariables is { Count: > 0 })
            {
                foreach (var pair in options.EnvironmentVariables)
                    startInfo.Environment[pair.Key] = pair.Value;
            }

            _process = Process.Start(startInfo) ?? throw new InvalidOperationException($"Failed to start MCP server process '{options.Command}'.");
            _stdin = _process.StandardInput.BaseStream;
            _stdout = _process.StandardOutput.BaseStream;
            _ = DrainStandardErrorAsync(_process.StandardError);
        }

        public async Task<string> CallToolAsync(string toolName, IReadOnlyDictionary<string, object?> arguments, CancellationToken ct)
        {
            await _lock.WaitAsync(ct);

            try
            {
                await EnsureInitializedAsync(ct);

                JsonElement result = await SendRequestAsync("tools/call", new
                {
                    name = toolName,
                    arguments
                }, ct);

                return ExtractToolResult(result);
            }
            finally
            {
                _lock.Release();
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (_disposed)
                return;

            _disposed = true;
            _lock.Dispose();

            try
            {
                if (!_process.HasExited)
                    _process.Kill(entireProcessTree: true);
            }
            catch
            {
            }

            _stdin.Dispose();
            _stdout.Dispose();
            _process.Dispose();
            await ValueTask.CompletedTask;
        }

        public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

        private async Task EnsureInitializedAsync(CancellationToken ct)
        {
            if (_initialized)
                return;

            _ = await SendRequestAsync("initialize", new
            {
                protocolVersion = _options.ProtocolVersion,
                capabilities = new { },
                clientInfo = new
                {
                    name = _options.ClientName,
                    version = _options.ClientVersion
                }
            }, ct);

            await SendNotificationAsync("notifications/initialized", null, ct);
            _initialized = true;
        }

        private async Task<JsonElement> SendRequestAsync(string method, object? parameters, CancellationToken ct)
        {
            int requestId = Interlocked.Increment(ref _nextRequestId);
            await WriteMessageAsync(new
            {
                jsonrpc = "2.0",
                id = requestId,
                method,
                @params = parameters
            }, ct);

            while (true)
            {
                using JsonDocument message = await ReadMessageAsync(ct);
                JsonElement root = message.RootElement;

                if (root.TryGetProperty("id", out var idElement) && idElement.ValueKind == JsonValueKind.Number && idElement.TryGetInt32(out int responseId) && responseId == requestId)
                {
                    if (root.TryGetProperty("error", out var errorElement))
                        throw new InvalidOperationException($"MCP request '{method}' failed: {errorElement.GetRawText()}");

                    if (!root.TryGetProperty("result", out var resultElement))
                        return JsonDocument.Parse("null").RootElement.Clone();

                    return resultElement.Clone();
                }
            }
        }

        private async Task SendNotificationAsync(string method, object? parameters, CancellationToken ct)
        {
            await WriteMessageAsync(new
            {
                jsonrpc = "2.0",
                method,
                @params = parameters
            }, ct);
        }

        private async Task WriteMessageAsync(object payload, CancellationToken ct)
        {
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(payload, s_jsonOptions);
            byte[] header = Encoding.ASCII.GetBytes($"Content-Length: {json.Length}\r\n\r\n");

            await _stdin.WriteAsync(header, ct);
            await _stdin.WriteAsync(json, ct);
            await _stdin.FlushAsync(ct);
        }

        private async Task<JsonDocument> ReadMessageAsync(CancellationToken ct)
        {
            int? contentLength = null;

            while (true)
            {
                string line = await ReadAsciiLineAsync(_stdout, ct);
                if (line.Length == 0)
                    break;

                if (line.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase)
                    && int.TryParse(line["Content-Length:".Length..].Trim(), out int parsedLength))
                {
                    contentLength = parsedLength;
                }
            }

            if (contentLength is not int length || length < 0)
                throw new InvalidOperationException("MCP server response did not contain a valid Content-Length header.");

            byte[] buffer = new byte[length];
            int offset = 0;

            while (offset < buffer.Length)
            {
                int read = await _stdout.ReadAsync(buffer.AsMemory(offset, buffer.Length - offset), ct);
                if (read == 0)
                    throw new EndOfStreamException("MCP server closed the stream while a message body was being read.");

                offset += read;
            }

            return JsonDocument.Parse(buffer);
        }

        private static async Task<string> ReadAsciiLineAsync(Stream stream, CancellationToken ct)
        {
            using var buffer = new MemoryStream();

            while (true)
            {
                byte[] singleByte = new byte[1];
                int read = await stream.ReadAsync(singleByte, ct);
                if (read == 0)
                    throw new EndOfStreamException("Unexpected end of stream while reading MCP headers.");

                if (singleByte[0] == (byte)'\n')
                    break;

                buffer.WriteByte(singleByte[0]);
            }

            byte[] bytes = buffer.ToArray();
            if (bytes.Length > 0 && bytes[^1] == (byte)'\r')
                Array.Resize(ref bytes, bytes.Length - 1);

            return Encoding.ASCII.GetString(bytes);
        }

        private static async Task DrainStandardErrorAsync(StreamReader standardError)
        {
            try
            {
                while (await standardError.ReadLineAsync() is not null)
                {
                }
            }
            catch
            {
            }
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
}
