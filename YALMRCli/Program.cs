using System.Data;
using YALMR.Diagnostics;
using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;
using YALMR.Web;

Console.OutputEncoding = System.Text.Encoding.UTF8;

// ── Runtime backend ───────────────────────────────────────────────────────────

string backendDir = await EnsureBackendAsync();

// ── Subcommand dispatch ───────────────────────────────────────────────────────

if (args.Length > 0 && string.Equals(args[0], "serve", StringComparison.OrdinalIgnoreCase))
{
    string serveModel = args.Length > 1 && !args[1].StartsWith('-')
        ? args[1]
        : @"C:\Users\Theo\.lmstudio\models\lmstudio-community\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf";

    string serveUrl = "http://localhost:5000";
    for (int i = 1; i < args.Length - 1; i++)
        if (args[i] == "--url") { serveUrl = args[i + 1]; break; }

    if (!File.Exists(serveModel))
    {
        Console.Error.WriteLine($"error: model not found: {serveModel}");
        Console.Error.WriteLine("usage: YALMRCli serve [path/to/model.gguf] [--url <url>]");
        return 1;
    }

    string serveModelId = Path.GetFileNameWithoutExtension(serveModel);

    SessionOptions serveOptions = new()
    {
        BackendDirectory = backendDir,
        ModelPath        = serveModel,
        ToolRegistry     = [],
        Compaction       = new ConversationCompactionOptions(
            MaxInputTokens:        32768,
            ReservedForGeneration: 8192,
            Strategy:              ContextCompactionStrategy.PinnedSystemFifo),
        DefaultInference = new InferenceOptions { MaxOutputTokens = 8192 },
        GpuLayers        = 99,
        FlashAttention   = true,
        KvCacheTypeK     = KvCacheQuantization.Q8_0,
        KvCacheTypeV     = KvCacheQuantization.Q8_0,
        ContextTokens    = 32768,
    };

    Console.Write($"Loading {Path.GetFileName(serveModel)} ...");
    await using var webServer = new YALMRServer();
    await webServer.LoadModelAsync(serveModelId, serveOptions);
    Console.WriteLine(" ready.\n");

    await using var api = new YALMRApiServer(webServer);
    await api.StartAsync(serveUrl);

    Console.WriteLine($"Listening on  {serveUrl}");
    Console.WriteLine($"  Chat UI  →  {serveUrl}/chat");
    Console.WriteLine($"  API      →  {serveUrl}/v1/health");
    Console.WriteLine("\nPress Ctrl+C to stop.\n");

    var cts = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

    try { await Task.Delay(Timeout.Infinite, cts.Token); }
    catch (OperationCanceledException) { }

    Console.WriteLine("\nShutting down…");
    await api.StopAsync();
    return 0;
}

// ── Model path ────────────────────────────────────────────────────────────────

string modelPath = args.Length > 0
    ? args[0]
    : @"C:\Users\theo\.lmstudio\models\lmstudio-community\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf";

if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"error: model not found: {modelPath}");
    Console.Error.WriteLine("usage: YALMRCli [path/to/model.gguf]");
    return 1;
}

// ── Tool registry ─────────────────────────────────────────────────────────────

ToolRegistry registry = new()
{
    new AgentTool(
        "get_datetime",
        "Returns the current local date and time.",
        [],
        _ => DateTimeOffset.Now.ToString("ddd, dd MMM yyyy HH:mm:ss zzz")),

    new AgentTool(
        "calculate",
        "Evaluates a basic arithmetic expression and returns the numeric result.",
        [new ToolParameter("expression", "string", "Arithmetic expression, e.g. \"(4 + 8) / 3\".")],
        args =>
        {
            string expr = GetArg(args, "expression");
            try   { return new DataTable().Compute(expr, null)?.ToString() ?? "null"; }
            catch (Exception ex) { return $"Error: {ex.Message}"; }
        }),

    new AgentTool(
        "list_directory",
        "Lists files and folders at a given path.",
        [new ToolParameter("path", "string", "Directory path to list. Omit for current directory.", Required: false)],
        args =>
        {
            string dir = GetArg(args, "path", Environment.CurrentDirectory);
            if (!Directory.Exists(dir)) return $"Directory not found: {dir}";
            string[] entries = Directory.GetFileSystemEntries(dir)
                .Select(Path.GetFileName).Take(40).ToArray()!;
            return entries.Length == 0 ? "(empty)" : string.Join("\n", entries);
        }),
};

// ── Session options ───────────────────────────────────────────────────────────

const int contextTokens = 8192;

InferenceOptions inference = new()
{
    Temperature    = 0.7f,
    MaxOutputTokens = 1024,
    EnableThinking = false,
    Tools          = [.. registry.Select(ToolToDefinition)],
};

SessionOptions options = new()
{
    BackendDirectory = backendDir,
    ModelPath        = modelPath,
    GpuLayers        = 99,
    FlashAttention   = true,
    KvCacheTypeK     = KvCacheQuantization.Q8_0,
    KvCacheTypeV     = KvCacheQuantization.Q8_0,
    ToolRegistry     = registry,
    ContextTokens    = contextTokens,
    Compaction       = new ConversationCompactionOptions(
        MaxInputTokens:       contextTokens,
        ReservedForGeneration: 512,
        Strategy:             ContextCompactionStrategy.PinnedSystemFifo),
    DefaultInference = inference,
};

// ── Load model ────────────────────────────────────────────────────────────────

Console.Write($"Loading {Path.GetFileName(modelPath)} ...");
await using var session = await Session.CreateAsync(options);
Console.WriteLine(" ready.\n");

// ── REPL ──────────────────────────────────────────────────────────────────────

var renderer      = new ConsoleChatRenderer(Console.Out);
bool debugEnabled = false;
string system     = "You are a helpful assistant. Use tools when they are the best way to answer.";
string? prevId    = null;       // chains response history across turns

session.DebugViewCreated += (_, view) => { if (debugEnabled) renderer.RenderDebug(view); };

PrintHelp();

while (true)
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write("\nuser> ");
    Console.ResetColor();

    string? line = Console.ReadLine()?.Trim();
    if (line is null || string.Equals(line, "/quit", StringComparison.OrdinalIgnoreCase))
        break;
    if (line.Length == 0)
        continue;

    // ── Commands ──────────────────────────────────────────────────────────────

    if (line.StartsWith('/'))
    {
        string[] parts = line.Split(' ', 2, StringSplitOptions.TrimEntries);
        switch (parts[0].ToLowerInvariant())
        {
            case "/help":
                PrintHelp();
                break;

            case "/reset":
                session.Reset();
                prevId = null;
                Console.WriteLine("Conversation cleared.");
                break;

            case "/system":
                if (parts.Length < 2 || string.IsNullOrWhiteSpace(parts[1]))
                    Console.WriteLine($"system> {system}");
                else
                {
                    system = parts[1];
                    session.Reset();
                    prevId = null;
                    Console.WriteLine("System prompt updated and conversation cleared.");
                }
                break;

            case "/history":
                PrintHistory(session);
                break;

            case "/debug":
                debugEnabled = !debugEnabled;
                Console.WriteLine($"Debug output {(debugEnabled ? "on" : "off")}.");
                break;

            default:
                Console.WriteLine($"Unknown command '{parts[0]}'. Type /help.");
                break;
        }
        continue;
    }

    // ── Inference turn ────────────────────────────────────────────────────────

    // System message is injected on the first turn only; subsequent turns chain
    // via PreviousResponseId so the session re-uses its stored history.
    List<ChatMessage> input = [];
    if (prevId is null)
        input.Add(new ChatMessage("system", system));
    input.Add(new ChatMessage("user", line));

    var request = new ResponseRequest
    {
        Model              = Path.GetFileNameWithoutExtension(modelPath),
        Input              = input,
        PreviousResponseId = prevId,
        Inference          = inference,
    };

    Console.WriteLine();
    renderer.BeginAssistantMessage();
    try
    {
        await foreach (var chunk in session.GenerateResponseAsync(request).ConfigureAwait(false))
            renderer.Render(chunk);

        prevId = session.LastResponse?.Id;
    }
    catch (Exception ex)
    {
        renderer.RenderError(ex);
    }
    renderer.EndAssistantMessage();
}

Console.WriteLine("\nGoodbye.");
return 0;

// ── Local helpers ─────────────────────────────────────────────────────────────

static async Task<string> EnsureBackendAsync()
{
    string? existing = LlamaRuntimeInstaller.FindInstalled(LlamaBackend.Cuda);
    if (existing is not null)
        return existing;

    Console.WriteLine("llama.cpp runtime not found — downloading CPU build from GitHub...");
    return await LlamaRuntimeInstaller.EnsureInstalledAsync(
        LlamaBackend.Cuda,
        cudaVersion: "12.6",
        progress: new Progress<(string msg, double pct)>(t =>
        {
            Console.Write($"\r  {t.msg}  {t.pct:F0}%     ");
            if (t.pct >= 100) Console.WriteLine();
        }));
}

// Converts a registry AgentTool into the ToolDefinition schema the template expects.
static ToolDefinition ToolToDefinition(AgentTool tool)
{
    var props = new Dictionary<string, object>(StringComparer.Ordinal);
    foreach (var p in tool.Parameters)
        props[p.Name] = new { type = p.Type, description = p.Description };

    string[] required = tool.Parameters
        .Where(p => p.Required)
        .Select(p => p.Name)
        .ToArray();

    return new ToolDefinition(
        Type:        "function",
        Name:        tool.Name,
        Description: tool.Description,
        Parameters:  new { type = "object", properties = props, required });
}

static string GetArg(IReadOnlyDictionary<string, object?> args, string key, string fallback = "")
    => args.TryGetValue(key, out var v) && v is not null ? v.ToString()! : fallback;

static void PrintHistory(Session session)
{
    var history = session.History;
    if (history.Count == 0) { Console.WriteLine("(no history)"); return; }

    Console.WriteLine($"─── history ({history.Count} messages) ───────────────");
    foreach (var msg in history)
    {
        string preview = msg.Content is { Length: > 0 } c
            ? (c.Length > 100 ? c[..100] + "…" : c)
            : msg.ToolCalls is { Count: > 0 }
                ? $"[tool calls: {string.Join(", ", msg.ToolCalls.Select(t => t.Name))}]"
                : "(no content)";

        Console.ForegroundColor = msg.Role switch
        {
            "user"      => ConsoleColor.Cyan,
            "assistant" => ConsoleColor.White,
            "system"    => ConsoleColor.DarkYellow,
            _           => ConsoleColor.DarkGray,
        };
        Console.Write($"  {msg.Role,-10}");
        Console.ResetColor();
        Console.WriteLine($" {preview}");
    }
    Console.WriteLine("──────────────────────────────────────────────");
}

static void PrintHelp()
{
    Console.WriteLine("CLI usage:");
    Console.WriteLine("  YALMRCli [model.gguf]                         interactive REPL");
    Console.WriteLine("  YALMRCli serve [model.gguf] [--url <url>]     start web API + chat UI");
    Console.WriteLine();
    Console.WriteLine("  Reasoning and vision capabilities are detected automatically from the model.");
    Console.WriteLine();
    Console.WriteLine("REPL commands:");
    Console.WriteLine("  /help              show this help");
    Console.WriteLine("  /reset             clear conversation history");
    Console.WriteLine("  /system [text]     show or set the system prompt (clears history)");
    Console.WriteLine("  /history           print conversation history");
    Console.WriteLine("  /debug             toggle prompt debug output");
    Console.WriteLine("  /quit              exit");
    Console.WriteLine();
}
