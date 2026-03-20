# YALMR

**Y**et **A**nother **L**LM **R**untime — a .NET 10 library for running local GGUF models via llama.cpp with built-in tool calling, conversation compaction, vision, MCP server integration, and a streaming API.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Runtime Installation](#runtime-installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
   - [Engine & Session](#engine--session)
   - [InferenceOptions](#inferenceoptions)
   - [SessionOptions](#sessionoptions)
5. [Tool Calling](#tool-calling)
6. [Streaming & Renderers](#streaming--renderers)
7. [Conversation Compaction](#conversation-compaction)
8. [Response API Style](#response-api-style)
9. [Vision / Multimodal](#vision--multimodal)
10. [Multi-model Server](#multi-model-server)
11. [MCP Tool Integration](#mcp-tool-integration)
12. [Diagnostics & Logging](#diagnostics--logging)

---

## Prerequisites

- .NET 10 SDK
- A GGUF model file (e.g. from [Hugging Face](https://huggingface.co/models?library=gguf) or [LM Studio](https://lmstudio.ai))
- llama.cpp native binaries (automatically installed — see below)

---

## Runtime Installation

YALMR ships `LlamaRuntimeInstaller` to download the correct llama.cpp native build from the official GitHub releases. Call it once at startup.

```csharp
using YALMR.LlamaCpp;

// Downloads the CPU build if not already present.
// Returns the directory path containing the native binaries.
string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

// GPU variants:
string cudaDir   = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cuda);
string vulkanDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Vulkan);
```

Binaries are cached under `%LOCALAPPDATA%\YALMR\llama-runtime` (Windows) or `~/.local/share/YALMR/llama-runtime` (Linux/macOS). Subsequent calls return immediately if the runtime is already present.

```csharp
// Check without downloading:
string? existing = LlamaRuntimeInstaller.FindInstalled(LlamaBackend.Cpu);
```

---

## Quick Start

```csharp
using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;

string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

var options = new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/model.gguf",
    ToolRegistry     = new ToolRegistry(),
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
};

await using var session = await Session.CreateAsync(options);

// Streaming chat
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Hello!")))
    Console.Write(chunk.Text);
```

---

## Core Concepts

### Engine & Session

```
┌─────────────────────────────────────────┐
│              Engine                     │
│  (owns the native model + vision ctx)   │
│                                         │
│   Session ──── InferenceContext         │
│   Session ──── InferenceContext         │
│   Session ──── InferenceContext         │
└─────────────────────────────────────────┘
```

- **`Engine`** loads the GGUF model once and holds the shared native model handle and optional vision context. It is thread-safe across sessions.
- **`Session`** owns per-conversation state: history, KV cache, compaction, and tool execution. Each session is independent; multiple sessions may share one engine.
- **`InferenceContext`** wraps the native llama context and KV cache for one session.

```csharp
// Shared engine, multiple sessions:
var engine = await Engine.CreateAsync(options);
var sessionA = Session.Create(optionsA, engine);
var sessionB = Session.Create(optionsB, engine);
```

### InferenceOptions

`InferenceOptions` controls how a response is generated. It is fully decoupled from request routing (`ResponseRequest`) and conversation history.

```csharp
var inference = new InferenceOptions
{
    Temperature     = 0.7f,       // 0 = greedy, >0 = stochastic
    TopP            = 0.9f,       // nucleus sampling
    TopK            = 40,         // top-k sampling (0 = disabled)
    PresencePenalty  = 0.0f,
    FrequencyPenalty = 0.0f,
    RepetitionPenalty = 1.1f,
    MaxOutputTokens = 1024,
    Seed            = 42,         // null for Random.Shared
    EnableThinking  = false,      // <think> block stripping
    ReasoningEffort = "none",     // "none" | "low" | "high" (maps to EnableThinking)
    AddVisionId     = false,
    Tools           = [ /* ToolDefinition list */ ],
};
```

Set `SessionOptions.DefaultInference` for session-wide defaults. Per-request values passed to `CreateResponseAsync` / `GenerateResponseAsync` are merged over the defaults at call time.

### SessionOptions

Full configuration reference:

| Property | Default | Purpose |
|---|---|---|
| `ModelPath` | *(required)* | Path to the `.gguf` file |
| `BackendDirectory` | `""` | Directory containing `llama.dll` / `libllama.so` |
| `ToolRegistry` | *(required)* | Registered tool handlers |
| `Compaction` | *(required)* | Token-window compaction policy |
| `DefaultInference` | `new InferenceOptions()` | Session-level generation defaults |
| `ContextTokens` | `8192` | Native context window size |
| `BatchTokens` | `1024` | Prompt encode batch size |
| `MicroBatchTokens` | `1024` | ubatch size (fine-grained decode) |
| `Threads` | `null` (auto) | CPU thread count |
| `MaxToolRounds` | `10` | Max tool call / re-generation loops |
| `FlashAttention` | `false` | Enable flash attention |
| `OffloadKvCacheToGpu` | `true` | KV cache GPU offload |
| `KvCacheTypeK/V` | `null` | KV cache quantization |
| `MmprojPath` | `null` (auto-detect) | Vision projector `.gguf` path |
| `UseGpuForVision` | `true` | GPU for vision encoding |
| `VisionImageMaxTokens` | `1024` | Max token budget per image |
| `ImageRetentionPolicy` | `KeepAllImages` | Drop old images under pressure |
| `UseMmap` | `true` | Memory-mapped model loading |
| `RopeFrequencyBase/Scale` | `null` | RoPE frequency override |

---

## Tool Calling

Tools are registered with `ToolRegistry` (provides execution handlers) and declared in `InferenceOptions.Tools` (provides JSON schemas to the model).

### 1 — Register tools

```csharp
ToolRegistry registry = new()
{
    new AgentTool(
        name:        "get_weather",
        description: "Returns the current weather for a city.",
        parameters: [
            new ToolParameter("city",    "string", "City name."),
            new ToolParameter("units",   "string", "\"celsius\" or \"fahrenheit\".", Required: false),
        ],
        handler: args =>
        {
            string city  = args.TryGetValue("city",  out var c) ? c?.ToString() ?? "" : "";
            string units = args.TryGetValue("units", out var u) ? u?.ToString() ?? "celsius" : "celsius";
            // call a real weather API here
            return $"The weather in {city} is 22°{(units == "fahrenheit" ? "F" : "C")}, sunny.";
        }),
};
```

### 2 — Build ToolDefinition schemas

The model receives tool schemas via `InferenceOptions.Tools`. Create a `ToolDefinition` for each registered tool:

```csharp
static ToolDefinition ToDefinition(AgentTool tool)
{
    var props = new Dictionary<string, object>(StringComparer.Ordinal);
    foreach (var p in tool.Parameters)
        props[p.Name] = new { type = p.Type, description = p.Description };

    string[] required = tool.Parameters.Where(p => p.Required).Select(p => p.Name).ToArray();

    return new ToolDefinition(
        Type:        "function",
        Name:        tool.Name,
        Description: tool.Description,
        Parameters:  new { type = "object", properties = props, required });
}

InferenceOptions inference = new()
{
    Tools = [.. registry.Select(ToDefinition)],
};
```

### 3 — Wire it up

```csharp
SessionOptions options = new()
{
    ToolRegistry     = registry,   // handles execution
    DefaultInference = inference,  // exposes schemas to the model
    // ...
};
```

Tool calls are **handled automatically** inside `GenerateAsync` / `GenerateResponseAsync`. When the model emits a tool call the session executes the handler, appends the result, and continues generation without any caller involvement.

### Remote tools (non-local handlers)

```csharp
// Declare a tool that routes to a remote process or URL
new AgentTool(
    "search_docs",
    "Searches the documentation corpus.",
    [new ToolParameter("query", "string", "Search query.")],
    remoteDefinition: new AgentToolRemoteDefinition(
        Server:   "docs-mcp",
        ToolName: "search_docs",
        Metadata: new McpServerProcessOptions("npx", ["-y", "@docs/mcp-server"])))
```

---

## Streaming & Renderers

### Raw streaming

```csharp
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Tell me a joke")))
{
    if (chunk.Text is not null)           Console.Write(chunk.Text);
    if (chunk.ReasoningText is not null)  Console.Write($"[think]{chunk.ReasoningText}");
    if (chunk.ToolCalls is { Count: > 0}) Console.WriteLine($"\n[tool call: {chunk.ToolCalls[0].Name}]");
    if (chunk.Usage is not null)          Console.WriteLine($"\n[tokens: {chunk.Usage.TotalTokens}]");
}
```

### ConsoleChatRenderer

`ConsoleChatRenderer` handles reasoning blocks, tool call annotations, usage lines, and error formatting out of the box:

```csharp
var renderer = new ConsoleChatRenderer(Console.Out);

renderer.BeginAssistantMessage();
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Hello")))
    renderer.Render(chunk);
renderer.EndAssistantMessage();
```

Implement `StreamingChatRenderer` (or `IChatRenderer`) to build custom output targets (Markdown, HTML, SignalR, etc.).

### Debug view

```csharp
session.DebugViewCreated += (_, view) =>
{
    Console.WriteLine($"prompt_tokens : {view.PromptTokens}");
    Console.WriteLine($"rendered prompt:\n{view.RenderedPrompt}");
};
```

Or pass an instance of `ConsoleChatRenderer` and call `renderer.RenderDebug(view)` for full structured output.

---

## Conversation Compaction

When the conversation grows beyond the context window YALMR automatically trims it. Configure via `ConversationCompactionOptions`:

```csharp
new ConversationCompactionOptions(
    MaxInputTokens:        8192,
    ReservedForGeneration: 512,           // kept free for output
    Strategy:              ContextCompactionStrategy.PinnedSystemFifo,
    Level:                 ConversationCompactionLevel.Balanced,
    AlwaysKeepSystem:      true,          // never drops the system message
    HotTrailMessages:      4              // always keeps the last N messages
)
```

**Built-in strategies:**

| Strategy | Behaviour |
|---|---|
| `FifoSlidingWindow` | Drops oldest messages first, no pinning |
| `PinnedSystemFifo` | Pins the system message, then FIFO *(default)* |
| `MiddleOutElision` | Keeps head + tail, removes from the middle |
| `HeuristicPruning` | Pins system, drops low-information turns |
| `RollingSummarization` | Requires a custom `IConversationCompactor` |
| `VectorAugmentedRecall` | Requires a custom `IConversationCompactor` |

Provide a custom compactor:

```csharp
public sealed class MySummarizingCompactor : IConversationCompactor
{
    public IReadOnlyList<ChatMessage> Compact(
        IReadOnlyList<ChatMessage> messages,
        ConversationCompactionContext ctx)
    {
        // ctx.CountTokens(messages) gives the token count
        // ctx.Options exposes MaxInputTokens, TokenBudget, etc.
        // Return the trimmed list
    }
}

new SessionOptions { ConversationCompactor = new MySummarizingCompactor(), ... }
```

---

## Response API Style

`CreateResponseAsync` / `GenerateResponseAsync` follow an OpenAI Responses-API-style pattern where each call is self-contained and conversation state is chained via `PreviousResponseId`.

```csharp
// Turn 1 — include system message in Input
var response1 = await session.CreateResponseAsync(new ResponseRequest
{
    Model  = "my-model",
    Input  =
    [
        new ChatMessage("system", "You are a helpful assistant."),
        new ChatMessage("user",   "What is 2 + 2?"),
    ],
    Inference = new InferenceOptions { Temperature = 0.0f },
});

Console.WriteLine(response1.Output[^1].Content);  // "4"

// Turn 2 — chain history via the previous response ID
var response2 = await session.CreateResponseAsync(new ResponseRequest
{
    Model              = "my-model",
    Input              = [new ChatMessage("user", "And multiplied by 3?")],
    PreviousResponseId = response1.Id,          // replays stored history
    Inference          = new InferenceOptions { Temperature = 0.0f },
});
```

Use `GenerateResponseAsync` for the streaming variant:

```csharp
await foreach (var chunk in session.GenerateResponseAsync(request))
    Console.Write(chunk.Text);

string? nextId = session.LastResponse?.Id;
```

`ResponseObject` carries the new output messages and aggregated token usage:

```csharp
public sealed record ResponseObject(
    string                    Id,
    long                      CreatedAt,
    string                    Model,
    IReadOnlyList<ChatMessage> Output,       // new messages this turn
    InferenceUsage             Usage,
    string?                   PreviousResponseId);
```

---

## Vision / Multimodal

Vision requires a multimodal projector (`mmproj`) GGUF alongside the base model.

```csharp
var options = new SessionOptions
{
    ModelPath           = "path/to/Qwen2.5-VL.gguf",
    MmprojPath          = "path/to/mmproj.gguf",  // auto-detected if null
    UseGpuForVision     = true,
    VisionImageMaxTokens = 1024,
    ImageRetentionPolicy = ImageRetentionPolicy.KeepLatestImage,  // saves memory
    // ...
};
```

Attach images via `ImagePart`:

```csharp
var message = new ChatMessage(
    Role:  "user",
    Parts:
    [
        new TextPart("What is in this image?"),
        ImagePart.FromFile("photo.jpg"),
        // or: ImagePart.FromBytes(imageBytes)
        // or: new ImagePart(base64String)
    ]);

await foreach (var chunk in session.GenerateAsync(message))
    Console.Write(chunk.Text);
```

Check vision availability at runtime:

```csharp
if (!session.VisionEnabled)
    Console.WriteLine($"Vision unavailable: {session.VisionDisabledReason}");
```

---

## Multi-model Server

`YALMRServer` manages a pool of named engines and sessions, suitable for serving multiple models or concurrent users.

```csharp
await using var server = new YALMRServer();

// Load models
await server.LoadModelAsync("llm",    optionsLlm);
await server.LoadModelAsync("vision", optionsVision);

// Create sessions (returns a session ID)
string sessionId = server.CreateSession("llm");

// Use the session
Session session = server.GetSession(sessionId);

// Tear down
await server.RemoveSessionAsync(sessionId);
await server.UnloadModelAsync("llm");
```

Sessions share the engine (weights) but each has its own KV cache and history. You can also supply per-session overrides:

```csharp
string sessionId = server.CreateSession(
    modelId:        "llm",
    sessionOptions: options with { DefaultInference = new InferenceOptions { Temperature = 0.0f } });
```

---

## MCP Tool Integration

YALMR supports calling tools exposed by Model Context Protocol (MCP) servers — both HTTP and stdio-spawned processes.

### HTTP MCP server

```csharp
var request = new ResponseRequest
{
    Input = [new ChatMessage("user", "Search the web for YALMR")],
    Inference = new InferenceOptions
    {
        Tools =
        [
            new ToolDefinition(
                Type:       "mcp",
                Name:       "brave-search",
                ServerLabel: "Brave Search",
                ServerUrl:  "https://mcp.bravetools.com",
                Headers:    new Dictionary<string, string> { ["Authorization"] = "Bearer <key>" }),
        ],
    },
};
```

### stdio MCP server

Register an `AgentTool` with a `AgentToolRemoteDefinition` whose `Metadata` is a `McpServerProcessOptions`. The session spawns the process and communicates over stdio with JSON-RPC.

```csharp
new AgentTool(
    "read_file",
    "Reads a file via the filesystem MCP server.",
    [new ToolParameter("path", "string", "Absolute path to the file.")],
    remoteDefinition: new AgentToolRemoteDefinition(
        Server:   "fs-server",
        ToolName: "read_file",
        Metadata: new McpServerProcessOptions(
            Command:   "npx",
            Arguments: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"])))
```

Add `McpRemoteToolExecutor` to the session's tool execution engine (it is included by default):

```csharp
new SessionOptions
{
    ToolExecutionEngine = new DefaultToolExecutionEngine([new McpRemoteToolExecutor()]),
    // ...
}
```

---

## Diagnostics & Logging

### Logging

Provide an `ILogger` implementation in `SessionOptions`:

```csharp
// Built-in: writes to stderr
new SessionOptions { Logger = new ConsoleErrorLogger() }

// Built-in: write to any TextWriter
new SessionOptions { Logger = new TextWriterLogger(myWriter) }

// Built-in: silence everything
new SessionOptions { Logger = NullLogger.Instance }

// Custom:
public sealed class MyLogger : ILogger
{
    public void Log(LogLevel level, string category, string message)
        => myLoggingFramework.Log(level, $"[{category}] {message}");
}
```

### OpenTelemetry

YALMR emits OpenTelemetry traces and metrics via `RuntimeTelemetry`. Register a standard OTLP exporter in your host to receive:

- **Spans:** `runtime.session.create`, `runtime.session.create_response`, `yalmr.context.generate_tokens`, `yalmr.tool.execute`
- **Counters:** `yalmr.sessions_created`, `yalmr.inference_calls`, `yalmr.tokens_generated`, `yalmr.tool_executions`
- **Histograms:** `yalmr.inference_duration_ms`, `yalmr.tool_execution_duration_ms`
