# YALMR

[![NuGet](https://img.shields.io/nuget/v/YALMR.svg)](https://www.nuget.org/packages/YALMR)

**Y**et **A**nother L**LM** **R**untime — run local GGUF models in .NET 10 via llama.cpp.

> Requires .NET 10 SDK and a GGUF model file (e.g. from [Hugging Face](https://huggingface.co/models?library=gguf)).

---

## Install the runtime

YALMR downloads the correct llama.cpp native binaries automatically. Call once at startup:

```csharp
using YALMR.LlamaCpp;

string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);
// LlamaBackend.Cuda or LlamaBackend.Vulkan for GPU
```

Binaries are cached in `%LOCALAPPDATA%\YALMR\llama-runtime` (Windows) or `~/.local/share/YALMR/llama-runtime` (Linux/macOS).

---

## Start a session

```csharp
using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;

string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/model.gguf",
    ToolRegistry     = new ToolRegistry(),
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
});
```

---

## Chat

```csharp
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Hello!")))
    Console.Write(chunk.Text);
```

Multi-turn conversation is tracked automatically. Each call to `GenerateAsync` appends to the session history.

---

## Structured output

Use `AskAsync<T>()` to get a typed response. The model's sampling is constrained via GBNF grammar so the output is always valid JSON matching your type's schema — no prompt-engineering or retry logic required.

```csharp
public record MovieReview(string Title, int ReleaseYear, double Score, string Summary);

var review = await session.AskAsync<MovieReview>(
    "Review the movie Interstellar.");

Console.WriteLine($"{review.Title} ({review.ReleaseYear}) — {review.Score}/10");
Console.WriteLine(review.Summary);
```

Any JSON-serializable type works: records, classes, and collections. Optional properties (nullable or with defaults) may be returned as `null` by the model.

---

## Features

- **Tool calling** — register handlers in `ToolRegistry`; the model calls them automatically
- **Streaming** — `IAsyncEnumerable<ChatResponseChunk>` with text, reasoning, and tool-call chunks
- **Structured output** — `AskAsync<T>()` constrains sampling via GBNF grammar and deserializes the result
- **Vision** — attach images via `ImagePart` with a multimodal projector model
- **Conversation compaction** — automatic context-window management with pluggable strategies
- **MCP integration** — call tools from HTTP or stdio MCP servers
- **Multi-model server** — `YALMRServer` manages named engines and sessions for concurrent use

---

## Tool calling

Decorate methods with `[Tool]` and parameters with `[ToolParam]`, then register the instance:

```csharp
public class AssistantTools
{
    [Tool("Returns the current date and time.")]
    public string GetDateTime() => DateTimeOffset.Now.ToString("R");

    [Tool("Returns the weather for a city.")]
    public string GetWeather(
        [ToolParam("City name.")] string city,
        [ToolParam("Temperature units: celsius or fahrenheit.")] string units = "celsius")
        => $"22 degrees, sunny in {city}.";
}

ToolRegistry registry = new();
registry.Register(new AssistantTools());

await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/model.gguf",
    ToolRegistry     = registry,
    DefaultInference = new InferenceOptions { Tools = registry.ToToolDefinitions() },
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
});
```

The session calls matching handlers automatically. Method names become snake_case (`GetWeather` to `get_weather`). Override with `[Tool("...", Name = "custom_name")]`.

### With a DI container

```csharp
registry.Register<AssistantTools>(serviceProvider);
```

### With YALMRServer (per-request tools)

```csharp
if (!server.TryGetModelOptions("my-model", out var baseOpts))
    throw new InvalidOperationException("Model not loaded.");

var registry = new ToolRegistry();
registry.Register(new AssistantTools());

var sessionId = server.CreateSession("my-model", baseOpts with
{
    ToolRegistry     = registry,
    DefaultInference = (baseOpts.DefaultInference ?? new InferenceOptions()) with
    {
        Tools = registry.ToToolDefinitions(),
    },
});
try
{
    var reply = await server.GetSession(sessionId)
        .SendAsync(new ChatMessage("user", "What time is it?"));
    Console.WriteLine(reply.Content);
}
finally { await server.RemoveSessionAsync(sessionId); }
```

---

### Argument types

When writing handlers manually, argument values have these CLR types:

| JSON Schema type | CLR type | Note |
|---|---|---|
| `"string"` | `string` | |
| `"integer"` | `long` | |
| `"number"` | `double` | Cast if you need `decimal`: `(decimal)(double)args["price"]` |
| `"boolean"` | `bool` | |
| `"array"` | `object?[]` | Elements follow the same rules |
| `"object"` | `Dictionary<string, object?>` | |

The reflection-based `Register` coerces all types automatically. The table above only matters when writing `AgentTool` handlers directly.

### Manual registration

For programmatic or stateful tools that don't fit the method-per-tool model:

```csharp
registry.Register(new AgentTool(
    "search_products",
    "Searches the product catalogue.",
    [
        new ToolParameter("query", "string",  "Search query."),
        new ToolParameter("limit", "integer", "Max results.", Required: false),
    ],
    args =>
    {
        string query = (string)args["query"];
        int    limit = args.TryGetValue("limit", out var l) && l is long n ? (int)n : 10;
        return SearchProducts(query, limit);
    }));
```
