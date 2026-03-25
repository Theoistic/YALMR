---
layout: page
title: Tool Calling
permalink: /features/tool-calling/
parent: Features
nav_order: 2
---

# Tool Calling

YALMR lets the model call C# methods during a conversation. When the model decides a tool should be invoked, YALMR executes the handler automatically and feeds the result back before resuming generation — all transparent to the caller.

---

## Attribute-based registration

Decorate methods with `[Tool]` and parameters with `[ToolParam]`, then register the class instance with the `ToolRegistry`:

```csharp
public class AssistantTools
{
    [Tool("Returns the current date and time.")]
    public string GetDateTime() => DateTimeOffset.Now.ToString("R");

    [Tool("Returns the weather forecast for a city.")]
    public string GetWeather(
        [ToolParam("City name.")]                                string city,
        [ToolParam("Temperature units: celsius or fahrenheit.")] string units = "celsius")
        => $"22 °{(units == "celsius" ? "C" : "F")}, sunny in {city}.";
}
```

Register and attach to a session:

```csharp
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

The session calls handlers automatically. Method names are converted to `snake_case` (`GetWeather` → `get_weather`). Override the generated name with the `Name` property:

```csharp
[Tool("Returns the weather.", Name = "weather_lookup")]
public string GetWeather(...) { ... }
```

---

## Manual registration with `AgentTool`

For programmatic or stateful tools that don't fit the method-per-class model:

```csharp
registry.Register(new AgentTool(
    "search_products",
    "Searches the product catalogue.",
    [
        new ToolParameter("query", "string",  "Search query."),
        new ToolParameter("limit", "integer", "Max results to return.", Required: false),
    ],
    args =>
    {
        string query = (string)args["query"];
        int    limit = args.TryGetValue("limit", out var l) && l is long n ? (int)n : 10;
        return SearchProducts(query, limit);
    }));
```

### Argument type mapping

When writing `AgentTool` handlers manually, argument values arrive as these CLR types:

| JSON Schema type | CLR type | Notes |
|---|---|---|
| `"string"` | `string` | |
| `"integer"` | `long` | |
| `"number"` | `double` | Cast to `decimal` if needed: `(decimal)(double)args["price"]` |
| `"boolean"` | `bool` | |
| `"array"` | `object?[]` | Elements follow the same rules |
| `"object"` | `Dictionary<string, object?>` | |

The reflection-based `Register(instance)` overload coerces types automatically, so this table only matters for `AgentTool` handlers written by hand.

---

## Registering with a DI container

If your tools depend on injected services, resolve them through your `IServiceProvider`:

```csharp
registry.Register<AssistantTools>(serviceProvider);
```

---

## Per-session tools with `YALMRServer`

When using `YALMRServer` you can assign a different tool registry to each session, which is useful when tools carry per-request state (e.g. the current user's shopping cart):

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
    ChatMessage reply = await server.GetSession(sessionId)
        .SendAsync(new ChatMessage("user", "What time is it?"));
    Console.WriteLine(reply.Content);
}
finally
{
    await server.RemoveSessionAsync(sessionId);
}
```

---

## Controlling tool round-trips

By default a session allows up to 10 consecutive tool-call / tool-result cycles before returning. Adjust this per session:

```csharp
new SessionOptions
{
    ...
    MaxToolRounds = 5,
}
```

---

## Tool execution at runtime

YALMR's `ToolRegistry` is thread-safe. You can add or remove tools between turns:

```csharp
registry.Remove("search_products");        // remove a tool
registry.Register(new AgentTool(...));     // add a new one
registry.Clear();                          // remove all
```

Changes take effect on the next `GenerateAsync` / `SendAsync` call.
