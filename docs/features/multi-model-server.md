---
layout: page
title: Multi-Model Server
permalink: /features/multi-model-server/
parent: Features
nav_order: 7
---

# Multi-Model Server

`YALMRServer` and `YALMRApiServer` let you host multiple models in a single process and serve them over an OpenAI-compatible HTTP API — useful for building chatbots, internal tools, or local alternatives to cloud LLM APIs.

---

## `YALMRServer` — in-process orchestration

`YALMRServer` manages named engines and sessions. A single engine can back many concurrent sessions, so the model weights are loaded into memory only once.

### Load a model

```csharp
await using var server = new YALMRServer();

string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

await server.LoadModelAsync("llama3", new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/llama3.gguf",
    ToolRegistry     = new ToolRegistry(),
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
    GpuLayers        = -1,                     // offload all layers to GPU
    FlashAttention   = true,
});
```

### Create sessions

```csharp
string sessionId = server.CreateSession("llama3");

Session session = server.GetSession(sessionId);
ChatMessage reply = await session.SendAsync(new ChatMessage("user", "Hello!"));
Console.WriteLine(reply.Content);
```

### Per-session tool registries

Override any session option — including the tool registry — without affecting the shared engine:

```csharp
if (!server.TryGetModelOptions("llama3", out var baseOpts))
    throw new InvalidOperationException("Model not loaded.");

var registry = new ToolRegistry();
registry.Register(new UserSpecificTools(currentUser));

string sessionId = server.CreateSession("llama3", baseOpts with
{
    ToolRegistry     = registry,
    DefaultInference = (baseOpts.DefaultInference ?? new InferenceOptions()) with
    {
        Tools = registry.ToToolDefinitions(),
    },
});
```

### Remove sessions

```csharp
await server.RemoveSessionAsync(sessionId);
```

### Unload a model

All sessions using the model must be removed before it can be unloaded:

```csharp
await server.UnloadModelAsync("llama3");
```

### Query capabilities

```csharp
if (server.TryGetModelCapabilities("llama3", out bool visionEnabled, out bool thinkingEnabled))
{
    Console.WriteLine($"Vision: {visionEnabled}, Thinking: {thinkingEnabled}");
}
```

---

## `YALMRApiServer` — HTTP API

`YALMRApiServer` wraps a `YALMRServer` and exposes an OpenAI-compatible REST API plus a built-in chat UI.

### Start the server

```csharp
string serveUrl = "http://localhost:5000";

await using var webServer = new YALMRServer();
await webServer.LoadModelAsync("my-model", new SessionOptions { ... });

await using var api = new YALMRApiServer(webServer);
await api.StartAsync(serveUrl);

Console.WriteLine($"Chat UI  → {serveUrl}/chat");
Console.WriteLine($"API      → {serveUrl}/v1/health");
```

### Available endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (streaming supported) |
| `GET` | `/chat` | Browser-based chat UI |

### Using the chat UI

Navigate to `http://localhost:5000/chat` in your browser. The UI supports multi-turn conversations, markdown rendering, code highlighting, thinking blocks, and tool-call display out of the box.

---

## `YALMRServer` API reference

### Model management

| Method | Description |
|---|---|
| `LoadModelAsync(modelId, options)` | Load a model under the given identifier. Returns `false` if already loaded. |
| `UnloadModelAsync(modelId)` | Unload a model. Throws if active sessions exist. |
| `IsModelLoaded(modelId)` | Returns `true` if the model is currently loaded. |
| `TryGetModelOptions(modelId, out options)` | Returns the `SessionOptions` used to load the model. |
| `TryGetModelCapabilities(modelId, out vision, out thinking)` | Returns the model's vision and thinking support. |
| `ModelIds` | Collection of identifiers of all loaded models. |

### Session management

| Method | Description |
|---|---|
| `CreateSession(modelId, options?)` | Create a session. Returns a generated session identifier. |
| `GetSession(sessionId)` | Returns the `Session` for the given identifier. Throws if not found. |
| `TryGetSession(sessionId, out session)` | Non-throwing session lookup. |
| `RemoveSessionAsync(sessionId)` | Dispose and remove a session. Returns `false` if not found. |
| `SessionIds` | Collection of identifiers of all active sessions. |
