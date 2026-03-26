---
layout: page
title: Getting Started
permalink: /getting-started/
nav_order: 2
---

# Getting Started
{: .no_toc }

This guide walks you through every step needed to go from a blank .NET project to a streaming local LLM conversation with YALMR.
{: .fs-5 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Before you begin, make sure you have the following installed:

| Requirement | Notes |
|---|---|
| [.NET 10 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/10.0) | .NET 10 or later is required |
| A GGUF model file | Download from [Hugging Face](https://huggingface.co/models?library=gguf). A 4-bit quantised model (Q4_K_M) is a good starting point. |
| *(Optional)* CUDA 12+ drivers | Only needed for `LlamaBackend.Cuda` (NVIDIA GPU inference) |
| *(Optional)* Vulkan drivers | Only needed for `LlamaBackend.Vulkan` (AMD / Intel GPU inference) |

### Recommended starter models

If you are unsure which model to download, these are good starting points:

- **[Qwen3-0.6B-Q8_0.gguf](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)** — tiny, fast, good for testing (0.6 GB)
- **[Qwen3-1.7B-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)** — small capable model (1 GB)
- **[Llama-3.2-3B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)** — solid general-purpose model (2 GB)

---

## Installation

Create a new console application and add the YALMR NuGet package:

```bash
dotnet new console -n MyLLMApp
cd MyLLMApp
dotnet add package YALMR
```

Or, if you prefer the Package Manager Console in Visual Studio:

```powershell
Install-Package YALMR
```

---

## Step 1 — Install the native runtime

YALMR uses llama.cpp as its inference backend. The required native binaries are downloaded automatically from GitHub Releases the first time you call `EnsureInstalledAsync`, then cached on disk for subsequent runs.

```csharp
using YALMR.LlamaCpp;

string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);
```

### Choosing a backend

| Backend constant | Hardware | When to use |
|---|---|---|
| `LlamaBackend.Cpu` | Any x86-64 / ARM64 CPU | Always works; no extra setup. Slower on large models. |
| `LlamaBackend.Cuda` | NVIDIA GPU (CUDA 12+) | Best performance on NVIDIA hardware. Requires matching drivers. |
| `LlamaBackend.Vulkan` | AMD / Intel GPU | Cross-platform GPU acceleration. Good fallback for non-NVIDIA cards. |

```csharp
// NVIDIA GPU
string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cuda);

// AMD / Intel GPU
string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Vulkan);
```

### Cache locations

Downloaded binaries are stored at:

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\YALMR\llama-runtime` |
| Linux / macOS | `~/.local/share/YALMR/llama-runtime` |

To force a re-download, delete the cache directory and call `EnsureInstalledAsync` again.

---

## Step 2 — Create a session

A `Session` wraps a loaded model engine with conversation history and inference settings. Use `Session.CreateAsync` to load a model and get a fully initialised session in one call.

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

`Session` implements `IAsyncDisposable` — always wrap it in `await using` to ensure the native model is unloaded cleanly.

### SessionOptions reference

| Property | Default | Description |
|---|---|---|
| `ModelPath` | *(required)* | Absolute or relative path to the `.gguf` model file |
| `BackendDirectory` | *(required)* | Directory returned by `EnsureInstalledAsync` |
| `ToolRegistry` | *(required)* | Registry of callable tools — use `new ToolRegistry()` if you have none |
| `Compaction` | *(required)* | Controls how the context window is managed when full |
| `ContextTokens` | `8192` | Total KV-cache size in tokens |
| `GpuLayers` | `0` | Layers to offload to GPU (`-1` = all layers) |
| `FlashAttention` | `false` | Enable Flash Attention (recommended for GPU inference) |
| `DefaultInference` | `null` | Default sampling settings applied to every request |
| `MmprojPath` | `null` | Path to a multimodal projector model for vision support |
| `MaxToolRounds` | `10` | Maximum consecutive tool-call cycles per generation |

### GPU inference configuration

To maximise GPU utilisation, set `GpuLayers` to `-1` (all layers on GPU) and enable `FlashAttention`:

```csharp
await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cuda),
    ModelPath        = "path/to/model.gguf",
    ToolRegistry     = new ToolRegistry(),
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
    GpuLayers        = -1,
    FlashAttention   = true,
    ContextTokens    = 32768,
});
```

---

## Step 3 — Your first conversation

### Streaming output

```csharp
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Hello!")))
    Console.Write(chunk.Text);
```

Each `ChatResponseChunk` arrives as soon as the model produces a token. The loop completes when the model stops generating. Conversation history is tracked automatically — every user message and assistant reply is appended to `session.History`.

### Collecting the full reply

If you don't need streaming, use `SendAsync` to receive the complete response in one call:

```csharp
ChatMessage reply = await session.SendAsync(new ChatMessage("user", "Summarise the French Revolution."));
Console.WriteLine(reply.Content);
```

### Multi-turn conversation

YALMR tracks conversation history automatically. Just keep calling `GenerateAsync` or `SendAsync` — the model always sees the full history:

```csharp
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "My name is Alice.")))
    Console.Write(chunk.Text);

Console.WriteLine();

await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "What is my name?")))
    Console.Write(chunk.Text);
// The model will reply "Your name is Alice."
```

---

## Step 4 — Setting a system prompt

Add a system message to `session.History` before the first user turn:

```csharp
session.History.Add(new ChatMessage("system", "You are a concise assistant. Always reply in bullet points."));

await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "What are the planets in our solar system?")))
    Console.Write(chunk.Text);
```

---

## Step 5 — Tuning inference parameters

Override sampling settings for a single request without changing session defaults:

```csharp
var options = new InferenceOptions
{
    Temperature     = 0.8f,   // 0.0 = deterministic, higher = more creative
    TopP            = 0.95f,  // nucleus sampling
    MaxOutputTokens = 512,    // cap response length
};

await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Tell me a story."), options))
    Console.Write(chunk.Text);
```

To apply settings to every request, set `DefaultInference` in `SessionOptions`:

```csharp
new SessionOptions
{
    ...
    DefaultInference = new InferenceOptions
    {
        Temperature     = 0.7f,
        MaxOutputTokens = 1024,
    },
}
```

---

## Step 6 — Enabling tool calling

YALMR can call your C# methods automatically when the model decides it needs information. Decorate methods with `[Tool]` and register them before creating the session:

```csharp
using YALMR.Tools;

public class MyTools
{
    [Tool("Returns the current UTC date and time.")]
    public string GetCurrentTime() => DateTimeOffset.UtcNow.ToString("R");

    [Tool("Returns a simple weather report for a city.")]
    public string GetWeather(
        [ToolParam("The city name.")] string city)
        => $"It is 22 °C and sunny in {city}.";
}

// Register tools
var registry = new ToolRegistry();
registry.Register(new MyTools());

await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/model.gguf",
    ToolRegistry     = registry,
    DefaultInference = new InferenceOptions { Tools = registry.ToToolDefinitions() },
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
});

// The model will call GetCurrentTime() or GetWeather() automatically
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "What is the weather in London right now?")))
    Console.Write(chunk.Text);
```

See the **[Tool Calling](features/tool-calling)** page for the full API including `AgentTool`, DI integration, and per-session tool registries.

---

## Step 7 — Getting structured output

Use `AskAsync<T>()` to receive a deserialised .NET object instead of raw text. YALMR constrains the model with a GBNF grammar derived from the type's JSON schema:

```csharp
public record CityInfo(string Name, string Country, int PopulationMillions);

CityInfo info = await session.AskAsync<CityInfo>(
    new ChatMessage("user", "Give me data about Tokyo."));

Console.WriteLine($"{info.Name}, {info.Country} — {info.PopulationMillions}M people");
```

See **[Structured Output](features/structured-output)** for more examples.

---

## Cancellation

Pass a `CancellationToken` to stop generation early:

```csharp
using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(15));

await foreach (var chunk in session.GenerateAsync(
    new ChatMessage("user", "Write me a very long essay."),
    cancellationToken: cts.Token))
{
    Console.Write(chunk.Text);
}
```

---

## Troubleshooting

### Model fails to load

- Verify the path in `ModelPath` is correct and the file is a valid GGUF.
- Ensure you are using the backend that matches your hardware (`Cpu` always works).
- Large models may exceed available RAM. Try a smaller quantisation (e.g. Q4_K_M instead of Q8_0).

### Out-of-memory with GPU backend

- Reduce `ContextTokens` (e.g. `4096` instead of `32768`).
- Set `GpuLayers` to a lower value instead of `-1` to keep some layers on CPU.

### Runtime download fails

- Check your internet connection and firewall settings.
- The installer fetches from GitHub Releases (`github.com/ggerganov/llama.cpp`).
- If offline, copy the binaries manually to the cache path shown in **Step 1**.

### Generation is very slow on CPU

- Use a smaller or more aggressively quantised model (Q4_K_M or Q3_K_M).
- Increase `GpuLayers` if you have a compatible GPU.
- The first token is always slower because the prompt must be fully processed first.

---

## Next Steps

- **[Streaming](features/streaming)** — handle text, reasoning, and tool-call chunks individually
- **[Tool Calling](features/tool-calling)** — let the model invoke your C# methods
- **[Structured Output](features/structured-output)** — get typed .NET objects back from the model
- **[Vision](features/vision)** — send images alongside text messages
- **[Conversation Compaction](features/conversation-compaction)** — keep long chats within the context window
- **[MCP Integration](features/mcp-integration)** — connect to external tool servers
- **[Multi-Model Server](features/multi-model-server)** — serve multiple models over HTTP

