---
layout: page
title: Getting Started
permalink: /getting-started/
nav_order: 2
---

# Getting Started

This guide walks you through installing YALMR, obtaining a model, creating a session, and running your first chat.

---

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/10.0) or later
- A GGUF model file — download one from [Hugging Face](https://huggingface.co/models?library=gguf) (e.g. a quantised Llama or Qwen model)

---

## Installation

Add the NuGet package to your project:

```bash
dotnet add package YALMR
```

Or via the Package Manager Console:

```powershell
Install-Package YALMR
```

---

## Step 1 — Install the llama.cpp runtime

YALMR downloads the correct llama.cpp native binaries automatically the first time you call `EnsureInstalledAsync`. Binaries are cached on disk and reused on subsequent runs.

```csharp
using YALMR.LlamaCpp;

// CPU backend (always available)
string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

// NVIDIA GPU (requires a CUDA-capable GPU and matching drivers)
// string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cuda);

// AMD/Intel GPU via Vulkan
// string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Vulkan);
```

**Cache locations:**

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\YALMR\llama-runtime` |
| Linux / macOS | `~/.local/share/YALMR/llama-runtime` |

---

## Step 2 — Create a session

A `Session` combines a loaded model engine with conversation history and inference settings. Use `Session.CreateAsync` to load a model and get a ready-to-use session in one call.

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

### Key `SessionOptions` properties

| Property | Default | Description |
|---|---|---|
| `ModelPath` | *(required)* | Path to the GGUF model file |
| `ToolRegistry` | *(required)* | Registry of callable tools (use `new ToolRegistry()` if you have none) |
| `Compaction` | *(required)* | Controls how the context window is managed when it fills up |
| `ContextTokens` | `8192` | Total KV-cache size in tokens |
| `GpuLayers` | `0` | Number of model layers to offload to GPU (`-1` = all) |
| `FlashAttention` | `false` | Enable Flash Attention (recommended for GPU inference) |
| `DefaultInference` | `null` | Default sampling settings (temperature, top-p, etc.) |
| `MmprojPath` | `null` | Path to a multimodal projector model for vision support |

---

## Step 3 — Your first chat

```csharp
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Hello!")))
    Console.Write(chunk.Text);
```

Multi-turn conversation is tracked automatically. Each call to `GenerateAsync` appends both the user message and the assistant reply to the session history.

### Sending a follow-up

```csharp
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "What did I just say?")))
    Console.Write(chunk.Text);
// The model sees the full conversation history and can refer back to it.
```

### Getting a single string response

`SendAsync` is a convenience wrapper that collects all streamed chunks and returns the full reply as a `ChatMessage`:

```csharp
ChatMessage reply = await session.SendAsync(new ChatMessage("user", "Summarise the French Revolution."));
Console.WriteLine(reply.Content);
```

---

## Step 4 — Setting a system prompt

Pass the system message before the first user message:

```csharp
session.History.Add(new ChatMessage("system", "You are a helpful assistant specialised in history."));

await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Who was Napoleon?")))
    Console.Write(chunk.Text);
```

---

## Next Steps

- **[Streaming](features/streaming)** — handle text, reasoning, and tool-call chunks individually
- **[Tool calling](features/tool-calling)** — let the model invoke your C# methods
- **[Structured output](features/structured-output)** — get typed .NET objects back from the model
- **[Vision](features/vision)** — send images alongside text messages
- **[Conversation compaction](features/conversation-compaction)** — keep long chats within the context window
- **[MCP integration](features/mcp-integration)** — connect to external tool servers
- **[Multi-model server](features/multi-model-server)** — serve multiple models over HTTP
