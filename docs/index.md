---
layout: home
title: YALMR
nav_order: 1
---

# YALMR

[![NuGet](https://img.shields.io/nuget/v/YALMR.svg)](https://www.nuget.org/packages/YALMR)

**Y**et **A**nother **L**LM **R**untime — run local GGUF models in .NET 10 via [llama.cpp](https://github.com/ggerganov/llama.cpp).

> Requires .NET 10 SDK and a GGUF model file (e.g. from [Hugging Face](https://huggingface.co/models?library=gguf)).

---

## What is YALMR?

YALMR is a lightweight, high-performance .NET library that lets you run large language models locally using GGUF-format weights. It wraps llama.cpp with a clean async API and adds production-ready features like tool calling, structured output, vision, conversation compaction, and a multi-model HTTP server — all without needing a Python runtime or a remote API.

---

## Features at a Glance

| Feature | Description |
|---|---|
| [Streaming](features/streaming) | `IAsyncEnumerable<ChatResponseChunk>` for text, reasoning, and tool-call chunks |
| [Tool calling](features/tool-calling) | Register handlers with `[Tool]` attributes or the `AgentTool` API; the model calls them automatically |
| [Structured output](features/structured-output) | `AskAsync<T>()` constrains sampling with GBNF grammar and deserializes the result |
| [Vision](features/vision) | Attach images via `ImagePart` with a multimodal projector model |
| [Conversation compaction](features/conversation-compaction) | Automatic context-window management with pluggable strategies |
| [MCP integration](features/mcp-integration) | Call tools from HTTP or stdio MCP servers |
| [Multi-model server](features/multi-model-server) | `YALMRServer` manages named engines and sessions for concurrent use |

---

## Quick Start

```bash
dotnet add package YALMR
```

```csharp
using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;

// 1. Install the native llama.cpp binaries (downloaded once, cached on disk)
string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

// 2. Open a session
await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/model.gguf",
    ToolRegistry     = new ToolRegistry(),
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
});

// 3. Chat
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Hello!")))
    Console.Write(chunk.Text);
```

See the **[Getting Started](getting-started)** guide for a full walkthrough.

---

## License

MIT
