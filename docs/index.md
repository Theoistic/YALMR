---
layout: home
title: Home
nav_order: 1
---

# YALMR
{: .fs-9 }

**Y**et **A**nother **L**LM **R**untime — run local GGUF models in .NET 10 via [llama.cpp](https://github.com/ggerganov/llama.cpp). No Python. No remote APIs. Just fast, private inference in your own process.
{: .fs-5 .fw-300 }

[![NuGet](https://img.shields.io/nuget/v/YALMR.svg)](https://www.nuget.org/packages/YALMR)
[![NuGet Downloads](https://img.shields.io/nuget/dt/YALMR.svg)](https://www.nuget.org/packages/YALMR)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Theoistic/YALMR/blob/master/LICENSE)

[Get Started](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/Theoistic/YALMR){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What is YALMR?

YALMR is a lightweight, high-performance .NET library that lets you run large language models **locally** using GGUF-format weights. It wraps llama.cpp with a clean async API and adds production-ready features — all without needing a Python runtime, a GPU cloud subscription, or a remote API key.

- **Private by design** — model weights and conversation data never leave your machine.
- **Zero-config runtimes** — native llama.cpp binaries are downloaded and cached automatically on first use.
- **Async-first** — every inference API is built on `IAsyncEnumerable` and `Task`/`ValueTask`.
- **Production ready** — tool calling, structured output, vision, context management, and a multi-model HTTP server included out of the box.

---

## Features at a Glance

| Feature | Description |
|---|---|
| [Streaming](features/streaming) | Token-by-token `IAsyncEnumerable<ChatResponseChunk>` with text, reasoning, and tool-call chunks |
| [Tool Calling](features/tool-calling) | Register C# methods with `[Tool]` attributes or the fluent `AgentTool` API; the model calls them automatically |
| [Structured Output](features/structured-output) | `AskAsync<T>()` constrains sampling with GBNF grammar and deserializes the result directly into your .NET type |
| [Vision](features/vision) | Attach images via `ImagePart` with a multimodal projector model (LLaVA-style) |
| [Conversation Compaction](features/conversation-compaction) | Automatic context-window management with pluggable summarisation or truncation strategies |
| [MCP Integration](features/mcp-integration) | Delegate tool calls to stdio or HTTP Model Context Protocol servers |
| [Multi-Model Server](features/multi-model-server) | `YALMRServer` hosts multiple named engines and exposes an OpenAI-compatible HTTP API |

---

## Quick Start

Add the package:

```bash
dotnet add package YALMR
```

Run your first chat in under 10 lines:

```csharp
using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;

// Download and cache the native llama.cpp binaries (once)
string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

// Load a model and open a session
await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/model.gguf",
    ToolRegistry     = new ToolRegistry(),
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
});

// Stream the response token by token
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Hello!")))
    Console.Write(chunk.Text);
```

> **Need a model?** Download any GGUF file from [Hugging Face](https://huggingface.co/models?library=gguf). A 4-bit quantised Qwen or Llama model works great on CPU.

---

## Supported Backends

| Backend | Hardware | Notes |
|---|---|---|
| `LlamaBackend.Cpu` | Any x86-64 / ARM64 | Always available; no extra drivers needed |
| `LlamaBackend.Cuda` | NVIDIA GPU | Requires CUDA-capable GPU and matching drivers |
| `LlamaBackend.Vulkan` | AMD / Intel GPU | Cross-platform GPU acceleration via Vulkan |

---

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/10.0) or later
- A GGUF model file (download from [Hugging Face](https://huggingface.co/models?library=gguf))
- For GPU inference: appropriate drivers (CUDA 12+ or Vulkan)

---

## License

MIT — see [LICENSE](https://github.com/Theoistic/YALMR/blob/master/LICENSE).

