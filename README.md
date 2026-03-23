# YALMR

[![NuGet](https://img.shields.io/nuget/v/YALMR.svg)](https://www.nuget.org/packages/YALMR)

**Y**et **A**nother L**LM** **R**untime � run local GGUF models in .NET 10 via llama.cpp.

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

## Features

- **Tool calling** � register handlers in `ToolRegistry`; the model calls them automatically
- **Streaming** � `IAsyncEnumerable<GenerationChunk>` with text, reasoning, and tool-call chunks
- **Vision** � attach images via `ImagePart` with a multimodal projector model
- **Conversation compaction** � automatic context-window management with pluggable strategies
- **MCP integration** � call tools from HTTP or stdio MCP servers
- **Multi-model server** � `YALMRServer` manages named engines and sessions for concurrent use
