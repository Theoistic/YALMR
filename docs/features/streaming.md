---
layout: page
title: Streaming
permalink: /features/streaming/
parent: Features
nav_order: 1
---

# Streaming

YALMR streams model output as it is generated, giving you token-by-token access to text, reasoning (chain-of-thought), and tool-call chunks through a standard `IAsyncEnumerable<ChatResponseChunk>`.

---

## Basic streaming

```csharp
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Tell me a joke.")))
    Console.Write(chunk.Text);
```

Each `ChatResponseChunk` arrives as soon as the model produces it. The loop completes when the model finishes generating.

---

## Chunk types

`ChatResponseChunk` carries different kinds of content depending on the model and inference settings:

| Property | Type | Description |
|---|---|---|
| `Text` | `string` | The generated text fragment (most common) |
| `ReasoningText` | `string?` | Chain-of-thought / thinking text emitted by reasoning models |
| `ToolCall` | `ToolCall?` | A complete tool-call request emitted by the model |
| `Usage` | `InferenceUsage?` | Token counts, present in the final chunk when available |

### Handling each chunk type

```csharp
await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "What is 2 + 2?")))
{
    if (chunk.ReasoningText is { Length: > 0 } reasoning)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.Write(reasoning);
        Console.ResetColor();
    }

    if (chunk.Text is { Length: > 0 } text)
        Console.Write(text);

    if (chunk.ToolCall is { } toolCall)
        Console.WriteLine($"\n[tool call: {toolCall.Name}]");
}
```

---

## Inference options

Pass an `InferenceOptions` instance to `GenerateAsync` to override the session defaults for a single request:

```csharp
var options = new InferenceOptions
{
    Temperature    = 0.7f,
    TopP           = 0.9f,
    MaxOutputTokens = 256,
    EnableThinking = false,    // suppress chain-of-thought output
};

await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Write a haiku."), options))
    Console.Write(chunk.Text);
```

### Sampling parameters

| Property | Default | Description |
|---|---|---|
| `Temperature` | `0.0` | Sampling temperature; higher = more random |
| `TopP` | `1.0` | Nucleus sampling probability mass |
| `TopK` | `0` | Top-K sampling; `0` = disabled |
| `PresencePenalty` | `0.0` | Penalty for repeating topic tokens |
| `FrequencyPenalty` | `0.0` | Penalty proportional to token frequency |
| `RepetitionPenalty` | `1.0` | Multiplicative penalty for recently seen tokens |
| `MaxOutputTokens` | `512` | Maximum tokens to generate |
| `Seed` | `null` | Fixed RNG seed for reproducible output |

---

## Collecting the full reply

If you need the complete response as a single string rather than a stream, use the `SendAsync` helper:

```csharp
ChatMessage reply = await session.SendAsync(new ChatMessage("user", "Summarise the French Revolution."));
Console.WriteLine(reply.Content);
```

`SendAsync` buffers the stream internally and returns the final `ChatMessage` with `Content`, `ReasoningContent`, `ToolCalls`, and `Usage` all populated.

---

## Cancellation

Pass a `CancellationToken` to stop generation early:

```csharp
using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));

await foreach (var chunk in session.GenerateAsync(new ChatMessage("user", "Tell me everything about space."), cancellationToken: cts.Token))
    Console.Write(chunk.Text);
```
