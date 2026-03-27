---
layout: page
title: Training Data Export
permalink: /features/training-export/
parent: Features
nav_order: 8
---

# Training Data Export
{: .no_toc }

YALMR can serialize any `Conversation` into training/finetuning JSONL for the most popular frameworks — OpenAI, ShareGPT, ChatML, and Alpaca — via a small interface-driven exporter system that is easy to extend with custom formats.

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Quick start

```csharp
using YALMR.Runtime;

var conversation = new Conversation
{
    new ChatMessage("system", "You are a helpful assistant."),
    new ChatMessage("user",  "What is the capital of France?"),
    new ChatMessage("assistant", "The capital of France is Paris."),
};

// Export a single conversation as a JSONL string (OpenAI format by default)
string jsonl = TrainingExporter.Export(conversation);

// Write directly to a .jsonl file
await TrainingExporter.ExportAsync(conversation, "dataset.jsonl");
```

---

## Supported formats

Pass a `TrainingExportOptions` to choose the output format.

| `TrainingFormat` | Schema | Compatible with |
|---|---|---|
| `OpenAI` *(default)* | `{"messages": [{"role": "...", "content": "..."}]}` | OpenAI fine-tuning, Unsloth, TRL |
| `ShareGPT` | `{"conversations": [{"from": "human", "value": "..."}]}` | Axolotl, LLaMA-Factory, FastChat |
| `ChatML` | `{"text": "<\|im_start\|>role\ncontent<\|im_end\|>"}` | Custom ChatML pipelines |
| `Alpaca` | `{"instruction": "...", "input": "", "output": "..."}` | Stanford Alpaca, single-turn finetuning |

```csharp
var options = new TrainingExportOptions
{
    Format = TrainingFormat.ShareGPT,
};

string jsonl = TrainingExporter.Export(conversation, options);
```

### Alpaca note

Alpaca is a single-turn format. Multi-turn conversations are automatically split into individual user/assistant pairs, each emitted as its own JSONL line. The system prompt (if present) is prepended to every instruction field.

---

## Batch export

Export a collection of conversations into a single `.jsonl` file — one record per line (multiple lines for Alpaca):

```csharp
IEnumerable<Conversation> dataset = LoadConversations();

await TrainingExporter.ExportAsync(dataset, "dataset.jsonl", new TrainingExportOptions
{
    Format = TrainingFormat.OpenAI,
});
```

---

## Reasoning content

Models that produce `<think>` blocks (DeepSeek R1, Qwen3) populate `ChatMessage.ReasoningContent`. By default this is stripped. Use `ReasoningMode` to keep it:

```csharp
// Inline — wraps reasoning in <think>...</think> before the response text
var options = new TrainingExportOptions
{
    ReasoningMode = ReasoningExportMode.Inline,
};

// SeparateField — emits a "reasoning" key alongside "content" (OpenAI format only)
var options = new TrainingExportOptions
{
    Format        = TrainingFormat.OpenAI,
    ReasoningMode = ReasoningExportMode.SeparateField,
};
```

---

## Tool calls

By default tool-call rounds are stripped so the exported dataset contains only clean text exchanges. Enable them when finetuning a tool-calling model:

```csharp
// Structured — emits tool_calls arrays and tool role messages (OpenAI finetuning spec)
var options = new TrainingExportOptions
{
    Format       = TrainingFormat.OpenAI,
    ToolCallMode = ToolCallExportMode.Structured,
};

// Inline — serialises calls as <tool_call> / <tool_response> tags in the text
var options = new TrainingExportOptions
{
    ToolCallMode = ToolCallExportMode.Inline,
};
```

---

## Images

Multimodal conversations can export their images in three ways:

```csharp
// Embed as base-64 data URLs (OpenAI vision content-parts format)
var options = new TrainingExportOptions
{
    ImageMode = ImageExportMode.InlineBase64,
};

// Save images to files and reference them by relative path
var options = new TrainingExportOptions
{
    ImageMode            = ImageExportMode.FileReference,
    ImageOutputDirectory = "images",   // relative to the output .jsonl file
};
await TrainingExporter.ExportAsync(conversation, "dataset.jsonl", options);
// → saves  dataset_dir/images/conv_<id>_img0.png, …

// Replace each image with a [image] placeholder token
var options = new TrainingExportOptions
{
    ImageMode = ImageExportMode.Placeholder,
};
```

---

## Filtering messages

Use `MessageFilter` to exclude specific messages, or toggle system messages entirely:

```csharp
var options = new TrainingExportOptions
{
    // Drop all system messages
    IncludeSystemMessages = false,

    // Custom predicate — skip empty assistant turns
    MessageFilter = msg =>
        !(msg.Role == "assistant" && string.IsNullOrWhiteSpace(msg.Content)),
};
```

---

## Custom exporters

Implement `ITrainingExporter` to add a format not covered by the built-ins:

```csharp
public sealed class LLamaFactoryExporter : ITrainingExporter
{
    public string Export(Conversation conversation, TrainingExportOptions options)
    {
        // Build your custom JSONL structure here
        var record = new
        {
            id           = conversation.Id,
            conversations = conversation
                .Where(m => TrainingExporter.ShouldIncludeMessage(m, options))
                .Select(m => new { role = m.Role, content = TrainingExporter.GetTextContent(m, options) })
                .ToList()
        };
        return JsonSerializer.Serialize(record);
    }
}
```

Pass it directly to any `Export` / `ExportAsync` overload:

```csharp
var exporter = new LLamaFactoryExporter();

// Single conversation
string jsonl = TrainingExporter.Export(conversation, exporter, options);

// Batch
string jsonl = TrainingExporter.Export(conversations, exporter, options);
```

---

## Saving and loading conversations

`Conversation` has built-in persistence separate from training export. Use `SaveAsync` / `LoadAsync` to round-trip a conversation as a structured JSON file:

```csharp
// Save
await conversation.SaveAsync("chat_2025_01_01.json");

// Load
Conversation loaded = await Conversation.LoadAsync("chat_2025_01_01.json");
```

Or serialize to/from a JSON string in-memory:

```csharp
string json = conversation.ToJson();
Conversation restored = Conversation.FromJson(json);
```

This format is YALMR-specific (preserves all fields including `ReasoningContent`, `ToolCalls`, `Parts`) and is intended for persistence, not for passing to training frameworks. Use `TrainingExporter` when you need a framework-compatible format.

---

## Building a dataset from a live session

Capture every conversation a session produces and append it to a growing dataset file:

```csharp
var allConversations = new List<Conversation>();

// After each chat, snapshot the session's conversation
Conversation snapshot = session.Conversation.Snapshot();
allConversations.Add(snapshot);

// Periodically flush to disk
await TrainingExporter.ExportAsync(allConversations, "dataset.jsonl", new TrainingExportOptions
{
    Format                = TrainingFormat.OpenAI,
    IncludeSystemMessages = true,
});
```
