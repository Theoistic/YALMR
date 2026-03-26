---
layout: page
title: Conversation Compaction
permalink: /features/conversation-compaction/
parent: Features
nav_order: 5
---

# Conversation Compaction

Every model has a fixed context window — the maximum number of tokens it can hold in memory at one time. As a conversation grows, older messages must eventually be discarded or summarised to keep the total token count within bounds.

YALMR handles this automatically through **conversation compaction**.

---

## Configuration

Pass a `ConversationCompactionOptions` when creating a session:

```csharp
await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/model.gguf",
    ToolRegistry     = new ToolRegistry(),
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
});
```

`MaxInputTokens` is the threshold at which compaction is triggered. When the conversation history would exceed this many tokens, YALMR compacts it before sending the next request.

---

## How the default compactor works

The built-in compactor retains:

1. The **system message** (if any) — always kept in full
2. The **most recent messages** — enough to fill the remaining token budget

Older messages are dropped. This is a simple but effective strategy for most chatbot and agent workloads.

---

## Custom compaction strategies

Implement `IConversationCompactor` to replace the default strategy:

```csharp
public sealed class SummarizingCompactor : IConversationCompactor
{
    public async Task<IReadOnlyList<ChatMessage>> CompactAsync(
        IReadOnlyList<ChatMessage> history,
        int maxTokens,
        CancellationToken ct)
    {
        // Summarise the oldest half of the conversation, then return
        // [system, summary-message, ...recent-messages]
        string summary = await SummariseAsync(history.Take(history.Count / 2));
        var recent     = history.Skip(history.Count / 2).ToList();

        var compacted = new List<ChatMessage>();
        if (history.FirstOrDefault(m => m.Role == "system") is { } sys)
            compacted.Add(sys);

        compacted.Add(new ChatMessage("assistant", $"[Earlier conversation summary: {summary}]"));
        compacted.AddRange(recent.Where(m => m.Role != "system"));
        return compacted;
    }
}
```

Register it on the session:

```csharp
new SessionOptions
{
    ...
    Compaction          = new ConversationCompactionOptions(MaxInputTokens: 8192),
    ConversationCompactor = new SummarizingCompactor(),
}
```

---

## KV-cache reset

When the context window is full, YALMR can reset the KV cache to free memory. `ResetContextTokens` controls how many tokens are reserved for the reset buffer:

```csharp
new SessionOptions
{
    ...
    ContextTokens      = 32768,
    ResetContextTokens = 2048,    // reserve 2048 tokens as reset headroom
}
```

---

## Tips

- Set `MaxInputTokens` to roughly 75–80 % of `ContextTokens` to leave room for the model's reply.
- For long-running agents, a summarising compactor preserves important context that would otherwise be lost.
- `KeepLatestImage` or `DropProcessedImages` (see [Vision](vision)) reduce the token cost of multimodal conversations and pair well with compaction.
