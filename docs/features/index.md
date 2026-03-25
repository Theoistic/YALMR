---
layout: page
title: Features
permalink: /features/
nav_order: 3
has_children: true
---

# Features

YALMR provides a comprehensive set of features for running local LLMs in .NET:

| Feature | Description |
|---|---|
| [Streaming](streaming) | Token-by-token `IAsyncEnumerable` output with text, reasoning, and tool-call chunks |
| [Tool Calling](tool-calling) | Attribute-based and manual tool registration; automatic execution during generation |
| [Structured Output](structured-output) | Grammar-constrained sampling with automatic JSON deserialization into .NET types |
| [Vision](vision) | Multimodal image+text messages with configurable image retention |
| [Conversation Compaction](conversation-compaction) | Automatic context-window management with pluggable strategies |
| [MCP Integration](mcp-integration) | Delegate tool calls to stdio or HTTP MCP servers |
| [Multi-Model Server](multi-model-server) | Host multiple models and expose an OpenAI-compatible HTTP API |
