---
layout: page
title: MCP Integration
permalink: /features/mcp-integration/
parent: Features
nav_order: 6
---

# MCP Integration

YALMR can delegate tool calls to external servers that implement the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). This lets you connect local or remote services — databases, file systems, code execution sandboxes, REST APIs — without writing adapter code inside your .NET application.

---

## Supported transports

| Transport | Description |
|---|---|
| **stdio** | Launch a local process and communicate over stdin/stdout using JSON-RPC framing |
| **HTTP** | Connect to a remote MCP-compatible endpoint |

---

## stdio (local process)

### 1 — Describe the remote tool

Register an `AgentTool` that points to an MCP server process. Pass a `McpServerProcessOptions` as the `Metadata` field of the `AgentToolRemoteDefinition`:

```csharp
using YALMR.Mcp;
using YALMR.Runtime;

var registry = new ToolRegistry();

registry.Register(new AgentTool(
    "read_file",
    "Reads the contents of a file from the local filesystem.",
    [
        new ToolParameter("path", "string", "Absolute or relative path to the file."),
    ],
    new AgentToolRemoteDefinition(
        Server:    "filesystem-mcp",
        ToolName:  "read_file",
        Transport: "mcp",
        Metadata:  new McpServerProcessOptions(
            Command:   "npx",
            Arguments: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]))));
```

### 2 — Attach an MCP executor to the session

```csharp
await using var mcpExecutor = new McpRemoteToolExecutor();

await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory    = backendDir,
    ModelPath           = "path/to/model.gguf",
    ToolRegistry        = registry,
    ToolExecutionEngine = mcpExecutor,
    DefaultInference    = new InferenceOptions { Tools = registry.ToToolDefinitions() },
    Compaction          = new ConversationCompactionOptions(MaxInputTokens: 8192),
});
```

When the model calls `read_file`, YALMR launches the MCP process (if not already running), sends the JSON-RPC `tools/call` request, and returns the result to the model — all automatically.

---

## `McpServerProcessOptions` reference

| Property | Default | Description |
|---|---|---|
| `Command` | *(required)* | Executable to launch (e.g. `node`, `npx`, `python`) |
| `Arguments` | `null` | Command-line arguments |
| `WorkingDirectory` | `null` | Working directory for the process |
| `EnvironmentVariables` | `null` | Extra environment variables |
| `ProtocolVersion` | `"2024-11-05"` | MCP protocol version to advertise |
| `ClientName` | `"YALMR"` | Client name sent in the `initialize` handshake |
| `ClientVersion` | `"1.0.0"` | Client version sent in the `initialize` handshake |

---

## Multiple MCP servers

You can register tools from multiple different servers in the same `ToolRegistry`. Each server is identified by the `Server` string in `AgentToolRemoteDefinition`, and `McpRemoteToolExecutor` keeps one persistent process per unique server label:

```csharp
registry.Register(new AgentTool(
    "query_database",
    "Runs a SQL query against the production database.",
    [...],
    new AgentToolRemoteDefinition(
        Server:   "db-mcp",
        ToolName: "query",
        Metadata: new McpServerProcessOptions("python", ["-m", "db_mcp_server"]))));

registry.Register(new AgentTool(
    "run_code",
    "Executes Python code in a sandbox.",
    [...],
    new AgentToolRemoteDefinition(
        Server:   "sandbox-mcp",
        ToolName: "execute",
        Metadata: new McpServerProcessOptions("docker", ["run", "--rm", "sandbox-image"]))));
```

---

## Mixing local and remote tools

Local (`AgentTool` with a handler) and remote (MCP) tools can coexist in the same registry:

```csharp
// local tool
registry.Register(new AgentTool("get_time", "Returns the current time.", [], _ => DateTimeOffset.Now.ToString("R")));

// remote tool
registry.Register(new AgentTool("search_web", "Searches the web.", [...], new AgentToolRemoteDefinition("search-mcp", "search")));
```

YALMR checks whether a tool has a local handler first; if not, it delegates to the registered `IRemoteToolExecutor`.

---

## Process lifecycle

`McpRemoteToolExecutor` starts each MCP server process on first use and keeps it alive for the lifetime of the executor. When the executor is disposed, all child processes are terminated:

```csharp
await using var mcpExecutor = new McpRemoteToolExecutor();
// ... use session ...
// processes are cleaned up automatically on dispose
```
