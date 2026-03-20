namespace YALMR.Mcp;

/// <summary>
/// Launch settings for an MCP server process reachable over stdio.
/// </summary>
public sealed record McpServerProcessOptions(
    string Command,
    IReadOnlyList<string>? Arguments = null,
    string? WorkingDirectory = null,
    IReadOnlyDictionary<string, string>? EnvironmentVariables = null,
    string ProtocolVersion = "2024-11-05",
    string ClientName = "YALMR",
    string ClientVersion = "1.0.0"
);
