using System.Diagnostics.CodeAnalysis;

namespace YALMR.Runtime;

/// <summary>
/// Describes a single named tool parameter.
/// </summary>
public sealed record ToolParameter(
    string Name,
    string Type,
    string Description,
    bool Required = true
);

/// <summary>
/// Describes how a tool is fulfilled by a remote MCP-capable endpoint rather than a local handler.
/// </summary>
public sealed record AgentToolRemoteDefinition(
    string Server,
    string ToolName,
    string Transport = "mcp",
    object? Metadata = null
);

/// <summary>
/// Represents a callable tool that can be exposed to the model.
/// </summary>
public sealed class AgentTool
{
    public string Name { get; }
    public string Description { get; }
    public IReadOnlyList<ToolParameter> Parameters { get; }
    public AgentToolRemoteDefinition? RemoteDefinition { get; }
    public bool IsRemote => RemoteDefinition is not null;
    public bool CanExecuteLocally => _handler is not null;

    private readonly Func<IReadOnlyDictionary<string, object?>, string>? _handler;

    public AgentTool(
        string name,
        string description,
        IReadOnlyList<ToolParameter> parameters,
        Func<IReadOnlyDictionary<string, object?>, string>? handler,
        AgentToolRemoteDefinition? remoteDefinition = null)
    {
        Name = name;
        Description = description;
        Parameters = parameters;
        _handler = handler;
        RemoteDefinition = remoteDefinition;
    }

    /// <summary>
    /// Creates a tool that is declared locally but intended to be fulfilled by a remote MCP endpoint.
    /// </summary>
    public AgentTool(
        string name,
        string description,
        IReadOnlyList<ToolParameter> parameters,
        AgentToolRemoteDefinition remoteDefinition)
        : this(name, description, parameters, handler: null, remoteDefinition)
    {
    }

    /// <summary>
    /// Executes the tool synchronously and converts handler failures into error text.
    /// </summary>
    public string Execute(IReadOnlyDictionary<string, object?> args)
    {
        if (_handler is null)
        {
            return RemoteDefinition is not null
                ? $"Error: tool '{Name}' is declared as a remote {RemoteDefinition.Transport} tool on '{RemoteDefinition.Server}' and cannot be executed by the local session."
                : $"Error: tool '{Name}' does not have a local handler.";
        }

        try
        {
            return _handler(args);
        }
        catch (Exception ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Executes the tool on a background task.
    /// </summary>
    public Task<string> ExecuteAsync(IReadOnlyDictionary<string, object?> args, CancellationToken ct = default) =>
        Task.Run(() => Execute(args), ct);

    /// <summary>
    /// Converts the tool definition into the template schema shape expected by the model.
    /// </summary>
    public object ToTemplateTool()
    {
        var properties = new Dictionary<string, object>();
        foreach (var p in Parameters)
            properties[p.Name] = new { type = p.Type, description = p.Description };

        var required = Parameters.Where(p => p.Required).Select(p => p.Name).ToArray();

        var schema = new
        {
            name = Name,
            description = Description,
            parameters = new
            {
                type = "object",
                properties,
                required
            }
        };

        return schema;
    }
}

/// <summary>
/// Stores registered tools and exposes lookup and template conversion helpers.
/// </summary>
public sealed class ToolRegistry : IEnumerable<AgentTool>
{
    private readonly Dictionary<string, AgentTool> _tools = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the number of registered tools.
    /// </summary>
    public int Count => _tools.Count;

    /// <summary>
    /// Adds a tool for collection-initializer scenarios.
    /// </summary>
    public void Add(AgentTool tool) => Register(tool);

    /// <summary>
    /// Registers or replaces a tool by name.
    /// </summary>
    public void Register(AgentTool tool) => _tools[tool.Name] = tool;

    /// <summary>
    /// Removes a tool by name.
    /// </summary>
    public bool Remove(string name) => _tools.Remove(name);

    /// <summary>
    /// Removes all registered tools.
    /// </summary>
    public void Clear() => _tools.Clear();

    /// <summary>
    /// Attempts to resolve a tool by name.
    /// </summary>
    public bool TryGet(string name, [NotNullWhen(true)] out AgentTool? tool) =>
        _tools.TryGetValue(name, out tool);

    /// <summary>
    /// Converts all registered tools into template tool definitions.
    /// </summary>
    public IReadOnlyList<object?> ToTemplateTools() =>
        [.. _tools.Values.Select(t => t.ToTemplateTool())];

    /// <summary>
    /// Enumerates registered tools.
    /// </summary>
    public IEnumerator<AgentTool> GetEnumerator() => _tools.Values.GetEnumerator();

    /// <summary>
    /// Enumerates registered tools through the non-generic interface.
    /// </summary>
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
}
