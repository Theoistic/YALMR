using System.Diagnostics;
using YALMR.Diagnostics;

namespace YALMR.Runtime;

/// <summary>
/// Describes a tool execution request resolved from a model-emitted tool call.
/// </summary>
public sealed record ToolExecutionRequest(
    ToolCall Call,
    AgentTool Tool,
    ToolRegistry Registry
);

/// <summary>
/// Executes a resolved tool call, regardless of whether the tool is local or remote.
/// </summary>
public interface IToolExecutionEngine
{
    /// <summary>
    /// Executes the tool represented by the request.
    /// </summary>
    Task<string> ExecuteAsync(ToolExecutionRequest request, CancellationToken ct = default);
}

/// <summary>
/// Handles execution of remote tools declared through <see cref="AgentToolRemoteDefinition"/>.
/// </summary>
public interface IRemoteToolExecutor
{
    /// <summary>
    /// Returns <c>true</c> when this executor can handle the given remote tool definition.
    /// </summary>
    bool CanExecute(AgentToolRemoteDefinition definition);

    /// <summary>
    /// Executes a remote tool call.
    /// </summary>
    Task<string> ExecuteAsync(AgentToolRemoteDefinition definition, ToolCall call, CancellationToken ct = default);
}

/// <summary>
/// Default tool execution engine that supports local tools and pluggable remote executors.
/// </summary>
public sealed class DefaultToolExecutionEngine : IToolExecutionEngine
{
    private readonly IReadOnlyList<IRemoteToolExecutor> _remoteExecutors;

    public DefaultToolExecutionEngine(IEnumerable<IRemoteToolExecutor>? remoteExecutors = null)
    {
        _remoteExecutors = remoteExecutors?.ToArray() ?? [];
    }

    /// <summary>
    /// Executes the resolved tool call using either the local handler or a matching remote executor.
    /// </summary>
    public async Task<string> ExecuteAsync(ToolExecutionRequest request, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        using var activity = RuntimeTelemetry.StartActivity("yalmr.tool.execute");
        RuntimeTelemetry.ToolExecutions.Add(1, new KeyValuePair<string, object?>("yalmr.tool.name", request.Call.Name));
        var sw = Stopwatch.StartNew();
        try
        {
            activity?.SetTag("yalmr.tool.name", request.Call.Name);
            activity?.SetTag("yalmr.tool.is_remote", request.Tool.IsRemote);

            if (request.Tool.CanExecuteLocally)
                return await request.Tool.ExecuteAsync(request.Call.Arguments, ct);

            if (!request.Tool.IsRemote || request.Tool.RemoteDefinition is null)
                return $"Error: tool '{request.Call.Name}' does not have a local handler or a remote definition.";

            var executor = _remoteExecutors.FirstOrDefault(candidate => candidate.CanExecute(request.Tool.RemoteDefinition));
            if (executor is null)
            {
                return $"Error: no remote executor is registered for {request.Tool.RemoteDefinition.Transport} tool '{request.Tool.RemoteDefinition.ToolName}' on '{request.Tool.RemoteDefinition.Server}'.";
            }

            return await executor.ExecuteAsync(request.Tool.RemoteDefinition, request.Call, ct);
        }
        catch (Exception ex)
        {
            RuntimeTelemetry.ToolExecutionErrors.Add(1, new KeyValuePair<string, object?>("yalmr.tool.name", request.Call.Name));
            RuntimeTelemetry.RecordException(activity, ex);
            throw;
        }
        finally
        {
            RuntimeTelemetry.ToolExecutionDuration.Record(sw.Elapsed.TotalMilliseconds,
                new KeyValuePair<string, object?>("yalmr.tool.name", request.Call.Name));
        }
    }
}