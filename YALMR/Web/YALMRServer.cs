using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using YALMR.Runtime;

namespace YALMR.Web;

/// <summary>
/// Central orchestrator for the YALMR framework.
/// Models are loaded once into a shared <see cref="Engine"/> and then any number of
/// independent <see cref="Session"/> instances can be created on top of that engine.
/// </summary>
public sealed class YALMRServer : IAsyncDisposable, IDisposable
{
    private readonly record struct SessionEntry(Session Session, string ModelId);

    private readonly ConcurrentDictionary<string, (Engine Engine, SessionOptions Options)> _models
        = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, SessionEntry> _sessions
        = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim _modelLock = new(1, 1);
    private bool _disposed;

    // -------------------------------------------------------------------------
    // Model lifecycle
    // -------------------------------------------------------------------------

    /// <summary>
    /// Identifiers of all currently loaded models.
    /// </summary>
    public IReadOnlyCollection<string> ModelIds => [.. _models.Keys];

    /// <summary>
    /// Loads a model under <paramref name="modelId"/> and keeps its engine resident for session use.
    /// Returns <c>false</c> when a model with that identifier is already loaded.
    /// </summary>
    public async Task<bool> LoadModelAsync(string modelId, SessionOptions options, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        ArgumentNullException.ThrowIfNull(options);
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_models.ContainsKey(modelId))
            return false;

        await _modelLock.WaitAsync(ct);
        try
        {
            if (_models.ContainsKey(modelId))
                return false;

            var engine = await Engine.CreateAsync(options, ct);

            // Reconcile EnableThinking with what the model's chat template actually supports.
            // If the caller explicitly disabled thinking (false), that choice is honoured.
            // Otherwise we fall back to the engine's own detection so a reasoning model
            // always enables thinking by default when no opinion was expressed.
            var userDefaultInference = options.DefaultInference ?? new InferenceOptions();
            var effectiveOptions = options with
            {
                DefaultInference = userDefaultInference with
                {
                    EnableThinking = userDefaultInference.EnableThinking == false
                        ? false
                        : engine.ThinkingEnabled
                }
            };

            _models[modelId] = (engine, effectiveOptions);
            return true;
        }
        finally
        {
            _modelLock.Release();
        }
    }

    /// <summary>
    /// Unloads the model identified by <paramref name="modelId"/> and disposes its engine.
    /// Returns <c>false</c> when no such model is loaded.
    /// Throws <see cref="InvalidOperationException"/> when one or more sessions are still using the model.
    /// </summary>
    public async Task<bool> UnloadModelAsync(string modelId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        ObjectDisposedException.ThrowIf(_disposed, this);

        await _modelLock.WaitAsync();
        try
        {
            if (!_models.TryGetValue(modelId, out var loaded))
                return false;

            var activeSessions = _sessions.Values
                .Where(e => string.Equals(e.ModelId, modelId, StringComparison.OrdinalIgnoreCase))
                .ToArray();

            if (activeSessions.Length > 0)
                throw new InvalidOperationException(
                    $"Cannot unload model '{modelId}': {activeSessions.Length} active session(s) are still using it. " +
                    $"Remove all dependent sessions before unloading.");

            _models.TryRemove(modelId, out _);
            await loaded.Engine.DisposeAsync();
            return true;
        }
        finally
        {
            _modelLock.Release();
        }
    }

    /// <summary>
    /// Returns <c>true</c> when a model with the given identifier is currently loaded.
    /// </summary>
    public bool IsModelLoaded(string modelId) => _models.ContainsKey(modelId);

    /// <summary>
    /// Returns <c>true</c> and populates <paramref name="options"/> when a model with
    /// the given identifier is loaded; otherwise returns <c>false</c>.
    /// </summary>
    public bool TryGetModelOptions(string modelId, [NotNullWhen(true)] out SessionOptions? options)
    {
        if (_models.TryGetValue(modelId, out var loaded))
        {
            options = loaded.Options;
            return true;
        }

        options = null;
        return false;
    }

    /// <summary>
    /// Returns whether the loaded model supports vision and/or thinking (chain-of-thought).
    /// </summary>
    public bool TryGetModelCapabilities(string modelId, out bool visionEnabled, out bool thinkingEnabled)
    {
        if (_models.TryGetValue(modelId, out var loaded))
        {
            visionEnabled   = loaded.Engine.VisionEnabled;
            thinkingEnabled = loaded.Engine.ThinkingEnabled;
            return true;
        }

        visionEnabled   = false;
        thinkingEnabled = false;
        return false;
    }

    // -------------------------------------------------------------------------
    // Session lifecycle
    // -------------------------------------------------------------------------

    /// <summary>
    /// Identifiers of all currently active sessions.
    /// </summary>
    public IReadOnlyCollection<string> SessionIds => [.. _sessions.Keys];

    /// <summary>
    /// Creates a new session against the loaded model identified by <paramref name="modelId"/>
    /// and returns its generated session identifier.
    /// <para>
    /// When <paramref name="sessionOptions"/> is provided it is used as the session configuration,
    /// allowing per-session tool registries, compaction policies, and generation defaults while
    /// re-using the already-resident engine.
    /// When omitted, the options used at model load time are reused.
    /// </para>
    /// </summary>
    /// <exception cref="InvalidOperationException">The requested model is not loaded.</exception>
    public string CreateSession(string modelId, SessionOptions? sessionOptions = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_models.TryGetValue(modelId, out var loaded))
            throw new InvalidOperationException($"Model '{modelId}' is not loaded.");

        var session = Session.Create(sessionOptions ?? loaded.Options, loaded.Engine);
        string sessionId = $"sess_{Guid.NewGuid():N}";
        _sessions[sessionId] = new SessionEntry(session, modelId);
        return sessionId;
    }

    /// <summary>
    /// Returns the session for the given <paramref name="sessionId"/>.
    /// </summary>
    /// <exception cref="KeyNotFoundException">No session with that identifier exists.</exception>
    public Session GetSession(string sessionId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);

        if (!_sessions.TryGetValue(sessionId, out var entry))
            throw new KeyNotFoundException($"Session '{sessionId}' does not exist.");

        return entry.Session;
    }

    /// <summary>
    /// Tries to retrieve a session by identifier without throwing.
    /// </summary>
    public bool TryGetSession(string sessionId, [NotNullWhen(true)] out Session? session)
    {
        if (_sessions.TryGetValue(sessionId, out var entry))
        {
            session = entry.Session;
            return true;
        }

        session = null;
        return false;
    }

    /// <summary>
    /// Removes and disposes the session identified by <paramref name="sessionId"/>.
    /// Returns <c>false</c> when no such session exists.
    /// </summary>
    public async Task<bool> RemoveSessionAsync(string sessionId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);

        if (!_sessions.TryRemove(sessionId, out var entry))
            return false;

        await entry.Session.DisposeAsync();
        return true;
    }

    // -------------------------------------------------------------------------
    // Disposal
    // -------------------------------------------------------------------------

    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        foreach (var entry in _sessions.Values)
            await entry.Session.DisposeAsync();

        _sessions.Clear();

        foreach (var (engine, _) in _models.Values)
            await engine.DisposeAsync();

        _models.Clear();
        _modelLock.Dispose();
    }

    /// <inheritdoc/>
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();
}
