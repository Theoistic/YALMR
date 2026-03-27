using System.Collections;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace YALMR.Runtime;

/// <summary>
/// A first-class conversation object that wraps a list of <see cref="ChatMessage"/> instances
/// and provides serialization for persistence.
/// For training/finetuning export, see <see cref="TrainingExporter"/> and <see cref="ITrainingExporter"/>.
/// </summary>
public sealed class Conversation : IList<ChatMessage>, IReadOnlyList<ChatMessage>
{
    private readonly List<ChatMessage> _messages;

    /// <summary>Unique identifier for this conversation.</summary>
    public string Id { get; set; }

    /// <summary>Optional model name associated with this conversation.</summary>
    public string? ModelName { get; set; }

    /// <summary>UTC timestamp when the conversation was created.</summary>
    public DateTimeOffset CreatedAt { get; set; }

    /// <summary>Arbitrary key-value metadata for custom tagging (e.g. dataset, quality label).</summary>
    public Dictionary<string, string> Metadata { get; set; } = [];

    public Conversation()
    {
        _messages = [];
        Id = $"conv_{Guid.NewGuid():N}";
        CreatedAt = DateTimeOffset.UtcNow;
    }

    public Conversation(IEnumerable<ChatMessage> messages) : this()
    {
        _messages = [.. messages];
    }

    public Conversation(string id, IEnumerable<ChatMessage> messages)
    {
        Id = id;
        _messages = [.. messages];
        CreatedAt = DateTimeOffset.UtcNow;
    }

    // ── IList<ChatMessage> / IReadOnlyList<ChatMessage> ─────────────

    public ChatMessage this[int index]
    {
        get => _messages[index];
        set => _messages[index] = value;
    }

    public int Count => _messages.Count;
    public bool IsReadOnly => false;

    public void Add(ChatMessage item) => _messages.Add(item);
    public void AddRange(IEnumerable<ChatMessage> items) => _messages.AddRange(items);
    public void Clear() => _messages.Clear();
    public bool Contains(ChatMessage item) => _messages.Contains(item);
    public void CopyTo(ChatMessage[] array, int arrayIndex) => _messages.CopyTo(array, arrayIndex);
    public int IndexOf(ChatMessage item) => _messages.IndexOf(item);
    public void Insert(int index, ChatMessage item) => _messages.Insert(index, item);
    public bool Remove(ChatMessage item) => _messages.Remove(item);
    public void RemoveAt(int index) => _messages.RemoveAt(index);

    public IEnumerator<ChatMessage> GetEnumerator() => _messages.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    /// Creates a shallow copy of this conversation with the same metadata and messages.
    /// </summary>
    public Conversation Snapshot()
    {
        var copy = new Conversation([.. _messages])
        {
            Id = Id,
            ModelName = ModelName,
            CreatedAt = CreatedAt,
            Metadata = new Dictionary<string, string>(Metadata, StringComparer.Ordinal)
        };
        return copy;
    }

    // ── Persistence ─────────────────────────────────────────────────

    private static readonly JsonSerializerOptions s_jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true,
    };

    /// <summary>
    /// Saves the conversation to a file as a JSON document.
    /// </summary>
    public async Task SaveAsync(string path, CancellationToken ct = default)
    {
        await using var stream = File.Create(path);
        await SaveAsync(stream, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Saves the conversation to a stream as a JSON document.
    /// </summary>
    public async Task SaveAsync(Stream stream, CancellationToken ct = default)
    {
        var envelope = ToEnvelope();
        await JsonSerializer.SerializeAsync(stream, envelope, s_jsonOptions, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Loads a conversation from a JSON file.
    /// </summary>
    public static async Task<Conversation> LoadAsync(string path, CancellationToken ct = default)
    {
        await using var stream = File.OpenRead(path);
        return await LoadAsync(stream, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Loads a conversation from a JSON stream.
    /// </summary>
    public static async Task<Conversation> LoadAsync(Stream stream, CancellationToken ct = default)
    {
        var envelope = await JsonSerializer.DeserializeAsync<ConversationEnvelope>(stream, s_jsonOptions, ct).ConfigureAwait(false)
            ?? throw new InvalidOperationException("Failed to deserialize conversation.");
        return FromEnvelope(envelope);
    }

    /// <summary>
    /// Serializes the conversation to a JSON string.
    /// </summary>
    public string ToJson() => JsonSerializer.Serialize(ToEnvelope(), s_jsonOptions);

    /// <summary>
    /// Deserializes a conversation from a JSON string.
    /// </summary>
    public static Conversation FromJson(string json)
    {
        var envelope = JsonSerializer.Deserialize<ConversationEnvelope>(json, s_jsonOptions)
            ?? throw new InvalidOperationException("Failed to deserialize conversation.");
        return FromEnvelope(envelope);
    }

    // ── Persistence models ──────────────────────────────────────────

    private sealed record ConversationEnvelope
    {
        public string? Id { get; init; }
        public string? ModelName { get; init; }
        public DateTimeOffset? CreatedAt { get; init; }
        public Dictionary<string, string>? Metadata { get; init; }
        public List<ChatMessage>? Messages { get; init; }
    }

    private ConversationEnvelope ToEnvelope() => new()
    {
        Id = Id,
        ModelName = ModelName,
        CreatedAt = CreatedAt,
        Metadata = Metadata.Count > 0 ? Metadata : null,
        Messages = [.. _messages]
    };

    private static Conversation FromEnvelope(ConversationEnvelope envelope)
    {
        return new Conversation(envelope.Messages ?? [])
        {
            Id = envelope.Id ?? $"conv_{Guid.NewGuid():N}",
            ModelName = envelope.ModelName,
            CreatedAt = envelope.CreatedAt ?? DateTimeOffset.UtcNow,
            Metadata = envelope.Metadata ?? []
        };
    }
}
