using YALMR.Runtime;

namespace YALMR.Utils;

public enum ConversationCompactionLevel { Light, Balanced, Aggressive }

public enum ContextCompactionStrategy
{
    FifoSlidingWindow,
    PinnedSystemFifo,
    MiddleOutElision,
    RollingSummarization,
    HeuristicPruning,
    VectorAugmentedRecall
}

public sealed record ConversationCompactionOptions(
    int MaxInputTokens,
    int ReservedForGeneration = 512,
    ContextCompactionStrategy Strategy = ContextCompactionStrategy.PinnedSystemFifo,
    ConversationCompactionLevel Level = ConversationCompactionLevel.Balanced,
    bool AlwaysKeepSystem = true,
    int HotTrailMessages = 4
)
{
    public int TokenBudget => MaxInputTokens - ReservedForGeneration;
}

/// <summary>
/// Provides the dependencies needed by a conversation compactor.
/// </summary>
public sealed record ConversationCompactionContext(
    ConversationCompactionOptions Options,
    Func<IReadOnlyList<ChatMessage>, int> CountTokens,
    Func<IReadOnlyList<ChatMessage>, bool> HasRenderableUserQuery
);

/// <summary>
/// Reduces conversation history to fit within the current prompt budget.
/// </summary>
public interface IConversationCompactor
{
    /// <summary>
    /// Returns the compacted conversation history to render into the next prompt.
    /// </summary>
    IReadOnlyList<ChatMessage> Compact(IReadOnlyList<ChatMessage> messages, ConversationCompactionContext context);
}

/// <summary>
/// Default compactor that applies token-window strategies based on the configured options.
/// </summary>
public sealed class TokenWindowConversationCompactor : IConversationCompactor
{
    private sealed record ResolvedCompactionOptions(
        ConversationCompactionOptions Options,
        int MinimumRetainedMessages,
        bool IncludeReasoning,
        bool IncludeToolCalls,
        bool IncludeToolResults);

    /// <summary>
    /// Compacts conversation history according to the selected strategy and level.
    /// </summary>
    public IReadOnlyList<ChatMessage> Compact(IReadOnlyList<ChatMessage> messages, ConversationCompactionContext context)
    {
        if (messages.Count == 0)
            return messages;

        var options = Resolve(context.Options);

        return options.Options.Strategy switch
        {
            ContextCompactionStrategy.FifoSlidingWindow => CompactFifo(messages, context, options, pinSystem: false, heuristicPruning: false),
            ContextCompactionStrategy.PinnedSystemFifo => CompactFifo(messages, context, options, pinSystem: options.Options.AlwaysKeepSystem, heuristicPruning: false),
            ContextCompactionStrategy.MiddleOutElision => CompactMiddleOut(messages, context, options),
            ContextCompactionStrategy.HeuristicPruning => CompactFifo(messages, context, options, pinSystem: options.Options.AlwaysKeepSystem, heuristicPruning: true),
            ContextCompactionStrategy.RollingSummarization => throw new InvalidOperationException("RollingSummarization requires a custom IConversationCompactor implementation."),
            ContextCompactionStrategy.VectorAugmentedRecall => throw new InvalidOperationException("VectorAugmentedRecall requires a custom IConversationCompactor implementation."),
            _ => CompactFifo(messages, context, options, pinSystem: options.Options.AlwaysKeepSystem, heuristicPruning: false)
        };
    }

    private static IReadOnlyList<ChatMessage> CompactFifo(
        IReadOnlyList<ChatMessage> messages,
        ConversationCompactionContext context,
        ResolvedCompactionOptions options,
        bool pinSystem,
        bool heuristicPruning)
    {
        int budget = options.Options.TokenBudget;
        ChatMessage? system = pinSystem && options.Options.AlwaysKeepSystem && messages[0].Role == "system" ? messages[0] : null;
        int start = system is null ? 0 : 1;
        int hotTrailStart = Math.Max(start, messages.Count - options.Options.HotTrailMessages);
        var history = heuristicPruning
            ? CompactHistory(messages, start, hotTrailStart, options)
            : [.. messages.Skip(start)];
        var tail = new List<ChatMessage>();

        for (int i = history.Count - 1; i >= 0; i--)
        {
            tail.Insert(0, history[i]);
            var candidate = new List<ChatMessage>();
            if (system is not null)
                candidate.Add(system);
            candidate.AddRange(tail);

            if (!context.HasRenderableUserQuery(candidate))
                continue;

            if (context.CountTokens(candidate) > budget)
            {
                tail.RemoveAt(0);

                // If removing the message eliminated the only user query, put it
                // back.  A prompt without a user message is unusable — the chat
                // template will throw (e.g. raise_exception).
                if (!HasRenderableUserQueryInTail(context, system, tail))
                    tail.Insert(0, history[i]);

                break;
            }
        }

        int minimumMessages = heuristicPruning ? options.MinimumRetainedMessages : Math.Max(1, options.Options.HotTrailMessages);
        while (tail.Count < minimumMessages && history.Count > tail.Count)
        {
            int index = history.Count - tail.Count - 1;
            if (index < 0)
                break;

            tail.Insert(0, history[index]);
        }

        var result = new List<ChatMessage>();
        if (system is not null)
            result.Add(system);
        result.AddRange(tail);
        return result;
    }

    private static IReadOnlyList<ChatMessage> CompactMiddleOut(
        IReadOnlyList<ChatMessage> messages,
        ConversationCompactionContext context,
        ResolvedCompactionOptions options)
    {
        int budget = options.Options.TokenBudget;

        ChatMessage? system = options.Options.AlwaysKeepSystem && messages[0].Role == "system" ? messages[0] : null;
        int start = system is null ? 0 : 1;
        int hotTrailStart = Math.Max(start, messages.Count - options.Options.HotTrailMessages);
        var history = CompactHistory(messages, start, hotTrailStart, options);
        var tail = new List<ChatMessage>();

        for (int i = history.Count - 1; i >= 0; i--)
        {
            tail.Insert(0, history[i]);
            var candidate = new List<ChatMessage>();
            if (system is not null)
                candidate.Add(system);
            candidate.AddRange(tail);

            if (!context.HasRenderableUserQuery(candidate))
                continue;

            if (context.CountTokens(candidate) > budget)
            {
                tail.RemoveAt(0);

                if (!HasRenderableUserQueryInTail(context, system, tail))
                    tail.Insert(0, history[i]);

                break;
            }
        }

        int minimumMessages = options.MinimumRetainedMessages;
        while (tail.Count < minimumMessages && history.Count > tail.Count)
        {
            int index = history.Count - tail.Count - 1;
            if (index < 0)
                break;

            tail.Insert(0, history[index]);
        }

        var result = new List<ChatMessage>();
        if (system is not null)
            result.Add(system);
        result.AddRange(tail);
        return result;
    }

    private static ResolvedCompactionOptions Resolve(ConversationCompactionOptions options)
    {
        int hotTrail = options.Level switch
        {
            ConversationCompactionLevel.Light => Math.Max(options.HotTrailMessages, 6),
            ConversationCompactionLevel.Aggressive => Math.Min(Math.Max(options.HotTrailMessages, 1), 2),
            _ => Math.Max(options.HotTrailMessages, 4)
        };

        int minimumRetainedMessages = options.Level switch
        {
            ConversationCompactionLevel.Light => Math.Max(hotTrail, 6),
            ConversationCompactionLevel.Aggressive => Math.Max(hotTrail, 2),
            _ => Math.Max(hotTrail, 4)
        };

        bool includeReasoning = options.Level == ConversationCompactionLevel.Light;
        bool includeToolCalls = options.Strategy != ContextCompactionStrategy.HeuristicPruning || options.Level != ConversationCompactionLevel.Aggressive;
        bool includeToolResults = options.Strategy switch
        {
            ContextCompactionStrategy.HeuristicPruning when options.Level != ConversationCompactionLevel.Light => false,
            _ => true
        };

        return new ResolvedCompactionOptions(
            options with { HotTrailMessages = hotTrail },
            minimumRetainedMessages,
            includeReasoning,
            includeToolCalls,
            includeToolResults);
    }

    private static List<ChatMessage> CompactHistory(IReadOnlyList<ChatMessage> messages, int start, int hotTrailStart, ResolvedCompactionOptions options)
    {
        var history = new List<ChatMessage>(messages.Count - start);

        for (int i = start; i < messages.Count; i++)
        {
            ChatMessage? message = i >= hotTrailStart ? messages[i] : CompactMessage(messages[i], options);
            if (message is not null)
                history.Add(message);
        }

        return history;
    }

    private static bool HasRenderableUserQueryInTail(
        ConversationCompactionContext context,
        ChatMessage? system,
        List<ChatMessage> tail)
    {
        var candidate = new List<ChatMessage>(tail.Count + 1);
        if (system is not null)
            candidate.Add(system);
        candidate.AddRange(tail);
        return context.HasRenderableUserQuery(candidate);
    }

    private static ChatMessage? CompactMessage(ChatMessage message, ResolvedCompactionOptions options)
    {
        if (message.Role == "tool" && !options.IncludeToolResults)
            return null;

        var compacted = message with
        {
            ReasoningContent = options.IncludeReasoning ? message.ReasoningContent : null,
            ToolCalls = options.IncludeToolCalls ? message.ToolCalls : null
        };

        if (compacted.Role == "assistant"
            && string.IsNullOrWhiteSpace(compacted.Content)
            && string.IsNullOrWhiteSpace(compacted.ReasoningContent)
            && compacted.ToolCalls is not { Count: > 0 })
            return null;

        return compacted;
    }
}
