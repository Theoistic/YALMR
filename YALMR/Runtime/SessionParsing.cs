using System.Text;
using YALMR.LlamaCpp;

namespace YALMR.Runtime;

internal sealed record ParsedAssistantOutput(
    string Content,
    string ReasoningContent,
    bool HasReasoning);

internal static class SessionParsing
{
    public static ParsedAssistantOutput ParseAssistantOutput(string rawText, bool enableThinking)
    {
        const string thinkStart = "<think>";
        const string thinkEnd = "</think>";

        int start = rawText.IndexOf(thinkStart, StringComparison.Ordinal);
        if (start >= 0)
        {
            int reasoningStart = start + thinkStart.Length;
            int end = rawText.IndexOf(thinkEnd, reasoningStart, StringComparison.Ordinal);

            if (end >= 0)
            {
                string content = rawText[..start] + rawText[(end + thinkEnd.Length)..];
                string reasoning = rawText[reasoningStart..end];
                return new ParsedAssistantOutput(content, reasoning, true);
            }

            return new ParsedAssistantOutput(rawText[..start], rawText[reasoningStart..], true);
        }

        if (enableThinking)
        {
            int end = rawText.IndexOf(thinkEnd, StringComparison.Ordinal);
            if (end >= 0)
            {
                string reasoning = rawText[..end];
                string content = rawText[(end + thinkEnd.Length)..];
                return new ParsedAssistantOutput(content, reasoning, true);
            }

            return new ParsedAssistantOutput(string.Empty, rawText, rawText.Length > 0);
        }

        return new ParsedAssistantOutput(rawText, string.Empty, false);
    }

    public static ChatMessage CreateAssistantMessage(string rawText, bool enableThinking, List<ToolCall>? toolCalls = null, InferenceUsage? usage = null)
    {
        var parsedOutput = ParseAssistantOutput(rawText, enableThinking);
        string content = toolCalls is { Count: > 0 }
            ? MiniJinjaChatTemplate.StripToolCallMarkup(parsedOutput.Content)
            : parsedOutput.Content;

        return new ChatMessage(
            "assistant",
            string.IsNullOrWhiteSpace(content) ? null : content.Trim(),
            ReasoningContent: parsedOutput.HasReasoning ? parsedOutput.ReasoningContent.Trim() : null,
            ToolCalls: toolCalls,
            RawContent: rawText,
            Usage: usage);
    }

    public static string? GetDelta(string current, ref int emittedLength)
    {
        if (current.Length <= emittedLength)
            return null;

        string delta = current[emittedLength..];
        emittedLength = current.Length;
        return delta.Length == 0 ? null : delta;
    }

    public static string GetStreamingVisibleContent(string content)
    {
        string visible = RemoveStreamingDelimitedBlock(content, "<|tool_call_start|>", "<|tool_call_end|>");
        visible = RemoveStreamingDelimitedBlock(visible, "<tool_call>", "</tool_call>");
        visible = RemoveStreamingDelimitedBlock(visible, "<tool_code>", "</tool_code>");
        return TrimTrailingTagPrefix(visible, "<|tool_call_start|>", "<|tool_call_end|>", "<tool_call>", "</tool_call>", "<tool_code>", "</tool_code>");
    }

    public static IReadOnlyList<ChatMessage> ApplyImageRetentionPolicy(IReadOnlyList<ChatMessage> messages, ImageRetentionPolicy policy)
    {
        if (policy == ImageRetentionPolicy.KeepAllImages)
            return messages;

        int remainingImages = 1;
        var result = new List<ChatMessage>(messages.Count);

        for (int i = messages.Count - 1; i >= 0; i--)
        {
            var message = messages[i];
            if (message.Parts is not { Count: > 0 })
            {
                result.Insert(0, message);
                continue;
            }

            var parts = new List<ContentPart>(message.Parts.Count);
            for (int j = message.Parts.Count - 1; j >= 0; j--)
            {
                ContentPart part = message.Parts[j];
                if (part is ImagePart)
                {
                    if (remainingImages > 0)
                    {
                        parts.Insert(0, part);
                        remainingImages--;
                    }

                    continue;
                }

                parts.Insert(0, part);
            }

            result.Insert(0, parts.Count == message.Parts.Count
                ? message
                : message with { Parts = parts.Count > 0 ? parts : null });
        }

        return result;
    }

    public static List<string> ExtractImageBase64s(IReadOnlyList<ChatMessage> messages)
    {
        var images = new List<string>();

        for (int i = 0; i < messages.Count; i++)
        {
            if (messages[i].Parts is not { Count: > 0 } parts) continue;

            for (int j = 0; j < parts.Count; j++)
            {
                if (parts[j] is ImagePart { Base64: { } base64 })
                    images.Add(base64);
            }
        }

        return images;
    }

    public static bool HasRenderableUserQuery(IReadOnlyList<ChatMessage> messages)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            var msg = messages[i];
            if (msg.Role == "user")
            {
                string content = msg.Content ?? string.Empty;
                return !(content.StartsWith("<tool_response>") && content.EndsWith("</tool_response>"));
            }
        }
        return false;
    }

    private static string RemoveStreamingDelimitedBlock(string text, string startTag, string endTag)
    {
        int searchStart = 0;
        var sb = new StringBuilder(text.Length);

        while (searchStart < text.Length)
        {
            int start = text.IndexOf(startTag, searchStart, StringComparison.Ordinal);
            if (start < 0)
            {
                sb.Append(text, searchStart, text.Length - searchStart);
                break;
            }

            sb.Append(text, searchStart, start - searchStart);

            int end = text.IndexOf(endTag, start + startTag.Length, StringComparison.Ordinal);
            if (end < 0)
                break;

            searchStart = end + endTag.Length;
        }

        return sb.ToString();
    }

    private static string TrimTrailingTagPrefix(string text, params string[] tags)
    {
        int trimLength = 0;

        foreach (string tag in tags)
        {
            int maxPrefixLength = Math.Min(tag.Length - 1, text.Length);
            for (int length = maxPrefixLength; length > trimLength; length--)
            {
                if (tag.AsSpan(0, length).SequenceEqual(text.AsSpan(text.Length - length, length)))
                {
                    trimLength = length;
                    break;
                }
            }
        }

        return trimLength == 0 ? text : text[..^trimLength];
    }
}
