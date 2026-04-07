using System.Text.Json;

namespace YALMR.Runtime;

/// <summary>
/// Exports conversations in the OpenAI messages format:
/// <c>{"messages": [{"role": "...", "content": "..."}]}</c>.
/// Compatible with OpenAI, Unsloth, TRL, and most finetuning tools.
/// </summary>
public sealed class OpenAITrainingExporter : ITrainingExporter
{
    public static readonly OpenAITrainingExporter Instance = new();

    public string Export(Conversation conversation, TrainingExportOptions options)
    {
        var output = new List<Dictionary<string, object?>>();
        int imgIdx = 0;

        foreach (var msg in conversation)
        {
            if (!TrainingExporter.ShouldIncludeMessage(msg, options)) continue;

            var entry = new Dictionary<string, object?> { ["role"] = msg.Role };

            // Tool result messages
            if (msg.Role == "tool")
            {
                entry["tool_call_id"] = msg.ToolCallId;
                entry["content"] = msg.Content ?? "";
                output.Add(entry);
                continue;
            }

            // Assistant messages with tool calls
            if (msg.Role == "assistant" && msg.ToolCalls is { Count: > 0 } toolCalls)
            {
                string? text = TrainingExporter.GetTextContent(msg, options);

                switch (options.ToolCallMode)
                {
                    case ToolCallExportMode.Structured:
                        entry["content"] = text;
                        entry["tool_calls"] = toolCalls.Select(tc => new Dictionary<string, object?>
                        {
                            ["id"] = tc.CallId,
                            ["type"] = "function",
                            ["function"] = new Dictionary<string, object?>
                            {
                                ["name"] = tc.Name,
                                ["arguments"] = JsonSerializer.Serialize(tc.Arguments, TrainingExporter.JsonOptions)
                            }
                        }).ToList();
                        break;

                    case ToolCallExportMode.Inline:
                        string callText = TrainingExporter.FormatToolCallsAsText(toolCalls);
                        entry["content"] = string.IsNullOrWhiteSpace(text) ? callText : $"{text}\n{callText}";
                        break;

                    default:
                        entry["content"] = text ?? "";
                        break;
                }
            }
            else
            {
                // Regular messages — may contain images
                entry["content"] = BuildContent(msg, conversation.Id, options, ref imgIdx);
            }

            // Reasoning as a separate field (only meaningful for custom tooling)
            if (msg.Role == "assistant"
                && options.ReasoningMode == ReasoningExportMode.SeparateField
                && msg.ReasoningContent is { Length: > 0 } reasoning)
            {
                entry["reasoning"] = reasoning;
            }

            output.Add(entry);
        }

        return JsonSerializer.Serialize(
            new Dictionary<string, object?> { ["messages"] = output }, TrainingExporter.JsonOptions);
    }

    /// <summary>
    /// Builds the <c>content</c> value for an OpenAI-format message.
    /// Returns a <see cref="string"/> for text-only messages or a
    /// <see cref="List{T}"/> of content-part dictionaries for multimodal messages.
    /// </summary>
    private static object BuildContent(
        ChatMessage msg, string convId, TrainingExportOptions options, ref int imgIdx)
    {
        bool hasImages = options.ImageMode != ImageExportMode.Omit
                         && msg.Parts?.OfType<ImagePart>().Any(ip => ip.Base64 is { Length: > 0 }) == true;

        if (!hasImages)
            return TrainingExporter.GetTextContent(msg, options) ?? "";

        // Build multimodal content-parts array
        var parts = new List<Dictionary<string, object?>>();
        bool addedText = false;

        foreach (var part in msg.Parts!)
        {
            switch (part)
            {
                case TextPart tp:
                    string text = tp.Text;
                    if (!addedText && msg.Role == "assistant"
                        && msg.ReasoningContent is { Length: > 0 } r
                        && options.ReasoningMode == ReasoningExportMode.Inline)
                    {
                        text = $"<think>{r}</think>{text}";
                    }
                    parts.Add(new Dictionary<string, object?> { ["type"] = "text", ["text"] = text });
                    addedText = true;
                    break;

                case ImagePart { Base64: { Length: > 0 } b64 }:
                    switch (options.ImageMode)
                    {
                        case ImageExportMode.InlineBase64:
                            parts.Add(new Dictionary<string, object?>
                            {
                                ["type"] = "image_url",
                                ["image_url"] = new Dictionary<string, object?>
                                {
                                    ["url"] = $"data:image/png;base64,{b64}"
                                }
                            });
                            break;

                        case ImageExportMode.FileReference:
                            string imgFile = $"{options.ImageOutputDirectory}/{convId}_img{imgIdx++}.png";
                            parts.Add(new Dictionary<string, object?>
                            {
                                ["type"] = "image_url",
                                ["image_url"] = new Dictionary<string, object?> { ["url"] = imgFile }
                            });
                            break;

                        case ImageExportMode.Placeholder:
                            parts.Add(new Dictionary<string, object?>
                            {
                                ["type"] = "text", ["text"] = "[image]"
                            });
                            break;
                    }
                    break;
            }
        }

        // If Parts had no TextParts, fall back to Content
        if (!addedText && msg.Content is { Length: > 0 } content)
        {
            if (msg.Role == "assistant"
                && msg.ReasoningContent is { Length: > 0 } reasoning
                && options.ReasoningMode == ReasoningExportMode.Inline)
            {
                content = $"<think>{reasoning}</think>{content}";
            }
            parts.Insert(0, new Dictionary<string, object?> { ["type"] = "text", ["text"] = content });
        }

        return parts;
    }
}
