using System.Text;
using System.Text.Json;

namespace YALMR.Runtime;

/// <summary>
/// Exports conversations in the ChatML format:
/// <c>{"text": "&lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;"}</c>.
/// Used for raw ChatML custom pipelines.
/// </summary>
public sealed class ChatMLTrainingExporter : ITrainingExporter
{
    public static readonly ChatMLTrainingExporter Instance = new();

    public string Export(Conversation conversation, TrainingExportOptions options)
    {
        var sb = new StringBuilder();

        foreach (var msg in conversation)
        {
            if (!TrainingExporter.ShouldIncludeMessage(msg, options)) continue;

            string? content = TrainingExporter.GetTextContent(msg, options);

            if (msg.Role == "assistant" && msg.ToolCalls is { Count: > 0 }
                && options.ToolCallMode == ToolCallExportMode.Inline)
            {
                string callText = TrainingExporter.FormatToolCallsAsText(msg.ToolCalls);
                content = string.IsNullOrWhiteSpace(content) ? callText : $"{content}\n{callText}";
            }

            if (msg.Role == "tool" && options.ToolCallMode != ToolCallExportMode.Omit)
                content = $"<tool_response>\n{msg.Content}\n</tool_response>";

            if (msg.Parts?.OfType<ImagePart>().Any() == true
                && options.ImageMode == ImageExportMode.Placeholder)
            {
                content = $"[image]\n{content}";
            }

            if (string.IsNullOrWhiteSpace(content)) continue;

            sb.Append($"<|im_start|>{msg.Role}\n{content}<|im_end|>\n");
        }

        return JsonSerializer.Serialize(
            new Dictionary<string, object?> { ["text"] = sb.ToString().TrimEnd() }, TrainingExporter.JsonOptions);
    }
}
