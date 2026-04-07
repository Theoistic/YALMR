using System.Text.Json;

namespace YALMR.Runtime;

/// <summary>
/// Exports conversations in the ShareGPT format:
/// <c>{"conversations": [{"from": "human", "value": "..."}]}</c>.
/// Compatible with Axolotl, LLaMA-Factory, and FastChat.
/// </summary>
public sealed class ShareGPTTrainingExporter : ITrainingExporter
{
    public static readonly ShareGPTTrainingExporter Instance = new();

    public string Export(Conversation conversation, TrainingExportOptions options)
    {
        var output = new List<Dictionary<string, object?>>();

        foreach (var msg in conversation)
        {
            if (!TrainingExporter.ShouldIncludeMessage(msg, options)) continue;

            string from = msg.Role switch
            {
                "user" => "human",
                "assistant" => "gpt",
                _ => msg.Role
            };

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

            output.Add(new Dictionary<string, object?> { ["from"] = from, ["value"] = content });
        }

        return JsonSerializer.Serialize(
            new Dictionary<string, object?> { ["conversations"] = output }, TrainingExporter.JsonOptions);
    }
}
