using System.Text.Json;

namespace YALMR.Runtime;

/// <summary>
/// Exports conversations in the Alpaca format:
/// <c>{"instruction": "...", "input": "...", "output": "..."}</c>.
/// Single-turn format — multi-turn conversations are split into user/assistant pairs.
/// </summary>
public sealed class AlpacaTrainingExporter : ITrainingExporter
{
    public static readonly AlpacaTrainingExporter Instance = new();

    public string Export(Conversation conversation, TrainingExportOptions options)
    {
        string? systemPrompt = null;
        var pairs = new List<(string Instruction, string Output)>();
        string? pendingUser = null;

        foreach (var msg in conversation)
        {
            if (!TrainingExporter.ShouldIncludeMessage(msg, options)) continue;

            if (msg.Role == "system")
            {
                systemPrompt = msg.Content;
                continue;
            }

            if (msg.Role == "user")
            {
                string? text = TrainingExporter.GetTextContent(msg, options);
                if (msg.Parts?.OfType<ImagePart>().Any() == true
                    && options.ImageMode == ImageExportMode.Placeholder)
                {
                    text = $"[image]\n{text}";
                }
                pendingUser = text;
                continue;
            }

            if (msg.Role == "assistant" && pendingUser is not null)
            {
                string? output = TrainingExporter.GetTextContent(msg, options);
                if (!string.IsNullOrWhiteSpace(output))
                    pairs.Add((pendingUser, output));
                pendingUser = null;
            }
        }

        var lines = pairs.Select(p =>
        {
            string instruction = systemPrompt is not null
                ? $"{systemPrompt}\n\n{p.Instruction}"
                : p.Instruction;
            return JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["instruction"] = instruction,
                ["input"] = "",
                ["output"] = p.Output
            }, TrainingExporter.JsonOptions);
        });

        return string.Join("\n", lines);
    }
}
