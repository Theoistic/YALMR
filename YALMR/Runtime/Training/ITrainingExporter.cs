namespace YALMR.Runtime;

/// <summary>
/// Defines a strategy for exporting a <see cref="Conversation"/> as training data.
/// Implement this interface to add custom training formats beyond the built-in ones.
/// </summary>
public interface ITrainingExporter
{
    /// <summary>
    /// Exports a single conversation as one or more JSONL lines.
    /// </summary>
    string Export(Conversation conversation, TrainingExportOptions options);
}
