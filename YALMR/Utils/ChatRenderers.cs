using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using YALMR.Runtime;

namespace YALMR.Utils;

/// <summary>
/// Renders streamed assistant output to a target format.
/// </summary>
public interface IChatRenderer
{
    /// <summary>
    /// Starts a new assistant message render pass.
    /// </summary>
    void BeginAssistantMessage();

    /// <summary>
    /// Renders a streamed output chunk.
    /// </summary>
    void Render(ChatResponseChunk chunk);

    /// <summary>
    /// Finishes rendering the current assistant message.
    /// </summary>
    void EndAssistantMessage();

    /// <summary>
    /// Renders an exception in the renderer's output format.
    /// </summary>
    void RenderError(Exception exception);

    /// <summary>
    /// Renders session-level debug information about the prompt being executed.
    /// </summary>
    void RenderDebug(SessionDebugView debugView);
}

/// <summary>
/// Base renderer that handles streamed reasoning, tool calls, and usage formatting.
/// </summary>
public abstract class StreamingChatRenderer : IChatRenderer
{
    private bool _atLineStart = true;
    private bool _reasoningOpen;
    private bool _trimLeadingText = true;
    private InferenceUsage? _usage;

    /// <summary>
    /// Resets renderer state and writes any assistant prefix.
    /// </summary>
    public void BeginAssistantMessage()
    {
        _atLineStart = true;
        _reasoningOpen = false;
        _trimLeadingText = true;
        _usage = null;
        WriteAssistantPrefix();
    }

    /// <summary>
    /// Renders one streamed response chunk.
    /// </summary>
    public void Render(ChatResponseChunk chunk)
    {
        if (chunk.ReasoningText is { Length: > 0 })
        {
            if (!_reasoningOpen)
            {
                EnsureLineStart();
                WriteReasoningStart();
                _reasoningOpen = true;
            }

            Write(chunk.ReasoningText);
        }

        if (chunk.Text is not null)
        {
            CloseReasoning();

            string text = _trimLeadingText ? chunk.Text.TrimStart() : chunk.Text;
            if (!string.IsNullOrEmpty(text))
            {
                Write(text);
                _trimLeadingText = false;
            }
        }

        if (chunk.ToolCalls is { Count: > 0 })
        {
            CloseReasoning();
            EnsureLineStart();
            WriteToolCalls(chunk.ToolCalls);
            _trimLeadingText = true;
        }

        if (chunk.Usage is not null)
            _usage = chunk.Usage;
    }

    /// <summary>
    /// Closes any open sections and writes final usage information.
    /// </summary>
    public void EndAssistantMessage()
    {
        CloseReasoning();

        if (_usage is not null)
        {
            EnsureLineStart();
            WriteUsage(_usage);
        }
    }

    /// <summary>
    /// Renders an exception after closing any open reasoning block.
    /// </summary>
    public void RenderError(Exception exception)
    {
        CloseReasoning();
        EnsureLineStart();
        WriteError(exception);
    }

    /// <summary>
    /// Renders a debug snapshot of the current session prompt state.
    /// </summary>
    public virtual void RenderDebug(SessionDebugView debugView)
    {
    }

    /// <summary>
    /// Writes raw text without appending a newline.
    /// </summary>
    protected abstract void Write(string text);

    /// <summary>
    /// Writes a line terminator and optional text.
    /// </summary>
    protected abstract void WriteLine(string text = "");

    /// <summary>
    /// Writes the assistant prefix shown at the start of a reply.
    /// </summary>
    protected virtual void WriteAssistantPrefix()
    {
    }

    /// <summary>
    /// Writes the opening marker for streamed reasoning.
    /// </summary>
    protected virtual void WriteReasoningStart() => WriteLine("<think>");

    /// <summary>
    /// Writes the closing marker for streamed reasoning.
    /// </summary>
    protected virtual void WriteReasoningEnd() => WriteLine("</think>");

    /// <summary>
    /// Writes tool call information.
    /// </summary>
    protected virtual void WriteToolCalls(IReadOnlyList<ToolCall> toolCalls) => WriteLine($"[tool] {string.Join(", ", toolCalls.Select(c => c.Name))}");

    /// <summary>
    /// Writes token usage information.
    /// </summary>
    protected virtual void WriteUsage(InferenceUsage usage) => WriteLine($"[usage] prompt={usage.PromptTokens}, completion={usage.CompletionTokens}, total={usage.TotalTokens}");

    /// <summary>
    /// Writes an error message.
    /// </summary>
    protected virtual void WriteError(Exception exception) => WriteLine($"(error: {exception.Message})");

    /// <summary>
    /// Tracks whether the current output position is at the start of a line.
    /// </summary>
    protected void MarkLineStart(bool atLineStart)
    {
        _atLineStart = atLineStart;
    }

    /// <summary>
    /// Ensures subsequent output begins on a new line.
    /// </summary>
    protected void EnsureLineStart()
    {
        if (!_atLineStart)
            WriteLine();
    }

    private void CloseReasoning()
    {
        if (!_reasoningOpen)
            return;

        EnsureLineStart();
        WriteReasoningEnd();
        _reasoningOpen = false;
        _trimLeadingText = true;
    }
}

/// <summary>
/// Streams assistant output directly to a text console.
/// </summary>
public sealed class ConsoleChatRenderer : StreamingChatRenderer
{
    private readonly TextWriter _writer;
    private static readonly JsonSerializerOptions s_jsonOptions = new() { WriteIndented = true };

    public ConsoleChatRenderer(TextWriter writer)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        _writer = writer ?? throw new ArgumentNullException(nameof(writer));
    }

    /// <summary>
    /// Writes the console assistant prompt.
    /// </summary>
    protected override void WriteAssistantPrefix()
    {
        Write("assistant> ");
    }

    /// <summary>
    /// Writes raw streamed text to the console writer.
    /// </summary>
    protected override void Write(string text)
    {
        if (string.IsNullOrEmpty(text))
            return;

        _writer.Write(text);
        _writer.Flush();
        MarkLineStart(text.EndsWith('\n'));
    }

    /// <summary>
    /// Writes a line to the console writer.
    /// </summary>
    protected override void WriteLine(string text = "")
    {
        _writer.WriteLine(text);
        _writer.Flush();
        MarkLineStart(true);
    }

    /// <summary>
    /// Writes a full debug snapshot showing the prompt view seen by the session.
    /// </summary>
    public override void RenderDebug(SessionDebugView debugView)
    {
        WriteLine("=== LMSESSION DEBUG ===");
        WriteLine($"prompt_tokens: {debugView.PromptTokens}");

        WriteLine("--- tools ---");
        WriteLine(JsonSerializer.Serialize(debugView.Tools ?? [], s_jsonOptions));

        WriteLine("--- history ---");
        WriteMessages(debugView.History);

        WriteLine("--- prompt messages ---");
        WriteMessages(debugView.PromptMessages);

        WriteLine("--- rendered prompt ---");
        WriteLine(debugView.RenderedPrompt);
        WriteLine("=== END DEBUG ===");
    }

    private void WriteMessages(IReadOnlyList<ChatMessage> messages)
    {
        if (messages.Count == 0)
        {
            WriteLine("<empty>");
            return;
        }

        for (int i = 0; i < messages.Count; i++)
        {
            ChatMessage message = messages[i];
            WriteLine($"[{i}] role={message.Role}");

            if (!string.IsNullOrWhiteSpace(message.Content))
                WriteLine($"  content: {message.Content}");

            if (!string.IsNullOrWhiteSpace(message.ReasoningContent))
                WriteLine($"  reasoning: {message.ReasoningContent}");

            if (message.ToolCalls is { Count: > 0 })
            {
                foreach (var toolCall in message.ToolCalls)
                    WriteLine($"  tool_call: {toolCall.Name} ({toolCall.CallId}) {JsonSerializer.Serialize(toolCall.Arguments, s_jsonOptions)}");
            }

            if (!string.IsNullOrWhiteSpace(message.ToolCallId))
                WriteLine($"  tool_call_id: {message.ToolCallId}");

            if (message.Parts is { Count: > 0 })
                WriteLine($"  parts: {string.Join(", ", message.Parts.Select(part => part.GetType().Name))}");
        }
    }
}

/// <summary>
/// Streams assistant output in a markdown-friendly format.
/// </summary>
public sealed class MarkdownChatRenderer(TextWriter writer) : StreamingChatRenderer
{
    private readonly TextWriter _writer = writer ?? throw new ArgumentNullException(nameof(writer));
    private static readonly JsonSerializerOptions s_jsonOptions = new() { WriteIndented = true };

    /// <summary>
    /// Writes the markdown assistant heading.
    /// </summary>
    protected override void WriteAssistantPrefix()
    {
        WriteLine("## Assistant");
        WriteLine();
    }

    /// <summary>
    /// Opens a markdown details block for reasoning.
    /// </summary>
    protected override void WriteReasoningStart()
    {
        WriteLine("<details>");
        WriteLine("<summary>Thinking</summary>");
        WriteLine();
    }

    /// <summary>
    /// Closes the markdown reasoning details block.
    /// </summary>
    protected override void WriteReasoningEnd()
    {
        EnsureLineStart();
        WriteLine();
        WriteLine("</details>");
    }

    /// <summary>
    /// Writes tool call information in markdown list form.
    /// </summary>
    protected override void WriteToolCalls(IReadOnlyList<ToolCall> toolCalls)
    {
        WriteLine($"- Tool: {string.Join(", ", toolCalls.Select(c => c.Name))}");
    }

    /// <summary>
    /// Writes usage information in markdown.
    /// </summary>
    protected override void WriteUsage(InferenceUsage usage)
    {
        WriteLine($"*Usage:* prompt={usage.PromptTokens}, completion={usage.CompletionTokens}, total={usage.TotalTokens}");
    }

    /// <summary>
    /// Writes an error in markdown quote form.
    /// </summary>
    protected override void WriteError(Exception exception)
    {
        WriteLine($"> Error: {exception.Message}");
    }

    /// <summary>
    /// Writes raw streamed text to the markdown writer.
    /// </summary>
    protected override void Write(string text)
    {
        if (string.IsNullOrEmpty(text))
            return;

        _writer.Write(text);
        _writer.Flush();
        MarkLineStart(text.EndsWith('\n'));
    }

    /// <summary>
    /// Writes a line to the markdown writer.
    /// </summary>
    protected override void WriteLine(string text = "")
    {
        _writer.WriteLine(text);
        _writer.Flush();
        MarkLineStart(true);
    }

    /// <summary>
    /// Writes a markdown representation of the current debug snapshot.
    /// </summary>
    public override void RenderDebug(SessionDebugView debugView)
    {
        WriteLine("## Session Debug");
        WriteLine();
        WriteLine($"- Prompt tokens: {debugView.PromptTokens}");
        WriteLine("- Tools:");
        WriteLine("```json");
        WriteLine(JsonSerializer.Serialize(debugView.Tools ?? [], s_jsonOptions));
        WriteLine("```");
        WriteLine();
        WriteLine("### Rendered Prompt");
        WriteLine("```text");
        WriteLine(debugView.RenderedPrompt);
        WriteLine("```");
    }
}
