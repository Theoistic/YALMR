using System;
using System.IO;

namespace YALMR.Diagnostics;

/// <summary>
/// Logging severities used by the runtime.
/// </summary>
public enum LogLevel
{
    Trace,
    Debug,
    Information,
    Warning,
    Error,
    Critical,
    None
}

/// <summary>
/// Minimal logging abstraction used throughout the runtime.
/// </summary>
public interface ILogger
{
    /// <summary>
    /// Writes a log entry.
    /// </summary>
    void Log(LogLevel level, string category, string message);
}

/// <summary>
/// Logger implementation that ignores all messages.
/// </summary>
public sealed class NullLogger : ILogger
{
    public static readonly NullLogger Instance = new();

    private NullLogger()
    {
    }

    /// <summary>
    /// Ignores the log entry.
    /// </summary>
    public void Log(LogLevel level, string category, string message)
    {
    }
}

/// <summary>
/// Writes log messages to a target <see cref="TextWriter"/>.
/// </summary>
public class TextWriterLogger : ILogger
{
    private readonly TextWriter _writer;

    public TextWriterLogger(TextWriter writer)
    {
        _writer = writer ?? throw new ArgumentNullException(nameof(writer));
    }

    /// <summary>
    /// Writes a formatted log line to the configured writer.
    /// </summary>
    public void Log(LogLevel level, string category, string message)
    {
        if (string.IsNullOrEmpty(message))
            return;

        _writer.Write($"[{category}] ");
        _writer.Write(message);
        _writer.Flush();
    }
}

/// <summary>
/// Convenience logger that writes to standard error.
/// </summary>
public sealed class ConsoleErrorLogger : TextWriterLogger
{
    public static readonly ConsoleErrorLogger Instance = new();

    /// <summary>
    /// Creates a console error logger that writes to <see cref="Console.Error"/>.
    /// </summary>
    public ConsoleErrorLogger() : base(Console.Error)
    {
    }
}
