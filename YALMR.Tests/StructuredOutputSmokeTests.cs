using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;
using Xunit.Abstractions;

namespace YALMR.Tests;

public sealed class StructuredOutputSmokeTests(ITestOutputHelper output)
{
    private const string DefaultModelPath = @"C:\Users\theo\.lmstudio\models\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q4_K_M.gguf";

    private sealed record MiniStructuredResponse(string Name, int Count, bool Confirmed);

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task AskAsync_WithLocalModel_ReturnsStructuredResponse()
    {
        string modelPath = Environment.GetEnvironmentVariable("YALMR_STRUCTURED_OUTPUT_MODEL") ?? DefaultModelPath;
        Assert.True(File.Exists(modelPath), $"Model not found: {modelPath}");

        string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);
        const int contextTokens = 4096;

        await using var session = await Session.CreateAsync(new SessionOptions
        {
            BackendDirectory = backendDir,
            ModelPath = modelPath,
            ToolRegistry = [],
            ContextTokens = contextTokens,
            Compaction = new ConversationCompactionOptions(
                MaxInputTokens: contextTokens,
                ReservedForGeneration: 256,
                Strategy: ContextCompactionStrategy.PinnedSystemFifo),
            DefaultInference = new InferenceOptions
            {
                Temperature = 0.0f,
                TopK = 1,
                TopP = 1.0f,
                MaxOutputTokens = 128,
                EnableThinking = false,
                Seed = 1234,
            },
        });

        MiniStructuredResponse response = await session.AskAsync<MiniStructuredResponse>(
            "Return a JSON object with Name as any short animal, Count as 2, and Confirmed as true.");

        output.WriteLine($"Name={response.Name}, Count={response.Count}, Confirmed={response.Confirmed}");

        Assert.False(string.IsNullOrWhiteSpace(response.Name));
        Assert.Equal(2, response.Count);
        Assert.True(response.Confirmed);
    }
}
