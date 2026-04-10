using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;
using Xunit.Abstractions;

namespace YALMR.Tests;

[Collection("LocalModel")]
public sealed class Gemma4SmokeTests(ITestOutputHelper output)
{
    private const string ModelPath  = @"C:\Users\Theo\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf";
    private const string MmprojPath = @"C:\Users\Theo\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\mmproj-gemma-4-E2B-it-BF16.gguf";

    private static string SampleImagePath => Path.Combine(AppContext.BaseDirectory, "SampleOCRImage.jpg");

    private static SessionOptions BaseOptions(string backendDir, ToolRegistry? registry = null, InferenceOptions? inference = null) =>
        new()
        {
            BackendDirectory = backendDir,
            ModelPath        = ModelPath,
            ToolRegistry     = registry ?? [],
            ContextTokens    = 4096,
            Compaction       = new ConversationCompactionOptions(
                MaxInputTokens:        4096,
                ReservedForGeneration: 256,
                Strategy:              ContextCompactionStrategy.PinnedSystemFifo),
            DefaultInference = inference ?? new InferenceOptions
            {
                Temperature     = 0.0f,
                TopK            = 1,
                TopP            = 1.0f,
                MaxOutputTokens = 128,
                EnableThinking  = false,
                Seed            = 1234,
            },
        };

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task Chat_BasicResponse_ReturnsExpectedWord()
    {
        Assert.True(File.Exists(ModelPath), $"Model not found: {ModelPath}");

        string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

        await using var session = await Session.CreateAsync(BaseOptions(backendDir));

        var response = await session.SendAsync(
            new ChatMessage("user", "Reply with only the single word: pineapple"));

        string content = response.Content ?? string.Empty;
        output.WriteLine($"[Chat]: {content}");

        Assert.Contains("pineapple", content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task Vision_RecognizesTextInImage()
    {
        Assert.True(File.Exists(ModelPath),       $"Model not found: {ModelPath}");
        Assert.True(File.Exists(MmprojPath),      $"Mmproj not found: {MmprojPath}");
        Assert.True(File.Exists(SampleImagePath), $"Sample image not found: {SampleImagePath}");

        string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

        var opts = BaseOptions(backendDir) with
        {
            MmprojPath           = MmprojPath,
            ImageRetentionPolicy = ImageRetentionPolicy.DropProcessedImages,
        };

        await using var session = await Session.CreateAsync(opts);

        var response = await session.SendAsync(new ChatMessage("user", Parts:
        [
            ImagePart.FromFile(SampleImagePath),
            new TextPart("Read all the text visible in this image. Reply with only the exact words."),
        ]));

        string ocr = response.Content ?? string.Empty;
        output.WriteLine($"[Vision]: {ocr}");

        Assert.False(string.IsNullOrWhiteSpace(ocr), "Vision model returned empty OCR result.");
    }

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task ToolCalling_InvokesRegisteredTool_AndReturnsResult()
    {
        Assert.True(File.Exists(ModelPath), $"Model not found: {ModelPath}");

        string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);

        bool toolInvoked = false;

        var thermometer = new AgentTool(
            "get_temperature",
            "Returns the current outdoor temperature for a given city.",
            [new ToolParameter("city", "string", "The name of the city.")],
            args =>
            {
                toolInvoked = true;
                string city = args.TryGetValue("city", out var c) ? c?.ToString() ?? "unknown" : "unknown";
                return $"The current temperature in {city} is 22°C.";
            });

        var registry = new ToolRegistry { thermometer };

        var inference = new InferenceOptions
        {
            Temperature     = 0.0f,
            TopK            = 1,
            TopP            = 1.0f,
            MaxOutputTokens = 256,
            EnableThinking  = false,
            Seed            = 1234,
            Tools           = registry.ToToolDefinitions(),
        };

        await using var session = await Session.CreateAsync(BaseOptions(backendDir, registry, inference));

        var response = await session.SendAsync(
            new ChatMessage("user", "What is the current temperature in Paris?"));

        string content = response.Content ?? string.Empty;
        output.WriteLine($"[Tool call response]: {content}");

        Assert.True(toolInvoked, "The model did not invoke the get_temperature tool.");
        Assert.Contains("22", content, StringComparison.Ordinal);
    }
}
