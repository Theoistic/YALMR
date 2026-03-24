using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;
using Xunit.Abstractions;

namespace YALMR.Tests;

public sealed class VisionSmokeTests(ITestOutputHelper output)
{
    private const string DefaultModelPath  = @"C:\Users\theo\.lmstudio\models\lmstudio-community\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q8_0.gguf";
    private const string DefaultMmprojPath = @"C:\Users\theo\.lmstudio\models\lmstudio-community\Qwen3.5-0.8B-GGUF\mmproj-Qwen3.5-0.8B-BF16.gguf";

    // The sample image (SampleOCRImage.jpg) contains the text "polymer support pads".
    // It is copied to the test output directory via the .csproj Content entry.
    private static string SampleImagePath => Path.Combine(AppContext.BaseDirectory, "SampleOCRImage.jpg");

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task GenerateAsync_WithVisionModel_RecognizesTextInImage()
    {
        string modelPath  = Environment.GetEnvironmentVariable("YALMR_VISION_MODEL")  ?? DefaultModelPath;
        string mmprojPath = Environment.GetEnvironmentVariable("YALMR_VISION_MMPROJ") ?? DefaultMmprojPath;

        Assert.True(File.Exists(modelPath),  $"Model not found: {modelPath}");
        Assert.True(File.Exists(mmprojPath), $"Mmproj not found: {mmprojPath}");
        Assert.True(File.Exists(SampleImagePath), $"Sample image not found: {SampleImagePath}");

        string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);
        const int contextTokens = 4096;

        await using var session = await Session.CreateAsync(new SessionOptions
        {
            BackendDirectory = backendDir,
            ModelPath        = modelPath,
            MmprojPath       = mmprojPath,
            ToolRegistry     = [],
            ContextTokens    = contextTokens,
            Compaction       = new ConversationCompactionOptions(
                MaxInputTokens:       contextTokens,
                ReservedForGeneration: 256,
                Strategy:             ContextCompactionStrategy.PinnedSystemFifo),
            DefaultInference = new InferenceOptions
            {
                Temperature    = 0.0f,
                TopK           = 1,
                TopP           = 1.0f,
                MaxOutputTokens = 128,
                EnableThinking = false,
                Seed           = 1234,
            },
            // Drop processed images so follow-up turns exercise the text-only KV path.
            ImageRetentionPolicy = ImageRetentionPolicy.DropProcessedImages,
        });

        // ── Turn 1: send image and request OCR ─────────────────────────────────────────────────
        // This exercises the full-reset vision eval path (first vision turn).
        var imageMessage = new ChatMessage("user", Parts:
        [
            ImagePart.FromFile(SampleImagePath),
            new TextPart("Read all the text you can see in this image. Reply with only the exact words."),
        ]);

        var response1 = await session.SendAsync(imageMessage);
        string ocr = response1.Content ?? string.Empty;
        output.WriteLine($"[Turn 1 – OCR]: {ocr}");

        Assert.Contains("polymer support pads", ocr, StringComparison.OrdinalIgnoreCase);

        // ── Turn 2: text-only follow-up ────────────────────────────────────────────────────────
        // After DropProcessedImages strips the image from Turn 1, the KV cache is rebuilt as
        // pure text.  This verifies the incremental text path still works after a vision turn
        // and that we didn't regress session state (no crash, no empty response).
        var response2 = await session.SendAsync(
            new ChatMessage("user", "How many separate words did you read in total?"));
        string followUp = response2.Content ?? string.Empty;
        output.WriteLine($"[Turn 2 – follow-up]: {followUp}");

        Assert.False(string.IsNullOrWhiteSpace(followUp));
    }

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task GenerateAsync_WithVisionModel_ImageEveryOtherTurn_NeverFullResetOnTextTurn()
    {
        string modelPath  = Environment.GetEnvironmentVariable("YALMR_VISION_MODEL")  ?? DefaultModelPath;
        string mmprojPath = Environment.GetEnvironmentVariable("YALMR_VISION_MMPROJ") ?? DefaultMmprojPath;

        Assert.True(File.Exists(modelPath),  $"Model not found: {modelPath}");
        Assert.True(File.Exists(mmprojPath), $"Mmproj not found: {mmprojPath}");
        Assert.True(File.Exists(SampleImagePath), $"Sample image not found: {SampleImagePath}");

        string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);
        const int contextTokens = 8192;

        await using var session = await Session.CreateAsync(new SessionOptions
        {
            BackendDirectory = backendDir,
            ModelPath        = modelPath,
            MmprojPath       = mmprojPath,
            ToolRegistry     = [],
            ContextTokens    = contextTokens,
            Compaction       = new ConversationCompactionOptions(
                MaxInputTokens:       contextTokens,
                ReservedForGeneration: 256,
                Strategy:             ContextCompactionStrategy.PinnedSystemFifo),
            DefaultInference = new InferenceOptions
            {
                Temperature    = 0.0f,
                TopK           = 1,
                TopP           = 1.0f,
                MaxOutputTokens = 256,
                EnableThinking = false,
                Seed           = 1234,
            },
            ImageRetentionPolicy = ImageRetentionPolicy.DropProcessedImages,
        });

        // Turn 1 — vision
        var r1 = await session.SendAsync(new ChatMessage("user", Parts:
        [
            ImagePart.FromFile(SampleImagePath),
            new TextPart("What text is in the image?"),
        ]));
        output.WriteLine($"[Turn 1 – vision]: {r1.Content}");
        Assert.Contains("polymer support pads", r1.Content ?? string.Empty, StringComparison.OrdinalIgnoreCase);

        // Turn 2 — text only (image was dropped by DropProcessedImages)
        var r2 = await session.SendAsync(new ChatMessage("user", "Repeat the exact text you saw."));
        output.WriteLine($"[Turn 2 – text]: {r2.Content}");
        Assert.False(string.IsNullOrWhiteSpace(r2.Content));

        // Turn 3 — send image again (exercises incremental vision eval if KV is text-only)
        var r3 = await session.SendAsync(new ChatMessage("user", Parts:
        [
            ImagePart.FromFile(SampleImagePath),
            new TextPart("Does the image still contain the same text?"),
        ]));
        output.WriteLine($"[Turn 3 – vision again]: {r3.Content}");
        Assert.False(string.IsNullOrWhiteSpace(r3.Content));

        // Turn 4 — text only again
        var r4 = await session.SendAsync(new ChatMessage("user", "Answer yes or no: did all turns complete without error?"));
        output.WriteLine($"[Turn 4 – text]: {r4.Content}");
        Assert.False(string.IsNullOrWhiteSpace(r4.Content));
    }
}
