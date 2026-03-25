using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;
using Xunit.Abstractions;

namespace YALMR.Tests;

[Collection("LocalModel")]
public sealed class StructuredOutputSmokeTests(ITestOutputHelper output)
{
    private const string DefaultModelPath = @"C:\Users\theo\.lmstudio\models\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q4_K_M.gguf";

    private sealed record MiniStructuredResponse(string Name, int Count, bool Confirmed);

    private sealed record OrderLine(string ProductName, int Qty, decimal Price);
    private sealed record Order(string OrderId, OrderLine[] Lines, decimal GrandTotal, bool Shipped);

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

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task AskAsync_WithLocalModel_ComplexObjectArray_ReturnsPopulatedArray()
    {
        string modelPath = Environment.GetEnvironmentVariable("YALMR_STRUCTURED_OUTPUT_MODEL") ?? DefaultModelPath;
        Assert.True(File.Exists(modelPath), $"Model not found: {modelPath}");

        string grammar = GbnfSchemaGenerator.FromType<Order>();
        output.WriteLine("=== Grammar ===");
        output.WriteLine(grammar);

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
                ReservedForGeneration: 512,
                Strategy: ContextCompactionStrategy.PinnedSystemFifo),
            DefaultInference = new InferenceOptions
            {
                Temperature = 0.0f,
                TopK = 1,
                TopP = 1.0f,
                MaxOutputTokens = 512,
                EnableThinking = false,
                Seed = 1234,
            },
        });

        Order order = await session.AskAsync<Order>(
            "Respond with ONLY a JSON object matching this schema exactly. " +
            "orderId: \"ORD-42\". " +
            "lines: array of 2 objects, first: productName \"Widget A\", qty 3, price 9.99; second: productName \"Widget B\", qty 1, price 14.99. " +
            "grandTotal: 44.96. shipped: false.");

        output.WriteLine($"OrderId={order.OrderId}, Lines={order.Lines?.Length ?? -1}, GrandTotal={order.GrandTotal}, Shipped={order.Shipped}");
        if (order.Lines is not null)
            foreach (OrderLine line in order.Lines)
                output.WriteLine($"  Line: {line.ProductName} x{line.Qty} @ {line.Price}");

        // Primary assertion: the array must not be empty (this was the original bug — grammar produced [] always)
        Assert.NotNull(order.Lines);
        Assert.True(order.Lines.Length > 0, $"Expected at least one order line but got an empty array. Grammar may be broken for complex typed arrays.");

        // Scalar fields must be populated correctly
        Assert.False(string.IsNullOrWhiteSpace(order.OrderId));
        Assert.True(order.GrandTotal > 0m, $"Expected positive grandTotal, got {order.GrandTotal}.");

        // At least one line item must have a non-empty name and positive price
        Assert.Contains(order.Lines, l => !string.IsNullOrWhiteSpace(l.ProductName) && l.Price > 0m);
    }
}
