using YALMR.LlamaCpp;
using YALMR.Runtime;
using YALMR.Utils;
using Xunit.Abstractions;

namespace YALMR.Tests;

/// <summary>
/// Edge-case coverage for GBNF grammar generation and end-to-end structured output.
///
/// Grammar tests run without a model (fast, always-on).  They guard against the two
/// failure modes that manifest as "halting" during <c>AskAsync</c>:
///   1. An empty rule body — the GBNF parser gets stuck with no valid continuation.
///   2. A runaway grammar size — the per-token CPU state-machine traversal becomes
///      so expensive that sampling appears to freeze even though the GPU is running.
///
/// Model tests are gated by [Trait("Category","LocalModel")] and each carries an
/// explicit <see cref="CancellationTokenSource"/> timeout so a genuine hang surfaces
/// as a failed test rather than a blocked test runner.
/// </summary>
public sealed class StructuredOutputEdgeCaseTests(ITestOutputHelper output)
{
    private const string DefaultModelPath =
        @"C:\Users\theo\.lmstudio\models\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q4_K_M.gguf";

    // ── Test-only types ──────────────────────────────────────────────────────
    // All defined as nested records (consistent with GbnfSchemaGeneratorTests pattern).

    // One-off wrapper types used by single grammar tests
    record WithList(string Title, List<string> Tags);
    record WithROList(string Title, IReadOnlyList<string> Tags);
    record WithDate(string Name, DateTimeOffset CreatedAt);
    record WithGuid(string Name, Guid Id);
    record WithDict(string Name, Dictionary<string, string> Metadata);
    record WithNullableAddress(string Name, Address? Home);
    record PersonWithSkills(string Name, int Age, string[] Skills);

    // Shared across grammar tests and model tests
    record LineItem(string Description, int Quantity, double UnitPrice);

    // Array-of-objects: highest-risk shape for per-token sampler slowdown because
    // the grammar forces every array element to satisfy all three fields before closing.
    record Cart(string OrderId, LineItem[] Items, double Total);

    record SentimentResult(
        string   Sentiment,   // "positive" | "negative" | "neutral"
        double   Confidence,
        string[] Keywords,
        string   Summary);

    // Wide object with nullable fields — tests that optional-field alternatives
    // don't balloon the grammar size.
    record PersonProfile(
        string   FirstName,
        string   LastName,
        int      Age,
        string?  Email,
        string?  Phone,
        string[] Skills,
        bool     IsActive);

    record Address(string Street, string City, string Country, string? PostalCode);

    // Three-level nesting: CompanyProfile → Address
    record CompanyProfile(
        string   CompanyName,
        Address  HeadOffice,
        string[] Departments,
        int      EmployeeCount,
        bool     IsPublic);

    // All primitive numeric types — verifies int/long → integer, float/double/decimal → number.
    record AllNumericFields(
        int     Int32Val,
        long    Int64Val,
        float   Float32Val,
        double  Float64Val,
        decimal DecimalVal);

    // ── Grammar-only edge cases (no model, always fast) ──────────────────────

    [Fact]
    public void Grammar_ListOfString_GeneratesArrayRule()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithList>();
        output.WriteLine(grammar);

        Assert.Contains("-array ::=", grammar);
        Assert.Contains("\"[\"",      grammar);
        Assert.Contains("\"]\"",      grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_IReadOnlyListOfString_GeneratesArrayRule()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithROList>();
        output.WriteLine(grammar);

        Assert.Contains("-array ::=", grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_ListOfObject_GeneratesArrayAndItemRules()
    {
        // List<LineItem> must produce an array rule that references a LineItem object
        // rule — not inline all three properties directly inside the array rule body.
        string grammar = GbnfSchemaGenerator.FromType<Cart>();
        output.WriteLine(grammar);

        Assert.Contains("-array ::=", grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_AllNumericTypes_MappedCorrectly()
    {
        // int / long  → integer terminal
        // float / double / decimal → number terminal
        string grammar = GbnfSchemaGenerator.FromType<AllNumericFields>();
        output.WriteLine(grammar);

        Assert.Contains("integer", grammar);
        Assert.Contains("number",  grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_DateTimeOffsetProperty_DoesNotCrashOrProduceEmptyRules()
    {
        // JsonSchemaExporter emits {"type":"string","format":"date-time"}.
        // BuildStringRule must fall back to the string terminal, not produce an empty body.
        string grammar = GbnfSchemaGenerator.FromType<WithDate>();
        output.WriteLine(grammar);

        Assert.StartsWith("root ::=", grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_GuidProperty_DoesNotCrashOrProduceEmptyRules()
    {
        // JsonSchemaExporter emits {"type":"string"} for Guid.
        string grammar = GbnfSchemaGenerator.FromType<WithGuid>();
        output.WriteLine(grammar);

        Assert.StartsWith("root ::=", grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_DictionaryProperty_DoesNotCrashOrProduceEmptyRules()
    {
        // JsonSchemaExporter emits {"type":"object","additionalProperties":{...}}.
        // BuildObjectRule sees no "properties" and emits an empty object literal
        // ("{" ws "}") — important that it does not leave a blank rule body.
        string grammar = GbnfSchemaGenerator.FromType<WithDict>();
        output.WriteLine(grammar);

        Assert.StartsWith("root ::=", grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_NullableNestedObject_ContainsNullAlternative()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullableAddress>();
        output.WriteLine(grammar);

        Assert.Contains("| null", grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_ThreeLevelNesting_DoesNotCrashOrProduceEmptyRules()
    {
        string grammar = GbnfSchemaGenerator.FromType<CompanyProfile>();
        output.WriteLine(grammar);

        Assert.StartsWith("root ::=", grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_WideObjectWithNullables_DoesNotCrashOrProduceEmptyRules()
    {
        string grammar = GbnfSchemaGenerator.FromType<PersonProfile>();
        output.WriteLine(grammar);

        Assert.StartsWith("root ::=", grammar);
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_FlatObject_SizeIsReasonable()
    {
        // A grammar above ~10 KB indicates runaway rule expansion.  Each token evaluation
        // walks the GBNF state machine on CPU; an oversized grammar makes per-token
        // sampling an order of magnitude slower, which looks like "halting" to the caller.
        string grammar = GbnfSchemaGenerator.FromType<PersonProfile>();
        output.WriteLine($"PersonProfile grammar: {grammar.Length} chars");

        Assert.True(grammar.Length < 10_000,
            $"Grammar unexpectedly large ({grammar.Length} chars). Oversized grammars slow per-token CPU sampling.");
    }

    [Fact]
    public void Grammar_ArrayOfObjects_SizeIsReasonable()
    {
        // Array-of-objects grammars are the largest because the item rule is fully expanded.
        string grammar = GbnfSchemaGenerator.FromType<Cart>();
        output.WriteLine($"Cart grammar: {grammar.Length} chars");

        Assert.True(grammar.Length < 10_000,
            $"Grammar unexpectedly large ({grammar.Length} chars). Oversized grammars slow per-token CPU sampling.");
    }

    [Fact]
    public void Grammar_DeepNesting_SizeIsReasonable()
    {
        string grammar = GbnfSchemaGenerator.FromType<CompanyProfile>();
        output.WriteLine($"CompanyProfile grammar: {grammar.Length} chars");

        Assert.True(grammar.Length < 10_000,
            $"Grammar unexpectedly large ({grammar.Length} chars).");
    }

    // ── End-to-end model tests ────────────────────────────────────────────────
    // Each test carries an explicit CancellationToken so a pathological grammar/model
    // combination cannot block the test runner indefinitely.
    //
    // MaxOutputTokens is sized per test: the JSON character count for the expected
    // output is estimated and doubled.  Hitting the limit mid-object causes
    // JsonSerializer to throw, which surfaces as a clear failure rather than a hang.

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task AskAsync_ArrayOfPrimitives_ReturnsValidResult()
    {
        // ~{"name":"Alex","age":32,"skills":["C#","SQL","Azure"]} ≈ 15 tokens
        await using var session = await CreateSessionAsync(maxOutputTokens: 256);
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));

        var result = await session.AskAsync<PersonWithSkills>(
            "Return: Name=Alex, Age=32, Skills=[\"C#\",\"SQL\",\"Azure\"].",
            ct: cts.Token);

        output.WriteLine($"Name={result.Name}, Age={result.Age}, Skills=[{string.Join(", ", result.Skills)}]");

        Assert.False(string.IsNullOrWhiteSpace(result.Name));
        Assert.Equal(32, result.Age);
        Assert.NotEmpty(result.Skills);
    }

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task AskAsync_NullableFields_SomeNullSomePopulated_ReturnsValidResult()
    {
        // Nullable string fields must deserialise as null without grammar rejection.
        // ~{"firstName":"Jane","lastName":"Doe","age":28,"email":null,"phone":null,
        //   "skills":["Python","ML"],"isActive":true} ≈ 28 tokens
        await using var session = await CreateSessionAsync(maxOutputTokens: 256);
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));

        var result = await session.AskAsync<PersonProfile>(
            "Return: FirstName=Jane, LastName=Doe, Age=28, no email, no phone, " +
            "Skills=[\"Python\",\"ML\"], IsActive=true.",
            ct: cts.Token);

        output.WriteLine($"Name={result.FirstName} {result.LastName}, Age={result.Age}, Active={result.IsActive}");
        output.WriteLine($"Email={result.Email ?? "<null>"}, Skills=[{string.Join(", ", result.Skills)}]");

        Assert.False(string.IsNullOrWhiteSpace(result.FirstName));
        Assert.Equal(28, result.Age);
        Assert.True(result.IsActive);
        Assert.NotEmpty(result.Skills);
    }

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task AskAsync_NestedObject_ReturnsValidResult()
    {
        // CompanyProfile → Address tests that nested object grammar rules are correctly
        // referenced and that the model can fill all required fields at both levels.
        // ≈ 55 tokens output
        await using var session = await CreateSessionAsync(maxOutputTokens: 256);
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));

        var result = await session.AskAsync<CompanyProfile>(
            "Return: CompanyName=Acme Corp, HeadOffice at 1 Main St, Springfield, USA, PostalCode=62701, " +
            "Departments=[\"Engineering\",\"Sales\"], EmployeeCount=500, IsPublic=false.",
            ct: cts.Token);

        output.WriteLine($"Company={result.CompanyName}, City={result.HeadOffice?.City}, Depts={result.Departments.Length}");

        Assert.False(string.IsNullOrWhiteSpace(result.CompanyName));
        Assert.NotNull(result.HeadOffice);
        Assert.False(string.IsNullOrWhiteSpace(result.HeadOffice.City));
        Assert.NotEmpty(result.Departments);
    }

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task AskAsync_ArrayOfObjects_ReturnsValidResult()
    {
        // Highest-risk pattern for both "halting" and truncation.  The grammar forces the
        // model to emit all three LineItem fields for every element before it can close the
        // array, so MaxOutputTokens must cover the full serialised JSON.
        //
        // ~{"orderId":"ORD-001","items":[
        //     {"description":"Widget","quantity":1,"unitPrice":9.99},
        //     {"description":"Bolts","quantity":3,"unitPrice":1.50}
        //   ],"total":14.49} ≈ 50 tokens → 512 gives ample margin.
        await using var session = await CreateSessionAsync(maxOutputTokens: 512);
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5));

        var result = await session.AskAsync<Cart>(
            "Return an order: OrderId=ORD-001, exactly 2 items: " +
            "1 Widget at 9.99 and 3 Bolts at 1.50. Total=14.49.",
            ct: cts.Token);

        output.WriteLine($"OrderId={result.OrderId}, Total={result.Total}, Items={result.Items.Length}");
        foreach (var item in result.Items)
            output.WriteLine($"  {item.Quantity}x {item.Description} @ {item.UnitPrice}");

        Assert.False(string.IsNullOrWhiteSpace(result.OrderId));
        Assert.Equal(2, result.Items.Length);
        Assert.True(result.Total > 0);
    }

    [Fact]
    [Trait("Category", "LocalModel")]
    public async Task AskAsync_ArrayAndDoubleField_ReturnsValidResult()
    {
        // SentimentResult exercises double (number terminal) and string[] (array rule)
        // coexisting in the same object — ensures field order is preserved and the
        // grammar doesn't mangle or drop either.
        await using var session = await CreateSessionAsync(maxOutputTokens: 256);
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));

        var result = await session.AskAsync<SentimentResult>(
            "Analyse: \"Absolutely loved it — fast shipping and great quality!\" " +
            "Return Sentiment=positive, Confidence=0.97, Keywords=[\"fast\",\"quality\"], " +
            "Summary as one short sentence.",
            ct: cts.Token);

        output.WriteLine($"Sentiment={result.Sentiment}, Confidence={result.Confidence:F2}");
        output.WriteLine($"Keywords=[{string.Join(", ", result.Keywords)}]");
        output.WriteLine($"Summary={result.Summary}");

        Assert.Contains(result.Sentiment,
            new[] { "positive", "negative", "neutral" },
            StringComparer.OrdinalIgnoreCase);
        Assert.InRange(result.Confidence, 0.0, 1.0);
        Assert.NotEmpty(result.Keywords);
        Assert.False(string.IsNullOrWhiteSpace(result.Summary));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static async Task<Session> CreateSessionAsync(int maxOutputTokens = 256)
    {
        string modelPath  = Environment.GetEnvironmentVariable("YALMR_STRUCTURED_OUTPUT_MODEL")
                            ?? DefaultModelPath;
        string backendDir = await LlamaRuntimeInstaller.EnsureInstalledAsync(LlamaBackend.Cpu);
        const int ctx     = 4096;

        return await Session.CreateAsync(new SessionOptions
        {
            BackendDirectory = backendDir,
            ModelPath        = modelPath,
            ToolRegistry     = [],
            ContextTokens    = ctx,
            Compaction       = new ConversationCompactionOptions(
                MaxInputTokens:        ctx,
                ReservedForGeneration: 512,
                Strategy:              ContextCompactionStrategy.PinnedSystemFifo),
            DefaultInference = new InferenceOptions
            {
                Temperature     = 0.0f,
                TopK            = 1,
                TopP            = 1.0f,
                MaxOutputTokens = maxOutputTokens,
                EnableThinking  = false,
                Seed            = 1234,
            },
        });
    }

    /// <summary>
    /// Asserts that every rule in <paramref name="grammar"/> has a non-empty body.
    /// An empty body causes the GBNF state machine to find no valid continuation for
    /// the current token, looping until <c>MaxOutputTokens</c> is exhausted.
    /// </summary>
    private void AssertNoEmptyBodyRules(string grammar)
    {
        foreach (string line in grammar.ReplaceLineEndings("\n")
                                       .Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!line.Contains("::=")) continue;
            string[] parts = line.Split("::=", 2);
            Assert.False(string.IsNullOrWhiteSpace(parts[1]),
                $"Rule '{parts[0].Trim()}' has an empty body — the GBNF sampler will stall.");
        }
    }
}
