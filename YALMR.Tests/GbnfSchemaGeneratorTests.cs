using System.Text.Json.Nodes;
using YALMR.Runtime;
using Xunit.Abstractions;

namespace YALMR.Tests;

/// <summary>
/// Unit tests for <see cref="GbnfSchemaGenerator"/>.
/// These tests verify grammar structure and correctness without requiring a
/// loaded model, allowing us to catch malformed GBNF before it reaches the
/// native llama_sampler_accept call.
/// </summary>
public class GbnfSchemaGeneratorTests(ITestOutputHelper output)
{
    // ── Test-only types ──────────────────────────────────────────────────────

    record SimpleRecord(string Name, int Age, double Score, bool Active);
    record WithNullable(string Name, string? Description, int? Count);
    record WithStringArray(string Title, string[] Tags);
    record WithIntArray(string Label, int[] Values);
    record NestedRecord(string Title, SimpleRecord Details);

    // Mirrors the type pattern in the reported crash
    record InvoiceResult(string InvoiceNumber, decimal Total, string[] Items);

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static IEnumerable<string> RuleLines(string grammar) =>
        grammar.ReplaceLineEndings("\n")
               .Split('\n', StringSplitOptions.RemoveEmptyEntries)
               .Where(l => l.Contains("::="));

    private static (string Name, string Body) SplitRule(string line)
    {
        var parts = line.Split("::=", 2);
        return (parts[0].Trim(), parts[1].Trim());
    }

    private static string EscapedJsonKey(string propertyName) => $"\"\\\"{propertyName}\\\"\"";

    private static void AssertContainsJsonKey(string grammar, string propertyName) =>
        Assert.Contains(EscapedJsonKey(propertyName), grammar);

    private void Dump(string grammar) => output.WriteLine($"\n{grammar}");

    private static string GetResolvedRootBody(string grammar)
    {
        var rootLine = grammar.ReplaceLineEndings("\n").Split('\n')[0];
        var body = rootLine["root ::= ".Length..];
        if (!body.Contains(' ') && !body.Contains('"'))
        {
            foreach (var line in RuleLines(grammar))
            {
                var (name, ruleBody) = SplitRule(line);
                if (name == body)
                    return ruleBody;
            }
        }
        return body;
    }

    // ── Structural

    [Fact]
    public void Grammar_StartsWithRootRule()
    {
        string grammar = GbnfSchemaGenerator.FromType<SimpleRecord>();
        Assert.StartsWith("root ::= ", grammar);
    }

    [Fact]
    public void Grammar_ContainsAllTerminalRules()
    {
        string grammar = GbnfSchemaGenerator.FromType<SimpleRecord>();
        Assert.Contains("string  ::=", grammar);
        Assert.Contains("number  ::=", grammar);
        Assert.Contains("integer ::=", grammar);
        Assert.Contains("boolean ::=", grammar);
    }

    [Theory]
    [InlineData(typeof(SimpleRecord))]
    [InlineData(typeof(WithNullable))]
    [InlineData(typeof(WithStringArray))]
    [InlineData(typeof(NestedRecord))]
    [InlineData(typeof(InvoiceResult))]
    public void Grammar_NoRuleHasEmptyBody(Type type)
    {
        string grammar = GbnfSchemaGenerator.FromType(type);
        Dump(grammar);
        foreach (var line in RuleLines(grammar))
        {
            var (name, body) = SplitRule(line);
            Assert.False(string.IsNullOrWhiteSpace(body),
                $"Rule '{name}' has an empty body");
        }
    }

    [Theory]
    [InlineData(typeof(SimpleRecord))]
    [InlineData(typeof(WithNullable))]
    [InlineData(typeof(WithStringArray))]
    [InlineData(typeof(NestedRecord))]
    [InlineData(typeof(InvoiceResult))]
    public void Grammar_RuleNamesAreValidGbnfIdentifiers(Type type)
    {
        // llama.cpp grammar-parser.cpp is_word_char: [a-zA-Z0-9-]
        string grammar = GbnfSchemaGenerator.FromType(type);
        foreach (var line in RuleLines(grammar))
        {
            var (name, _) = SplitRule(line);
            Assert.Matches(@"^[a-zA-Z][a-zA-Z0-9\-]*$", name);
        }
    }

    // ── Type mappings

    [Fact]
    public void Grammar_SimpleRecord_AllPropertyKeysPresent()
    {
        string grammar = GbnfSchemaGenerator.FromType<SimpleRecord>();
        AssertContainsJsonKey(grammar, "name");
        AssertContainsJsonKey(grammar, "age");
        AssertContainsJsonKey(grammar, "score");
        AssertContainsJsonKey(grammar, "active");
    }

    [Fact]
    public void Grammar_SimpleRecord_CorrectTypeRefsInRootRule()
    {
        string grammar  = GbnfSchemaGenerator.FromType<SimpleRecord>();
        string resolved = GetResolvedRootBody(grammar);
        Assert.Contains("string",  resolved); // Name   → string
        Assert.Contains("integer", resolved); // Age    → integer
        Assert.Contains("number",  resolved); // Score  → number
        Assert.Contains("boolean", resolved); // Active → boolean
    }

    // ── Nullable ─────────────────────────────────────────────────────────────

    [Fact]
    public void Grammar_NullableField_ContainsNullAlternative()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullable>();
        Dump(grammar);
        Assert.Contains("| null", grammar);
    }

    [Fact]
    public void Grammar_NullableField_NoSpuriousDoubleNull()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullable>();
        Assert.DoesNotContain("null | null", grammar);
    }

    // ── Arrays ───────────────────────────────────────────────────────────────

    [Fact]
    public void Grammar_StringArrayField_GeneratesArrayRule()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithStringArray>();
        Dump(grammar);
        Assert.True(RuleLines(grammar).Any(l =>
        {
            var (_, body) = SplitRule(l);
            return body.Contains("\"[\"") && body.Contains("\"]\"");
        }), "No array rule found in grammar");
        Assert.Contains("\"[\"", grammar);
        Assert.Contains("\"]\"", grammar);
    }

    [Fact]
    public void Grammar_IntArrayField_ItemRuleUsesIntegerTerminal()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithIntArray>();
        // The array rule body should reference integer
        string arrayRuleLine = RuleLines(grammar).First(l =>
        {
            var (_, body) = SplitRule(l);
            return body.Contains("\"[\"") && body.Contains("\"]\"");
        });
        Assert.Contains("integer", arrayRuleLine);
    }

    // ── JSON Schema variants ─────────────────────────────────────────────────

    [Fact]
    public void FromJsonSchemaNode_TypeAsStringArray_HandledWithoutCrash()
    {
        // .NET 10 JsonSchemaExporter can emit "type": ["string", "null"]
        // instead of anyOf — GetValue<string>() on a JsonArray would crash.
        var schema = new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject
            {
                ["value"] = new JsonObject
                {
                    ["type"] = new JsonArray { "string", "null" },
                },
            },
            ["required"] = new JsonArray { "value" },
        };

        string grammar = GbnfSchemaGenerator.FromJsonSchemaNode(schema);
        Dump(grammar);
        Assert.StartsWith("root ::= ", grammar);
        AssertContainsJsonKey(grammar, "value");
        Assert.Contains("| null", grammar);
    }

    [Fact]
    public void FromJsonSchemaNode_TypeAsIntegerArray_HandledWithoutCrash()
    {
        var schema = new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject
            {
                ["count"] = new JsonObject
                {
                    ["type"] = new JsonArray { "integer", "null" },
                },
            },
        };

        string grammar = GbnfSchemaGenerator.FromJsonSchemaNode(schema);
        Assert.StartsWith("root ::= ", grammar);
        Assert.Contains("integer", grammar);
    }

    [Fact]
    public void FromJsonSchemaNode_WithDefs_ResolvesRef()
    {
        var schema = new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject
            {
                ["address"] = new JsonObject { ["$ref"] = "#/$defs/Address" },
            },
            ["required"] = new JsonArray { "address" },
            ["$defs"] = new JsonObject
            {
                ["Address"] = new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = new JsonObject
                    {
                        ["street"] = new JsonObject { ["type"] = "string" },
                        ["city"]   = new JsonObject { ["type"] = "string" },
                    },
                    ["required"] = new JsonArray { "street", "city" },
                },
            },
        };

        string grammar = GbnfSchemaGenerator.FromJsonSchemaNode(schema);
        Dump(grammar);
        AssertContainsJsonKey(grammar, "street");
        AssertContainsJsonKey(grammar, "city");
        // The $defs type should appear as a named rule (not inlined into root)
        Assert.Contains("address ::=", grammar);
    }

    [Fact]
    public void FromJsonSchemaNode_MissingRef_FallsBackToStringWithoutCrash()
    {
        var schema = new JsonObject
        {
            ["$ref"] = "#/$defs/Missing",
        };

        string grammar = GbnfSchemaGenerator.FromJsonSchemaNode(schema);
        Assert.StartsWith("root ::= ", grammar);
    }

    [Fact]
    public void FromJsonSchemaNode_AnyOf_CollapsesNullablePattern()
    {
        var schema = new JsonObject
        {
            ["anyOf"] = new JsonArray
            {
                new JsonObject { ["type"] = "string" },
                new JsonObject { ["type"] = "null" },
            },
        };

        string grammar = GbnfSchemaGenerator.FromJsonSchemaNode(schema);
        string resolved = GetResolvedRootBody(grammar);
        // Should produce (string | null), not a multi-variant union
        Assert.Contains("string", resolved);
        Assert.Contains("| null", resolved);
    }

    // ── GbnfLiteral correctness (tested through generated output) ────────────

    [Fact]
    public void Grammar_PropertyKey_IsGbnfEscapedInOutput()
    {
        // A property named "name" must appear as "\"name\"" in the grammar
        // (GBNF literal that matches the JSON key "name" including quotes).
        string grammar = GbnfSchemaGenerator.FromType<SimpleRecord>();
        Assert.Contains("\"\\\"name\\\"\"", grammar);
    }

    [Fact]
    public void Grammar_ObjectBraces_AreGbnfLiterals()
    {
        string grammar = GbnfSchemaGenerator.FromType<SimpleRecord>();
        Assert.Contains("\"{\"", grammar);
        Assert.Contains("\"}\"", grammar);
    }

    // ── Regression: InvoiceResult crash type ─────────────────────────────────

    [Fact]
    public void Grammar_InvoiceResult_RootRuleContainsAllKeys()
    {
        string grammar  = GbnfSchemaGenerator.FromType<InvoiceResult>();
        string resolved = GetResolvedRootBody(grammar);
        Dump(grammar);
        Assert.StartsWith("root ::= ", grammar);
        Assert.Contains(EscapedJsonKey("invoiceNumber"), resolved);
        Assert.Contains(EscapedJsonKey("total"), resolved);
        Assert.Contains("number",              grammar); // decimal → number
    }

    [Fact]
    public void Grammar_InvoiceResult_ArrayRuleExistsAndIsWellFormed()
    {
        string grammar = GbnfSchemaGenerator.FromType<InvoiceResult>();
        string arrayLine = RuleLines(grammar).FirstOrDefault(l =>
            {
                var (_, body) = SplitRule(l);
                return body.Contains("\"[\"") && body.Contains("\"]\"");
            })
            ?? throw new Exception("No array rule found in grammar");
        Assert.Contains("\"[\"", arrayLine);
        Assert.Contains("\"]\"", arrayLine);
    }

    [Fact]
    public void Grammar_InvoiceResult_NoRuleHasEmptyBody()
    {
        string grammar = GbnfSchemaGenerator.FromType<InvoiceResult>();
        foreach (var line in RuleLines(grammar))
        {
            var (name, body) = SplitRule(line);
            Assert.False(string.IsNullOrWhiteSpace(body),
                $"Rule '{name}' has an empty body");
        }
    }

    [Fact]
    public void Grammar_InvoiceResult_RuleNamesAreValidIdentifiers()
    {
        string grammar = GbnfSchemaGenerator.FromType<InvoiceResult>();
        foreach (var line in RuleLines(grammar))
        {
            var (name, _) = SplitRule(line);
            Assert.Matches(@"^[a-zA-Z][a-zA-Z0-9\-]*$", name);
        }
    }
}
