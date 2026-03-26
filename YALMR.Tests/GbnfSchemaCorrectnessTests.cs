using System.Text.Json.Serialization;
using YALMR.Runtime;
using Xunit.Abstractions;

namespace YALMR.Tests;

/// <summary>
/// Verifies the <see cref="GbnfSchemaGenerator"/> produces GBNF grammars that correctly
/// reflect the C# type's nullability, primitive type mappings, and structural layout.
///
/// These tests run without a model and validate:
///   - Non-nullable properties do NOT get spurious <c>| null</c> alternatives.
///   - Nullable properties DO get <c>| null</c> alternatives (from the JSON Schema, not injected).
///   - Primitive types map to the correct GBNF terminals (number, integer, boolean).
///   - The root type is not nullable for non-nullable reference types.
///   - Array items are not nullable for non-nullable item types.
///   - Self-referential / deeply nested types produce valid recursive grammars.
///   - Enum properties produce literal alternatives.
/// </summary>
public sealed class GbnfSchemaCorrectnessTests(ITestOutputHelper output)
{
    // ── Test-only types ──────────────────────────────────────────────────────

    record SimpleAllRequired(string Name, int Age, bool Active, decimal Score);

    record WithNullableFields(string Name, string? Nickname, int? OptionalAge, decimal? OptionalScore);

    record MixedNullability(string Name, string? MiddleName, int Age, int? LuckyNumber, bool Active, bool? Verified);

    record NumericTypes(int Int32Val, long Int64Val, float FloatVal, double DoubleVal, decimal DecimalVal);

    record WithNullableNumericTypes(int? NullableInt, long? NullableLong, float? NullableFloat, double? NullableDouble, decimal? NullableDecimal);

    record WithArray(string Label, string[] Tags);

    record ArrayItem(string Name, int Quantity, decimal Price);

    record WithObjectArray(string OrderId, ArrayItem[] Items);

    record WithNullableArray(string Label, string[]? Tags);

    record WithNullableObjectArray(string Label, ArrayItem[]? Items);

    record InnerObject(string Street, string City);

    record WithNestedObject(string Name, InnerObject Address);

    record WithNullableNestedObject(string Name, InnerObject? Address);

    // Self-referential type
    record TreeNode(string Value, TreeNode? Left, TreeNode? Right);

    // Deep nesting: Level1 → Level2 → Level3 → Level4
    record Level4(string Leaf);
    record Level3(string C, Level4 Inner);
    record Level2(string B, Level3 Inner);
    record Level1(string A, Level2 Inner);

    // Deeply recursive linked list
    record LinkedNode(string Data, LinkedNode? Next);

    // Enum type
    [JsonConverter(typeof(JsonStringEnumConverter))]
    enum Priority { Low, Medium, High, Critical }

    record WithEnum(string Title, Priority Level);

    record WithNullableEnum(string Title, Priority? Level);

    // Mixed complex: nullable fields, arrays, nested objects, enums
    record ComplexDocument(
        string Title,
        string? Subtitle,
        Priority Level,
        InnerObject Address,
        InnerObject? AltAddress,
        ArrayItem[] Items,
        string[] Tags,
        int Count,
        decimal? Total,
        bool Published);

    // Regular class (not a record) — mirrors the ClassifiedInvoice production pattern
    private sealed class InvoiceClass
    {
        public string InvoiceNumber { get; set; } = "";
        public decimal Total { get; set; }
        public decimal UnitPrice { get; set; }
        public bool Paid { get; set; }
        public string? Note { get; set; }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Extracts the body of the root rule from a grammar string.
    /// </summary>
    private static string GetRootRuleBody(string grammar)
    {
        string? rootBody = null;
        foreach (string line in grammar.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            if (line.StartsWith("root ::=", StringComparison.Ordinal))
            {
                rootBody = line["root ::= ".Length..];
                break;
            }
        }
        if (rootBody is null)
            throw new InvalidOperationException("No root rule found in grammar.");

        // If the root body is a simple rule reference, resolve it
        string candidate = rootBody.Trim();
        if (!candidate.Contains(' ') && !candidate.Contains('"'))
        {
            string? resolved = GetRuleBody(grammar, candidate);
            if (resolved is not null)
                return resolved;
        }
        return rootBody;
    }

    /// <summary>
    /// Extracts the body of a named rule from a grammar string.
    /// </summary>
    private static string? GetRuleBody(string grammar, string ruleName)
    {
        string prefix = $"{ruleName} ::= ";
        foreach (string line in grammar.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            if (line.StartsWith(prefix, StringComparison.Ordinal))
                return line[prefix.Length..];
        }
        return null;
    }

    /// <summary>
    /// Returns the GBNF value expression for a given JSON property key in the root rule body.
    /// E.g. for key "name" returns "string" from: "\"name\"" ws ":" ws string
    /// </summary>
    private static string? GetPropertyValueExpr(string rootBody, string jsonKey)
    {
        // Find the pattern: "\"jsonKey\"" (":" | ws ":" ws) VALUE
        string keyLiteral = $"\"\\\"" + jsonKey + "\\\"\"";
        int idx = rootBody.IndexOf(keyLiteral, StringComparison.Ordinal);
        if (idx < 0) return null;

        // Skip past the key literal and try both compact and ws separators
        string remainder = rootBody[(idx + keyLiteral.Length)..];
        int colonIdx = -1;
        string colonPattern = " \":\" ";
        colonIdx = remainder.IndexOf(colonPattern, StringComparison.Ordinal);
        if (colonIdx < 0)
        {
            colonPattern = " ws \":\" ws ";
            colonIdx = remainder.IndexOf(colonPattern, StringComparison.Ordinal);
        }
        if (colonIdx < 0) return null;

        string afterColon = remainder[(colonIdx + colonPattern.Length)..];

        // The value expression ends at the next " " that signals ws before comma/close
        // or at end-of-rule. Handle parenthesized expressions.
        int depth = 0;
        int end = 0;
        for (int i = 0; i < afterColon.Length; i++)
        {
            char c = afterColon[i];
            if (c == '(') depth++;
            else if (c == ')') depth--;
            else if (c == ' ' && depth == 0) { end = i; break; }
            end = i + 1;
        }
        return afterColon[..end];
    }

    private void AssertNoEmptyBodyRules(string grammar)
    {
        foreach (string line in grammar.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!line.Contains("::=")) continue;
            string[] parts = line.Split("::=", 2);
            Assert.False(string.IsNullOrWhiteSpace(parts[1]),
                $"Rule '{parts[0].Trim()}' has an empty body.");
        }
    }

    // ── Non-nullable property tests ──────────────────────────────────────────

    [Fact]
    public void Grammar_NonNullableProperties_DoNotContainNull()
    {
        string grammar = GbnfSchemaGenerator.FromType<SimpleAllRequired>();
        output.WriteLine(grammar);

        string rootBody = GetRootRuleBody(grammar);

        // The root rule itself should not contain "| null"
        Assert.DoesNotContain("| null", rootBody);

        // Verify specific property types are inlined directly
        Assert.Equal("string", GetPropertyValueExpr(rootBody, "name"));
        Assert.Equal("integer", GetPropertyValueExpr(rootBody, "age"));
        Assert.Equal("boolean", GetPropertyValueExpr(rootBody, "active"));
        Assert.Equal("number", GetPropertyValueExpr(rootBody, "score"));

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_NullableProperties_ContainNullAlternative()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullableFields>();
        output.WriteLine(grammar);

        // Non-nullable Name should be directly "string"
        string rootBody = GetRootRuleBody(grammar);
        Assert.Equal("string", GetPropertyValueExpr(rootBody, "name"));

        // Nullable fields should reference rules that contain "| null"
        string? nicknameExpr = GetPropertyValueExpr(rootBody, "nickname");
        Assert.NotNull(nicknameExpr);
        // Either inlined as (string | null) or a rule reference
        if (nicknameExpr!.StartsWith('('))
            Assert.Contains("| null", nicknameExpr);
        else
        {
            string? ruleBody = GetRuleBody(grammar, nicknameExpr);
            Assert.NotNull(ruleBody);
            Assert.Contains("| null", ruleBody);
        }

        string? optionalAgeExpr = GetPropertyValueExpr(rootBody, "optionalAge");
        Assert.NotNull(optionalAgeExpr);
        if (optionalAgeExpr!.StartsWith('('))
            Assert.Contains("| null", optionalAgeExpr);
        else
        {
            string? ruleBody = GetRuleBody(grammar, optionalAgeExpr);
            Assert.NotNull(ruleBody);
            Assert.Contains("| null", ruleBody);
        }

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_MixedNullability_CorrectlyDistinguishes()
    {
        string grammar = GbnfSchemaGenerator.FromType<MixedNullability>();
        output.WriteLine(grammar);

        string rootBody = GetRootRuleBody(grammar);

        // Non-nullable fields: should NOT involve null
        Assert.Equal("string", GetPropertyValueExpr(rootBody, "name"));
        Assert.Equal("integer", GetPropertyValueExpr(rootBody, "age"));
        Assert.Equal("boolean", GetPropertyValueExpr(rootBody, "active"));

        // Nullable fields: should involve null (via rule reference or inline)
        void AssertNullable(string propertyName)
        {
            string? expr = GetPropertyValueExpr(rootBody, propertyName);
            Assert.NotNull(expr);
            if (expr!.StartsWith('('))
                Assert.Contains("| null", expr);
            else
            {
                string? ruleBody = GetRuleBody(grammar, expr);
                Assert.NotNull(ruleBody);
                Assert.Contains("| null", ruleBody);
            }
        }

        AssertNullable("middleName");
        AssertNullable("luckyNumber");
        AssertNullable("verified");

        AssertNoEmptyBodyRules(grammar);
    }

    // ── Primitive type mapping tests ─────────────────────────────────────────

    [Fact]
    public void Grammar_IntegerTypes_MapToIntegerTerminal()
    {
        string grammar = GbnfSchemaGenerator.FromType<NumericTypes>();
        output.WriteLine(grammar);

        string rootBody = GetRootRuleBody(grammar);

        // int and long should map to "integer"
        Assert.Equal("integer", GetPropertyValueExpr(rootBody, "int32Val"));
        Assert.Equal("integer", GetPropertyValueExpr(rootBody, "int64Val"));

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_FloatingPointTypes_MapToNumberTerminal()
    {
        string grammar = GbnfSchemaGenerator.FromType<NumericTypes>();
        output.WriteLine(grammar);

        string rootBody = GetRootRuleBody(grammar);

        // float, double, decimal should map to "number"
        Assert.Equal("number", GetPropertyValueExpr(rootBody, "floatVal"));
        Assert.Equal("number", GetPropertyValueExpr(rootBody, "doubleVal"));
        Assert.Equal("number", GetPropertyValueExpr(rootBody, "decimalVal"));

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_NullableNumericTypes_ContainNullAndCorrectTerminal()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullableNumericTypes>();
        output.WriteLine(grammar);

        // Each nullable numeric should have both the correct terminal AND null
        void AssertNullableTerminal(string propertyName, string expectedTerminal)
        {
            string rootBody = GetRootRuleBody(grammar);
            string? expr = GetPropertyValueExpr(rootBody, propertyName);
            Assert.NotNull(expr);

            // Resolve to rule body if it's a rule reference
            string resolved = expr!;
            if (!resolved.StartsWith('('))
            {
                string? body = GetRuleBody(grammar, resolved);
                if (body is not null) resolved = body;
            }

            Assert.Contains(expectedTerminal, resolved);
            Assert.Contains("null", resolved);
        }

        AssertNullableTerminal("nullableInt", "integer");
        AssertNullableTerminal("nullableLong", "integer");
        AssertNullableTerminal("nullableFloat", "number");
        AssertNullableTerminal("nullableDouble", "number");
        AssertNullableTerminal("nullableDecimal", "number");

        AssertNoEmptyBodyRules(grammar);
    }

    // ── Root nullability tests ───────────────────────────────────────────────

    [Fact]
    public void Grammar_RootObjectNotNullable()
    {
        string grammar = GbnfSchemaGenerator.FromType<SimpleAllRequired>();
        output.WriteLine(grammar);

        string rootBody = GetRootRuleBody(grammar);

        // The root rule should start with the object open brace, not contain "| null"
        Assert.DoesNotContain("| null", rootBody);
        Assert.Contains("\"{\"", rootBody);
    }

    // ── Array item nullability tests ─────────────────────────────────────────

    [Fact]
    public void Grammar_ArrayItems_NotNullableForNonNullableItemType()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithObjectArray>();
        output.WriteLine(grammar);

        // Find the array item rule — it should define an object, not "(object | null)"
        string? itemRule = null;
        foreach (string line in grammar.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            if (line.Contains("-item ::="))
            {
                itemRule = line;
                break;
            }
        }

        Assert.NotNull(itemRule);
        // The item rule should not end with "| null)" — it should just be the object
        Assert.DoesNotContain("| null)", itemRule);

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_NullableArrayProperty_ContainsNullAlternative()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullableArray>();
        output.WriteLine(grammar);

        // The array property itself should allow null since string[]? is nullable
        Assert.Contains("| null", grammar);

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_NullableObjectArrayProperty_ContainsNullAlternative()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullableObjectArray>();
        output.WriteLine(grammar);

        // The array property itself should allow null
        Assert.Contains("| null", grammar);

        AssertNoEmptyBodyRules(grammar);
    }

    // ── Nested object tests ──────────────────────────────────────────────────

    [Fact]
    public void Grammar_NonNullableNestedObject_DoesNotContainNull()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNestedObject>();
        output.WriteLine(grammar);

        // The root should not contain "| null" for the non-nullable Address
        string rootBody = GetRootRuleBody(grammar);
        Assert.DoesNotContain("| null", rootBody);

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_NullableNestedObject_ContainsNullAlternative()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullableNestedObject>();
        output.WriteLine(grammar);

        // The grammar should contain a null alternative for the nullable address
        Assert.Contains("| null", grammar);

        AssertNoEmptyBodyRules(grammar);
    }

    // ── Recursive / self-referential type tests ──────────────────────────────

    [Fact]
    public void Grammar_SelfReferentialType_ProducesRecursiveRules()
    {
        string grammar = GbnfSchemaGenerator.FromType<TreeNode>();
        output.WriteLine(grammar);

        // Should produce rules via JSON Pointer $ref resolution (ref-left, ref-left-right)
        Assert.Contains("ref-", grammar);

        // Left and Right are nullable TreeNode? — should have null alternatives
        Assert.Contains("| null", grammar);

        // Verify proper recursion: a ref rule should reference itself or another ref rule
        bool hasSelfRef = false;
        foreach (string line in grammar.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!line.Contains("::=")) continue;
            string[] parts = line.Split("::=", 2);
            string ruleName = parts[0].Trim();
            string ruleBody = parts[1];
            if (ruleName.StartsWith("ref-", StringComparison.Ordinal) && ruleBody.Contains(ruleName))
                hasSelfRef = true;
        }
        Assert.True(hasSelfRef, "Expected at least one self-referential ref- rule for TreeNode.");

        // No empty rule bodies (would cause sampler to stall)
        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_SelfReferentialType_RootIsNotNullable()
    {
        string grammar = GbnfSchemaGenerator.FromType<TreeNode>();
        output.WriteLine(grammar);

        string rootBody = GetRootRuleBody(grammar);

        // Root should reference the tree-node rule without a null alternative
        Assert.DoesNotContain("| null", rootBody);
    }

    [Fact]
    public void Grammar_LinkedList_ProducesRecursiveRules()
    {
        string grammar = GbnfSchemaGenerator.FromType<LinkedNode>();
        output.WriteLine(grammar);

        // Should have a self-referential rule via JSON Pointer ref resolution
        Assert.Contains("ref-next", grammar);

        // Verify ref-next references itself (proper recursion)
        string? refNextBody = GetRuleBody(grammar, "ref-next");
        Assert.NotNull(refNextBody);
        Assert.Contains("ref-next", refNextBody);

        // Next is nullable — should have null alternative
        Assert.Contains("| null", grammar);

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_LinkedList_SizeIsReasonable()
    {
        string grammar = GbnfSchemaGenerator.FromType<LinkedNode>();
        output.WriteLine($"LinkedNode grammar: {grammar.Length} chars");

        // Recursive grammars should not balloon in size
        Assert.True(grammar.Length < 5_000,
            $"Grammar unexpectedly large ({grammar.Length} chars).");
    }

    // ── Deep nesting tests ───────────────────────────────────────────────────

    [Fact]
    public void Grammar_FourLevelNesting_ProducesCorrectRules()
    {
        string grammar = GbnfSchemaGenerator.FromType<Level1>();
        output.WriteLine(grammar);

        // All levels should produce named rules or inline correctly
        Assert.StartsWith("root ::=", grammar);

        // The deepest level should still have "leaf" as a string property
        Assert.Contains("leaf", grammar);

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_FourLevelNesting_SizeIsReasonable()
    {
        string grammar = GbnfSchemaGenerator.FromType<Level1>();
        output.WriteLine($"Level1 (4-level) grammar: {grammar.Length} chars");

        Assert.True(grammar.Length < 10_000,
            $"Grammar unexpectedly large ({grammar.Length} chars).");
    }

    [Fact]
    public void Grammar_FourLevelNesting_NoNullAlternatives()
    {
        // All nested types are non-nullable — grammar should have no null alternatives
        string grammar = GbnfSchemaGenerator.FromType<Level1>();
        output.WriteLine(grammar);

        // Only the terminal null rule definition should mention "null"
        foreach (string line in grammar.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            if (line.StartsWith("null", StringComparison.Ordinal)) continue; // null ::= "null" terminal
            if (line.TrimStart().StartsWith("null", StringComparison.Ordinal)) continue; // indented terminal
            if (!line.Contains("::=")) continue;

            Assert.DoesNotContain("| null", line);
        }
    }

    // ── Enum tests ───────────────────────────────────────────────────────────

    [Fact]
    public void Grammar_EnumProperty_ProducesLiteralAlternatives()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithEnum>();
        output.WriteLine(grammar);

        // Enum values should appear as GBNF literals
        Assert.Contains("\\\"Low\\\"", grammar);
        Assert.Contains("\\\"Medium\\\"", grammar);
        Assert.Contains("\\\"High\\\"", grammar);
        Assert.Contains("\\\"Critical\\\"", grammar);

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_NullableEnumProperty_ContainsNullAlternative()
    {
        string grammar = GbnfSchemaGenerator.FromType<WithNullableEnum>();
        output.WriteLine(grammar);

        // Should contain enum literals
        Assert.Contains("\\\"Low\\\"", grammar);
        Assert.Contains("\\\"Critical\\\"", grammar);

        // Nullable enum should include a null alternative (from the enum array's null entry)
        Assert.Contains("| null", grammar);

        AssertNoEmptyBodyRules(grammar);
    }

    // ── Complex document test ────────────────────────────────────────────────

    [Fact]
    public void Grammar_ComplexDocument_AllFieldsCorrectlyTyped()
    {
        string grammar = GbnfSchemaGenerator.FromType<ComplexDocument>();
        output.WriteLine(grammar);

        string rootBody = GetRootRuleBody(grammar);

        // Non-nullable fields should not involve null in their root expression
        Assert.Equal("string", GetPropertyValueExpr(rootBody, "title"));
        Assert.Equal("integer", GetPropertyValueExpr(rootBody, "count"));
        Assert.Equal("boolean", GetPropertyValueExpr(rootBody, "published"));

        // Nullable fields should reference rules containing null
        string? subtitleExpr = GetPropertyValueExpr(rootBody, "subtitle");
        Assert.NotNull(subtitleExpr);

        string? totalExpr = GetPropertyValueExpr(rootBody, "total");
        Assert.NotNull(totalExpr);

        AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_ComplexDocument_SizeIsReasonable()
    {
        string grammar = GbnfSchemaGenerator.FromType<ComplexDocument>();
        output.WriteLine($"ComplexDocument grammar: {grammar.Length} chars");

        Assert.True(grammar.Length < 10_000,
            $"Grammar unexpectedly large ({grammar.Length} chars).");
    }

    // ── Class-based (non-record) type tests ──────────────────────────────────

    [Fact]
    public void Grammar_RegularClass_DecimalMapsToNumber()
    {
        string grammar = GbnfSchemaGenerator.FromType<InvoiceClass>();
        output.WriteLine(grammar);

        string rootBody = GetRootRuleBody(grammar);

        Assert.Equal("string",  GetPropertyValueExpr(rootBody, "invoiceNumber"));
        Assert.Equal("number",  GetPropertyValueExpr(rootBody, "total"));
        Assert.Equal("number",  GetPropertyValueExpr(rootBody, "unitPrice"));
        Assert.Equal("boolean", GetPropertyValueExpr(rootBody, "paid"));

        // string? should produce a nullable alternative
        string? noteExpr = GetPropertyValueExpr(rootBody, "note");
        Assert.NotNull(noteExpr);
        string resolved = noteExpr!;
        if (!resolved.StartsWith('('))
        {
            string? body = GetRuleBody(grammar, resolved);
            if (body is not null) resolved = body;
        }
        Assert.Contains("| null", resolved);

        AssertNoEmptyBodyRules(grammar);
    }

    // ── Structural correctness tests ─────────────────────────────────────────

    [Fact]
    public void Grammar_AllRulesHaveNonEmptyBodies()
    {
        // Run against several types to ensure no rule has an empty body
        string[] grammars =
        [
            GbnfSchemaGenerator.FromType<SimpleAllRequired>(),
            GbnfSchemaGenerator.FromType<WithNullableFields>(),
            GbnfSchemaGenerator.FromType<TreeNode>(),
            GbnfSchemaGenerator.FromType<Level1>(),
            GbnfSchemaGenerator.FromType<ComplexDocument>(),
        ];

        foreach (string grammar in grammars)
            AssertNoEmptyBodyRules(grammar);
    }

    [Fact]
    public void Grammar_AlwaysStartsWithRootRule()
    {
        string[] grammars =
        [
            GbnfSchemaGenerator.FromType<SimpleAllRequired>(),
            GbnfSchemaGenerator.FromType<WithNullableFields>(),
            GbnfSchemaGenerator.FromType<TreeNode>(),
            GbnfSchemaGenerator.FromType<Level1>(),
            GbnfSchemaGenerator.FromType<ComplexDocument>(),
        ];

        foreach (string grammar in grammars)
            Assert.StartsWith("root ::=", grammar);
    }

    [Fact]
    public void Grammar_ContainsCommonTerminals()
    {
        string grammar = GbnfSchemaGenerator.FromType<SimpleAllRequired>();
        output.WriteLine(grammar);

        Assert.Contains("string  ::=", grammar);
        Assert.Contains("number  ::=", grammar);
        Assert.Contains("integer ::=", grammar);
        Assert.Contains("boolean ::=", grammar);
    }
}
