using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;
using System.Text.Json.Serialization;

namespace YALMR.Runtime;

/// <summary>
/// Converts a C# type into a GBNF grammar string that constrains llama.cpp sampling
/// to only produce valid JSON matching the type's schema.
/// </summary>
public static class GbnfSchemaGenerator
{
    private static readonly JsonSerializerOptions s_schemaOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    // Common terminal rules appended to every grammar.
    private const string CommonRules = """
        string  ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
        number  ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (([eE] [-+]? [0-9]+))?
        integer ::= "-"? ([0-9] | [1-9] [0-9]*)
        boolean ::= "true" | "false"
        null    ::= "null"
        ws      ::= ([ \t\n\r] ws)?
        """;

    /// <summary>
    /// Generates a GBNF grammar for <typeparamref name="T"/>.
    /// </summary>
    public static string FromType<T>() => FromType(typeof(T));

    /// <summary>
    /// Generates a GBNF grammar for the given <paramref name="type"/>.
    /// </summary>
    public static string FromType(Type type)
    {
        var schema = JsonSchemaExporter.GetJsonSchemaAsNode(s_schemaOptions, type);
        return FromJsonSchemaNode(schema);
    }

    /// <summary>
    /// Generates a GBNF grammar from a raw JSON Schema node.
    /// </summary>
    public static string FromJsonSchemaNode(JsonNode schema)
    {
        var rules = new Dictionary<string, string>(StringComparer.Ordinal);
        string root = BuildRule(schema, "root", rules);
        if (!rules.ContainsKey("root"))
            rules["root"] = root;

        var sb = new StringBuilder();
        // root rule always first
        sb.AppendLine($"root ::= {rules["root"]}");
        foreach (var (name, body) in rules)
        {
            if (name == "root") continue;
            sb.AppendLine($"{name} ::= {body}");
        }
        sb.AppendLine(CommonRules);
        return sb.ToString();
    }

    // -------------------------------------------------------------------------

    private static string BuildRule(JsonNode schema, string name, Dictionary<string, string> rules)
    {
        if (schema is not JsonObject obj)
            return "string"; // fallback

        // anyOf → nullable or union
        if (obj["anyOf"] is JsonArray anyOf)
        {
            var variants = anyOf
                .OfType<JsonObject>()
                .Select((v, i) => BuildRule(v, $"{name}-v{i}", rules))
                .Distinct()
                .ToList();

            // Collapse the common "T | null" pattern
            if (variants.Count == 2 && variants.Contains("null"))
            {
                string nonNull = variants.First(v => v != "null");
                return $"({nonNull} | null)";
            }

            return $"({string.Join(" | ", variants)})";
        }

        string? type = obj["type"]?.GetValue<string>();

        return type switch
        {
            "string"  => BuildStringRule(obj),
            "integer" => "integer",
            "number"  => "number",
            "boolean" => "boolean",
            "null"    => "null",
            "array"   => BuildArrayRule(obj, name, rules),
            "object"  => BuildObjectRule(obj, name, rules),
            _         => "string",
        };
    }

    private static string BuildStringRule(JsonObject obj)
    {
        // enum values
        if (obj["enum"] is JsonArray enums)
        {
            var vals = enums
                .Select(e => GbnfLiteral(e?.GetValue<string>() ?? ""))
                .ToList();
            return $"({string.Join(" | ", vals)})";
        }
        return "string";
    }

    private static string BuildArrayRule(JsonObject obj, string name, Dictionary<string, string> rules)
    {
        string itemRule = "string";
        if (obj["items"] is JsonNode itemSchema)
        {
            string itemRuleName = $"{name}-item";
            string built = BuildRule(itemSchema, itemRuleName, rules);
            // Emit a named rule for compound expressions, inline simple rule references
            if (built.Contains(' ') || built.Contains('"') || built.Contains('('))
            {
                rules[itemRuleName] = built;
                itemRule = itemRuleName;
            }
            else
            {
                itemRule = built;
            }
        }

        string ruleName = $"{name}-array";
        rules[ruleName] = $"{GbnfLiteral("[")} ws ({itemRule} ({GbnfLiteral(",")} ws {itemRule})*)? ws {GbnfLiteral("]")}";
        return ruleName;
    }

    private static string BuildObjectRule(JsonObject obj, string name, Dictionary<string, string> rules)
    {
        var properties = obj["properties"] as JsonObject;
        if (properties is null || !properties.Any())
            return GbnfLiteral("{") + " ws " + GbnfLiteral("}");

        var required = (obj["required"] as JsonArray)
            ?.Select(n => n?.GetValue<string>() ?? "")
            .ToHashSet(StringComparer.Ordinal)
            ?? new HashSet<string>(StringComparer.Ordinal);

        var sb = new StringBuilder();
        sb.Append(GbnfLiteral("{"));
        sb.Append(" ws ");

        bool first = true;
        foreach (var prop in properties)
        {
            string propName     = prop.Key;
            string propRuleName = $"{name}-{ToRuleName(propName)}";
            string propBody     = BuildRule(prop.Value ?? JsonValue.Create("string")!, propRuleName, rules);

            // Emit a named rule for compound expressions, inline simple rule references
            string valueRef;
            if (propBody.Contains(' ') || propBody.Contains('"') || propBody.Contains('('))
            {
                rules[propRuleName] = propBody;
                valueRef = propRuleName;
            }
            else
            {
                valueRef = propBody;
            }

            // Wrap non-required fields so they can be null
            if (!required.Contains(propName))
                valueRef = $"({valueRef} | null)";

            if (!first)
            {
                sb.Append(' ');
                sb.Append(GbnfLiteral(","));
                sb.Append(" ws ");
            }

            sb.Append(GbnfJsonKey(propName));
            sb.Append(" ws ");
            sb.Append(GbnfLiteral(":"));
            sb.Append(" ws ");
            sb.Append(valueRef);

            first = false;
        }

        sb.Append(" ws ");
        sb.Append(GbnfLiteral("}"));
        return sb.ToString();
    }

    /// <summary>
    /// Returns a GBNF quoted literal that matches <paramref name="rawText"/> exactly.
    /// E.g. <c>GbnfLiteral("{") → "{"</c> in GBNF.
    /// </summary>
    private static string GbnfLiteral(string rawText)
    {
        var sb = new StringBuilder(rawText.Length + 2);
        sb.Append('"');
        foreach (char c in rawText)
        {
            if (c is '"' or '\\') sb.Append('\\');
            sb.Append(c);
        }
        sb.Append('"');
        return sb.ToString();
    }

    /// <summary>
    /// Returns a GBNF literal matching a JSON property key with its surrounding quotes.
    /// E.g. for <c>"name"</c> produces <c>"\"name\""</c> (matches the literal text <c>"name"</c>).
    /// </summary>
    private static string GbnfJsonKey(string propertyName) =>
        GbnfLiteral('"' + propertyName + '"');

    private static string ToRuleName(string propName)
    {
        // Convert camelCase / PascalCase to lowercase-hyphenated for GBNF rule names
        var sb = new StringBuilder();
        foreach (char c in propName)
        {
            if (char.IsUpper(c) && sb.Length > 0) sb.Append('-');
            sb.Append(char.ToLowerInvariant(c));
        }
        return sb.ToString();
    }
}

/// <summary>
/// Typed structured-output helpers for <see cref="Session"/>.
/// </summary>
public static class StructuredOutputExtensions
{
    private static readonly JsonSerializerOptions s_deserializeOptions = new()
    {
        PropertyNamingPolicy        = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
    };

    /// <summary>
    /// Sends <paramref name="prompt"/> and deserializes the model's JSON response into
    /// <typeparamref name="T"/>. Grammar-constrained sampling ensures the output is
    /// always valid JSON that matches the type's schema.
    /// </summary>
    public static async Task<T> AskAsync<T>(
        this Session session,
        string prompt,
        CancellationToken ct = default)
    {
        string grammar  = GbnfSchemaGenerator.FromType<T>();
        var    override_ = session.DefaultInference with { Grammar = grammar };
        var    reply     = await session.SendAsync(new ChatMessage("user", prompt), override_, ct);

        string json = ExtractJson(reply.Content ?? string.Empty);

        return JsonSerializer.Deserialize<T>(json, s_deserializeOptions)
               ?? throw new InvalidOperationException("Model returned null for structured output.");
    }

    /// <summary>
    /// Sends <paramref name="message"/> and deserializes the model's JSON response into
    /// <typeparamref name="T"/>. Grammar-constrained sampling ensures the output is
    /// always valid JSON that matches the type's schema.
    /// </summary>
    public static async Task<T> AskAsync<T>(
        this Session session,
        ChatMessage message,
        CancellationToken ct = default)
    {
        string grammar   = GbnfSchemaGenerator.FromType<T>();
        var    override_ = session.DefaultInference with { Grammar = grammar };
        var    reply     = await session.SendAsync(message, override_, ct);

        string json = ExtractJson(reply.Content ?? string.Empty);

        return JsonSerializer.Deserialize<T>(json, s_deserializeOptions)
               ?? throw new InvalidOperationException("Model returned null for structured output.");
    }

    // Strip any surrounding prose / markdown fences the model may emit
    private static string ExtractJson(string text)
    {
        text = text.Trim();

        // ```json ... ``` fence
        if (text.StartsWith("```", StringComparison.Ordinal))
        {
            int start = text.IndexOf('\n');
            int end   = text.LastIndexOf("```", StringComparison.Ordinal);
            if (start >= 0 && end > start)
                return text[(start + 1)..end].Trim();
        }

        // Grab first {...} or [...] block
        int brace   = text.IndexOf('{');
        int bracket = text.IndexOf('[');

        if (brace >= 0 && (bracket < 0 || brace < bracket)) return text[brace..];
        if (bracket >= 0) return text[bracket..];

        return text;
    }
}
