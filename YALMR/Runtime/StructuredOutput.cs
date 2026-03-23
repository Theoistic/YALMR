using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;

namespace YALMR.Runtime;

/// <summary>
/// Converts a C# type into a GBNF grammar string that constrains llama.cpp sampling
/// to only produce valid JSON matching the type's schema.
/// </summary>
public static class GbnfSchemaGenerator
{
    private static readonly JsonSerializerOptions s_schemaOptions = new()
    {
        PropertyNamingPolicy   = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        TypeInfoResolver       = new DefaultJsonTypeInfoResolver(),
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
    /// <param name="options">
    /// Optional serializer options. When supplied the same naming policy is used for
    /// both the GBNF property keys and the final JSON deserialization.
    /// </param>
    public static string FromType<T>(JsonSerializerOptions? options = null) => FromType(typeof(T), options);

    /// <summary>
    /// Generates a GBNF grammar for the given <paramref name="type"/>.
    /// </summary>
    /// <param name="options">
    /// Optional serializer options. When supplied the same naming policy is used for
    /// both the GBNF property keys and the final JSON deserialization.
    /// </param>
    public static string FromType(Type type, JsonSerializerOptions? options = null)
    {
        var schema = JsonSchemaExporter.GetJsonSchemaAsNode(EnsureSchemaOptions(options), type);
        return FromJsonSchemaNode(schema);
    }

    // Returns options that are safe to pass to JsonSchemaExporter (TypeInfoResolver required).
    private static JsonSerializerOptions EnsureSchemaOptions(JsonSerializerOptions? options)
    {
        if (options is null) return s_schemaOptions;
        // Already read-only → TypeInfoResolver was set when it was first used.
        if (options.IsReadOnly || options.TypeInfoResolver is not null) return options;
        return new JsonSerializerOptions(options) { TypeInfoResolver = new DefaultJsonTypeInfoResolver() };
    }

    /// <summary>
    /// Generates a GBNF grammar from a raw JSON Schema node.
    /// </summary>
    public static string FromJsonSchemaNode(JsonNode schema)
    {
        var rules = new Dictionary<string, string>(StringComparer.Ordinal);
        var defs  = (schema as JsonObject)?["$defs"] as JsonObject;
        string root = BuildRule(schema, "root", rules, defs);
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

    private static string BuildRule(JsonNode schema, string name, Dictionary<string, string> rules, JsonObject? defs = null)
    {
        if (schema is not JsonObject obj)
            return "string"; // fallback

        // $ref → resolve from $defs
        if (obj["$ref"] is JsonValue refVal && refVal.TryGetValue<string>(out var refPath) &&
            refPath.StartsWith("#/$defs/", StringComparison.Ordinal))
        {
            string defKey   = refPath["#/$defs/".Length..];
            string ruleName = ToRuleName(defKey);
            if (!rules.ContainsKey(ruleName))
            {
                if (defs?[defKey] is JsonNode defSchema)
                {
                    rules[ruleName] = string.Empty; // reserve slot to guard against self-referential types
                    rules[ruleName] = BuildRule(defSchema, ruleName, rules, defs);
                }
                else return "string";
            }
            return ruleName;
        }

        // anyOf → nullable or union
        if (obj["anyOf"] is JsonArray anyOf)
        {
            var variants = anyOf
                .OfType<JsonObject>()
                .Select((v, i) => BuildRule(v, $"{name}-v{i}", rules, defs))
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

        // "type" can be a string or an array (e.g. ["string", "null"])
        if (obj["type"] is JsonArray typeArr)
        {
            var variants = typeArr
                .Select((v, i) =>
                {
                    if (v is not JsonValue jv || !jv.TryGetValue<string>(out var typeName)) return "string";
                    return typeName == "null"
                        ? "null"
                        : BuildRule(new JsonObject { ["type"] = JsonValue.Create(typeName) }, $"{name}-v{i}", rules, defs);
                })
                .Distinct()
                .ToList();

            if (variants.Count == 2 && variants.Contains("null"))
            {
                string nonNull = variants.First(v => v != "null");
                return $"({nonNull} | null)";
            }
            return $"({string.Join(" | ", variants)})";
        }

        string? type = obj["type"] is JsonValue typeVal && typeVal.TryGetValue<string>(out var ts) ? ts : null;

        return type switch
        {
            "string"  => BuildStringRule(obj),
            "integer" => "integer",
            "number"  => "number",
            "boolean" => "boolean",
            "null"    => "null",
            "array"   => BuildArrayRule(obj, name, rules, defs),
            "object"  => BuildObjectRule(obj, name, rules, defs),
            _         => "string",
        };
    }

    private static string BuildStringRule(JsonObject obj)
    {
        // enum values
        if (obj["enum"] is JsonArray enums)
        {
            var vals = enums
                .Select(e => e is JsonValue v && v.TryGetValue<string>(out var s) ? GbnfLiteral(s) : null)
                .OfType<string>()
                .ToList();
            if (vals.Count > 0)
                return $"({string.Join(" | ", vals)})";
        }
        return "string";
    }

    private static string BuildArrayRule(JsonObject obj, string name, Dictionary<string, string> rules, JsonObject? defs = null)
    {
        string itemRule = "string";
        if (obj["items"] is JsonNode itemSchema)
        {
            string itemRuleName = $"{name}-item";
            string built = BuildRule(itemSchema, itemRuleName, rules, defs);
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

    private static string BuildObjectRule(JsonObject obj, string name, Dictionary<string, string> rules, JsonObject? defs = null)
    {
        var properties = obj["properties"] as JsonObject;
        if (properties is null || !properties.Any())
            return GbnfLiteral("{") + " ws " + GbnfLiteral("}");

        var required = (obj["required"] as JsonArray)
            ?.Select(n => n is JsonValue v && v.TryGetValue<string>(out var s) ? s : "")
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
            string propBody     = BuildRule(prop.Value ?? JsonValue.Create("string")!, propRuleName, rules, defs);

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
        TypeInfoResolver            = new DefaultJsonTypeInfoResolver(),
        ReferenceHandler            = ReferenceHandler.IgnoreCycles,
    };

    /// <summary>
    /// Sends <paramref name="prompt"/> and deserializes the model's JSON response into
    /// <typeparamref name="T"/>. Grammar-constrained sampling ensures the output is
    /// always valid JSON that matches the type's schema.
    /// </summary>
    public static async Task<T> AskAsync<T>(
        this Session session,
        string prompt,
        JsonSerializerOptions? options = null,
        CancellationToken ct = default)
    {
        string grammar   = GbnfSchemaGenerator.FromType<T>(options);
        var    override_ = session.DefaultInference with { Grammar = grammar };
        var    reply     = await session.SendAsync(new ChatMessage("user", prompt), override_, ct);
        string json      = ExtractJson(reply.Content ?? string.Empty);

        return JsonSerializer.Deserialize<T>(json, options ?? s_deserializeOptions)
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
        JsonSerializerOptions? options = null,
        CancellationToken ct = default)
    {
        string grammar   = GbnfSchemaGenerator.FromType<T>(options);
        var    override_ = session.DefaultInference with { Grammar = grammar };
        var    reply     = await session.SendAsync(message, override_, ct);
        string json      = ExtractJson(reply.Content ?? string.Empty);

        return JsonSerializer.Deserialize<T>(json, options ?? s_deserializeOptions)
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
