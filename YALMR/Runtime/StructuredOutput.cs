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

    private static readonly JsonSchemaExporterOptions s_exporterOptions = new()
    {
        TreatNullObliviousAsNonNullable = true,
    };

    // Terminal rules that may be appended to the grammar.
    // Only terminals actually referenced by the generated rules are emitted.
    //
    // ws is bounded to at most 4 characters instead of the recursive ([ \t\n\r] ws)?
    // used by the llama.cpp reference grammar.  The recursive form allows unlimited
    // whitespace, and small models can get stuck generating spaces indefinitely when
    // they are uncertain about the next value.  Bounding ws to 4 characters is enough
    // for newline + 3-space indent while preventing the sampler from looping.
    private const string CommonRules = """
        string  ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
        number  ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (([eE] [-+]? [0-9]+))?
        integer ::= "-"? ([0-9] | [1-9] [0-9]*)
        boolean ::= "true" | "false"
        null    ::= "null"
        ws      ::= ([ \t\n\r] ([ \t\n\r] ([ \t\n\r] ([ \t\n\r])?)?)?)?
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
        var schema = JsonSchemaExporter.GetJsonSchemaAsNode(EnsureSchemaOptions(options), type, s_exporterOptions);
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
        var rules     = new Dictionary<string, string>(StringComparer.Ordinal);
        var defs      = (schema as JsonObject)?["$defs"] as JsonObject;
        var refToRule = new Dictionary<string, string>(StringComparer.Ordinal);
        string root = BuildRule(schema, "root", rules, defs, schema, refToRule);
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
        // Emit only the terminal rules actually referenced by the generated grammar.
        // ws is always emitted since every object/array rule uses it.
        string allBodies = string.Join(" ", rules.Values);
        foreach (string line in CommonRules.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            string terminalName = line.Split("::=", 2)[0].Trim();
            if (terminalName == "ws" || ReferencesTerminal(allBodies, terminalName))
                sb.AppendLine(line);
        }
        return sb.ToString();
    }

    // -------------------------------------------------------------------------

    private static string BuildRule(JsonNode schema, string name, Dictionary<string, string> rules,
        JsonObject? defs = null, JsonNode? rootSchema = null, Dictionary<string, string>? refToRule = null)
    {
        if (schema is not JsonObject obj)
            return "string"; // fallback

        // $ref → resolve from $defs or via JSON Pointer
        if (obj["$ref"] is JsonValue refVal && refVal.TryGetValue<string>(out var refPath))
        {
            // #/$defs/... (standard JSON Schema $defs)
            if (refPath.StartsWith("#/$defs/", StringComparison.Ordinal))
            {
                string defKey   = refPath["#/$defs/".Length..];
                string ruleName = ToRuleName(defKey);
                if (!rules.ContainsKey(ruleName))
                {
                    if (defs?[defKey] is JsonNode defSchema)
                    {
                        rules[ruleName] = string.Empty; // reserve slot to guard against self-referential types
                        rules[ruleName] = BuildRule(defSchema, ruleName, rules, defs, rootSchema, refToRule);
                    }
                    else return "string";
                }
                return ruleName;
            }

            // JSON Pointer (e.g. #/properties/left) — used by .NET 10+
            if (refPath.StartsWith("#/", StringComparison.Ordinal) && rootSchema is not null)
            {
                if (refToRule is not null && refToRule.TryGetValue(refPath, out string? existing))
                    return existing;

                string ptrRuleName = RefPathToRuleName(refPath);
                while (rules.ContainsKey(ptrRuleName)) ptrRuleName += "-r";

                refToRule ??= new Dictionary<string, string>(StringComparer.Ordinal);
                refToRule[refPath] = ptrRuleName;
                rules[ptrRuleName] = string.Empty; // guard against recursion

                var targetSchema = ResolveJsonPointer(rootSchema, refPath);
                if (targetSchema is not null)
                    rules[ptrRuleName] = BuildRule(targetSchema, ptrRuleName, rules, defs, rootSchema, refToRule);
                else
                    return "string";

                return ptrRuleName;
            }

            return "string";
        }

        // anyOf → nullable or union
        if (obj["anyOf"] is JsonArray anyOf)
        {
            var variants = anyOf
                .OfType<JsonObject>()
                .Select((v, i) => BuildRule(v, $"{name}-v{i}", rules, defs, rootSchema, refToRule))
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
                        : BuildRule(CloneSchemaWithType(obj, typeName), $"{name}-v{i}", rules, defs, rootSchema, refToRule);
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

        // enum values (may appear without a "type" key, e.g. JsonStringEnumConverter)
        if (obj["enum"] is JsonArray topEnums)
        {
            var vals = new List<string>();
            bool hasNull = false;
            foreach (var e in topEnums)
            {
                if (e is JsonValue v && v.TryGetValue<string>(out var s))
                    vals.Add(GbnfLiteral(s));
                else if (e is null || (e is JsonValue nv && nv.GetValueKind() == JsonValueKind.Null))
                    hasNull = true;
            }
            if (vals.Count > 0)
            {
                string alternatives = string.Join(" | ", vals);
                return hasNull ? $"({alternatives} | null)" : $"({alternatives})";
            }
        }

        return type switch
        {
            "string"  => BuildStringRule(obj),
            "integer" => "integer",
            "number"  => "number",
            "boolean" => "boolean",
            "null"    => "null",
            "array"   => BuildArrayRule(obj, name, rules, defs, rootSchema, refToRule),
            "object"  => BuildObjectRule(obj, name, rules, defs, rootSchema, refToRule),
            _         => "string",
        };
    }

    private static string BuildStringRule(JsonObject obj)
    {
        // enum values
        if (obj["enum"] is JsonArray enums)
        {
            var vals = new List<string>();
            bool hasNull = false;
            foreach (var e in enums)
            {
                if (e is JsonValue v && v.TryGetValue<string>(out var s))
                    vals.Add(GbnfLiteral(s));
                else if (e is null || (e is JsonValue nv && nv.GetValueKind() == JsonValueKind.Null))
                    hasNull = true;
            }
            if (vals.Count > 0)
            {
                string alternatives = string.Join(" | ", vals);
                return hasNull ? $"({alternatives} | null)" : $"({alternatives})";
            }
        }
        return "string";
    }

    private static string BuildArrayRule(JsonObject obj, string name, Dictionary<string, string> rules,
        JsonObject? defs = null, JsonNode? rootSchema = null, Dictionary<string, string>? refToRule = null)
    {
        string itemRule = "string";
        if (obj["items"] is JsonNode itemSchema)
        {
            string itemRuleName = $"{name}-item";
            string built = BuildRule(itemSchema, itemRuleName, rules, defs, rootSchema, refToRule);
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
        rules[ruleName] = $"{GbnfLiteral("[")} ws ({itemRule} (ws {GbnfLiteral(",")} ws {itemRule})*)? ws {GbnfLiteral("]")}";
        return ruleName;
    }

    private static string BuildObjectRule(JsonObject obj, string name, Dictionary<string, string> rules,
        JsonObject? defs = null, JsonNode? rootSchema = null, Dictionary<string, string>? refToRule = null)
    {
        var properties = obj["properties"] as JsonObject;
        if (properties is null || !properties.Any())
            return GbnfLiteral("{") + " ws " + GbnfLiteral("}");

        var sb = new StringBuilder();
        sb.Append(GbnfLiteral("{"));
        sb.Append(" ws ");

        bool first = true;
        foreach (var prop in properties)
        {
            string propName     = prop.Key;
            string propRuleName = $"{name}-{ToRuleName(propName)}";
            string propBody     = BuildRule(prop.Value ?? JsonValue.Create("string")!, propRuleName, rules, defs, rootSchema, refToRule);

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

            if (!first)
            {
                sb.Append(" ws ");
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

    private static JsonObject CloneSchemaWithType(JsonObject schema, string typeName)
    {
        var clone = (JsonObject)schema.DeepClone();
        clone["type"] = JsonValue.Create(typeName);
        return clone;
    }

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

    /// <summary>
    /// Resolves a JSON Pointer (<c>#/properties/left</c>) against <paramref name="root"/>.
    /// </summary>
    private static JsonNode? ResolveJsonPointer(JsonNode root, string pointer)
    {
        if (!pointer.StartsWith("#/", StringComparison.Ordinal)) return null;
        string[] segments = pointer[2..].Split('/');
        JsonNode? current = root;
        foreach (string segment in segments)
        {
            if (current is JsonObject obj)
                current = obj[segment];
            else if (current is JsonArray arr && int.TryParse(segment, out int index) && index < arr.Count)
                current = arr[index];
            else
                return null;
        }
        return current;
    }

    /// <summary>
    /// Converts a JSON Pointer path to a GBNF rule name.
    /// E.g. <c>#/properties/left/properties/right</c> → <c>ref-left-right</c>.
    /// </summary>
    private static string RefPathToRuleName(string refPath)
    {
        var meaningful = refPath[2..].Split('/')
            .Where(s => s is not "properties" and not "items")
            .Select(ToRuleName);
        return "ref-" + string.Join("-", meaningful);
    }

    /// <summary>
    /// Returns <see langword="true"/> when <paramref name="terminalName"/> appears in
    /// <paramref name="body"/> as a standalone word (not as part of a longer rule name).
    /// </summary>
    private static bool ReferencesTerminal(string body, string terminalName)
    {
        int i = 0;
        while ((i = body.IndexOf(terminalName, i, StringComparison.Ordinal)) >= 0)
        {
            bool prevOk = i == 0 || (body[i - 1] != '-' && !char.IsLetterOrDigit(body[i - 1]));
            bool nextOk = i + terminalName.Length >= body.Length ||
                          (body[i + terminalName.Length] != '-' && !char.IsLetterOrDigit(body[i + terminalName.Length]));
            if (prevOk && nextOk) return true;
            i++;
        }
        return false;
    }
}

/// <summary>
/// Typed structured-output helpers for <see cref="Session"/>.
/// </summary>
public static class StructuredOutputExtensions
{
    private const int MaxExtractionPromptChars = 12_000;
    private const int MaxExtractionAssistantChars = 4_000;
    private const int MaxExtractionToolChars = 2_000;

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
    /// When <paramref name="allowTools"/> is <see langword="false"/> (default), tool calling is
    /// disabled so the GBNF grammar constraint can take effect immediately.
    /// When <paramref name="allowTools"/> is <see langword="true"/>, the model processes the
    /// prompt with full tool access and the JSON from its response is deserialized directly.
    /// Deserialization to <typeparamref name="T"/> validates the output against the schema.
    /// </summary>
    public static Task<T> AskAsync<T>(
        this Session session,
        string prompt,
        JsonSerializerOptions? options = null,
        bool allowTools = false,
        CancellationToken ct = default)
        => session.AskAsync<T>(new ChatMessage("user", prompt), options, allowTools, ct);

    /// <summary>
    /// Sends <paramref name="message"/> and deserializes the model's JSON response into
    /// <typeparamref name="T"/>. Grammar-constrained sampling ensures the output is
    /// always valid JSON that matches the type's schema.
    /// When <paramref name="allowTools"/> is <see langword="false"/> (default), tool calling is
    /// disabled so the GBNF grammar constraint can take effect immediately.
    /// When <paramref name="allowTools"/> is <see langword="true"/>, the model processes the
    /// message with full tool access and the JSON from its response is deserialized directly.
    /// Deserialization to <typeparamref name="T"/> validates the output against the schema.
    /// </summary>
    public static async Task<T> AskAsync<T>(
        this Session session,
        ChatMessage message,
        JsonSerializerOptions? options = null,
        bool allowTools = false,
        CancellationToken ct = default)
    {
        string grammar = GbnfSchemaGenerator.FromType<T>(options);

        if (allowTools)
        {
            // Phase 1: full tool access — model analyses with tools, no grammar constraint.
            await session.SendAsync(message, ct);

            // Extract from a bounded turn-local prompt so the second pass only
            // sees the original request plus evidence from the current turn.
            ChatMessage extractionMessage = BuildToolExtractionMessage(message, session.History);
            var grammarInference = session.DefaultInference with { Grammar = grammar, Tools = null };
            string extractedText = await session.ExtractIsolatedAsync([extractionMessage], grammarInference, ct);
            string extracted = ExtractJson(extractedText);

            return JsonSerializer.Deserialize<T>(extracted, options ?? s_deserializeOptions)
                   ?? throw new InvalidOperationException("Model returned null for structured output.");
        }

        var override_ = session.DefaultInference with { Grammar = grammar, Tools = null };
        var reply     = await session.SendAsync(message, override_, ct);
        string json   = ExtractJson(reply.Content ?? string.Empty);

        return JsonSerializer.Deserialize<T>(json, options ?? s_deserializeOptions)
               ?? throw new InvalidOperationException("Model returned null for structured output.");
    }

    private static ChatMessage BuildToolExtractionMessage(ChatMessage originalMessage, IReadOnlyList<ChatMessage> history)
    {
        IReadOnlyList<ChatMessage> turnMessages = GetCurrentTurnMessages(history);
        string? finalAssistant = turnMessages
            .LastOrDefault(m => m.Role == "assistant" && !string.IsNullOrWhiteSpace(m.Content))
            ?.Content;

        var sb = new StringBuilder();
        int remaining = MaxExtractionPromptChars;

        AppendSection(sb, "Produce ONLY the JSON result that matches the requested schema. Use the evidence below, but do not add commentary.", ref remaining);
        AppendSection(sb, "Original user request:", ref remaining);
        AppendSection(sb, GetMessageText(originalMessage), ref remaining);

        if (!string.IsNullOrWhiteSpace(finalAssistant))
        {
            AppendSection(sb, "Assistant answer from the tool-enabled pass:", ref remaining);
            AppendSection(sb, TrimForPrompt(finalAssistant, MaxExtractionAssistantChars), ref remaining);
        }

        int toolIndex = 0;
        foreach (ChatMessage toolMessage in turnMessages)
        {
            if (toolMessage.Role != "tool" || string.IsNullOrWhiteSpace(toolMessage.Content))
                continue;

            toolIndex++;
            AppendSection(sb, $"Tool result {toolIndex}:", ref remaining);
            AppendSection(sb, TrimForPrompt(toolMessage.Content, MaxExtractionToolChars), ref remaining);

            if (remaining <= 0)
                break;
        }

        if (toolIndex == 0 && string.IsNullOrWhiteSpace(finalAssistant))
            AppendSection(sb, "No tool evidence was captured. Follow the original request strictly.", ref remaining);

        return new ChatMessage("user", sb.ToString().Trim());
    }

    private static IReadOnlyList<ChatMessage> GetCurrentTurnMessages(IReadOnlyList<ChatMessage> history)
    {
        for (int i = history.Count - 1; i >= 0; i--)
        {
            if (history[i].Role == "user")
                return [.. history.Skip(i)];
        }

        return history;
    }

    private static string GetMessageText(ChatMessage message)
    {
        if (!string.IsNullOrWhiteSpace(message.Content))
            return message.Content;

        if (message.Parts is not { Count: > 0 })
            return string.Empty;

        return string.Join("\n", message.Parts.OfType<TextPart>().Select(p => p.Text));
    }

    private static string TrimForPrompt(string text, int maxChars)
    {
        text = text.Trim();
        if (text.Length <= maxChars)
            return text;

        const string marker = "\n...[truncated]...";
        int take = Math.Max(0, maxChars - marker.Length);
        return text[..take] + marker;
    }

    private static void AppendSection(StringBuilder sb, string text, ref int remaining)
    {
        if (remaining <= 0 || string.IsNullOrWhiteSpace(text))
            return;

        string chunk = text.Trim();
        if (chunk.Length > remaining)
            chunk = chunk[..remaining];

        sb.AppendLine(chunk);
        sb.AppendLine();
        remaining -= chunk.Length;
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
