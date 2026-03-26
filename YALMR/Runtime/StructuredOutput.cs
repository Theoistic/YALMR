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
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver(),
    };

    private static readonly JsonSchemaExporterOptions s_exporterOptions = new()
    {
        TreatNullObliviousAsNonNullable = true,
    };

    public sealed record Options
    {
        /// <summary>
        /// Emit compact JSON grammar with no interior whitespace. This is the most stable mode for llama.cpp.
        /// </summary>
        public bool CompactJson { get; init; } = true;

        /// <summary>
        /// Allow arrays to be empty. For extraction workloads you may want false.
        /// </summary>
        public bool AllowEmptyArrays { get; init; } = true;

        /// <summary>
        /// Use underscore rule names instead of hyphenated names.
        /// Note: llama.cpp's grammar parser only supports [a-zA-Z0-9-] in rule names,
        /// so this should be false when using llama.cpp backends.
        /// </summary>
        public bool UseUnderscoreRuleNames { get; init; } = false;

        /// <summary>
        /// Emit only terminal rules actually referenced by generated rules.
        /// </summary>
        public bool EmitOnlyReferencedTerminals { get; init; } = true;

        /// <summary>
        /// Include a ws terminal and allow optional whitespace around structural tokens.
        /// CompactJson=false will automatically use ws.
        /// </summary>
        public bool AllowInteriorWhitespace { get; init; } = false;
    }

    private const string CommonRules = """
        string  ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
        number  ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (([eE] [-+]? [0-9]+))?
        integer ::= "-"? ([0-9] | [1-9] [0-9]*)
        boolean ::= "true" | "false"
        null    ::= "null"
        ws      ::= [ \t\n\r]*
        """;

    public static string FromType<T>(JsonSerializerOptions? serializerOptions = null, Options? generatorOptions = null)
        => FromType(typeof(T), serializerOptions, generatorOptions);

    public static string FromType(Type type, JsonSerializerOptions? serializerOptions = null, Options? generatorOptions = null)
    {
        var schema = JsonSchemaExporter.GetJsonSchemaAsNode(EnsureSchemaOptions(serializerOptions), type, s_exporterOptions);
        return FromJsonSchemaNode(schema, generatorOptions);
    }

    public static string FromJsonSchemaNode(JsonNode schema, Options? generatorOptions = null)
    {
        var options = generatorOptions ?? new Options();
        var context = new GeneratorContext(schema, options);
        string rootRef = context.EnsureRule(schema, "root_type");

        var sb = new StringBuilder();
        sb.AppendLine($"root ::= {rootRef}");

        foreach (var kvp in context.Rules)
        {
            if (string.IsNullOrEmpty(kvp.Value))
                continue;

            sb.AppendLine($"{kvp.Key} ::= {kvp.Value}");
        }

        sb.AppendLine();

        string allBodies = string.Join(" ", context.Rules.Values.Where(v => !string.IsNullOrEmpty(v)));
        foreach (string line in CommonRules.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            string terminalName = line.Split("::=", 2)[0].Trim();
            if (!options.EmitOnlyReferencedTerminals || terminalName == "ws" && context.UsesWhitespace || ReferencesRule(allBodies, terminalName))
            {
                if (terminalName == "ws" && !context.UsesWhitespace)
                    continue;

                sb.AppendLine(line);
            }
        }

        return sb.ToString();
    }

    private static JsonSerializerOptions EnsureSchemaOptions(JsonSerializerOptions? options)
    {
        if (options is null)
            return s_schemaOptions;

        if (options.IsReadOnly || options.TypeInfoResolver is not null)
            return options;

        return new JsonSerializerOptions(options)
        {
            TypeInfoResolver = new DefaultJsonTypeInfoResolver()
        };
    }

    private sealed class GeneratorContext
    {
        private readonly JsonNode _rootSchema;
        private readonly JsonObject? _defs;
        private readonly Options _options;

        // Memoization by schema path / ref key.
        private readonly Dictionary<string, string> _schemaKeyToRule = new(StringComparer.Ordinal);

        public Dictionary<string, string> Rules { get; } = new(StringComparer.Ordinal);

        public bool UsesWhitespace => !_options.CompactJson || _options.AllowInteriorWhitespace;

        public GeneratorContext(JsonNode rootSchema, Options options)
        {
            _rootSchema = rootSchema;
            _defs = (rootSchema as JsonObject)?["$defs"] as JsonObject;
            _options = options;
        }

        public string EnsureRule(JsonNode? schema, string suggestedName, string? schemaKey = null)
        {
            if (schema is not JsonObject obj)
                return "string";

            if (TryGetDirectTerminal(obj, out string? terminal))
                return terminal!;

            if (obj["$ref"] is JsonValue refVal && refVal.TryGetValue<string>(out string? refPath))
                return ResolveReference(refPath!);

            schemaKey ??= suggestedName;
            if (_schemaKeyToRule.TryGetValue(schemaKey, out string? existing))
                return existing;

            string ruleName = EnsureUniqueName(suggestedName);
            _schemaKeyToRule[schemaKey] = ruleName;
            Rules[ruleName] = string.Empty; // recursion sentinel
            Rules[ruleName] = BuildRuleBody(obj, ruleName, schemaKey);
            return ruleName;
        }

        private string BuildRuleBody(JsonObject obj, string ruleName, string schemaKey)
        {
            if (obj["enum"] is JsonArray enums)
                return BuildEnumBody(enums);

            if (obj["anyOf"] is JsonArray anyOf)
                return BuildAnyOfBody(anyOf, ruleName, schemaKey);

            if (obj["type"] is JsonArray typeArray)
                return BuildTypeArrayBody(obj, typeArray, ruleName, schemaKey);

            string? type = obj["type"] is JsonValue typeVal && typeVal.TryGetValue<string>(out string? typeName)
                ? typeName
                : null;

            return type switch
            {
                "object" => BuildObjectBody(obj, ruleName, schemaKey),
                "array" => BuildArrayBody(obj, ruleName, schemaKey),
                "integer" => "integer",
                "number" => "number",
                "boolean" => "boolean",
                "null" => "null",
                _ => "string",
            };
        }

        private bool TryGetDirectTerminal(JsonObject obj, out string? terminal)
        {
            terminal = null;

            if (obj["enum"] is JsonArray)
                return false;
            if (obj["anyOf"] is JsonArray)
                return false;
            if (obj["type"] is JsonArray)
                return false;

            string? type = obj["type"] is JsonValue typeVal && typeVal.TryGetValue<string>(out string? typeName)
                ? typeName
                : null;

            terminal = type switch
            {
                "string" => "string",
                "integer" => "integer",
                "number" => "number",
                "boolean" => "boolean",
                "null" => "null",
                _ => null,
            };

            return terminal is not null;
        }

        private string BuildAnyOfBody(JsonArray anyOf, string ruleName, string schemaKey)
        {
            var variants = new List<string>();
            for (int i = 0; i < anyOf.Count; i++)
            {
                var child = anyOf[i];
                if (child is null)
                    continue;

                variants.Add(EnsureRule(child, $"{ruleName}_v{i}", $"{schemaKey}/anyOf/{i}"));
            }

            return JoinDistinctAlternatives(variants);
        }

        private string BuildTypeArrayBody(JsonObject original, JsonArray types, string ruleName, string schemaKey)
        {
            var variants = new List<string>();
            int branch = 0;
            foreach (var entry in types)
            {
                if (entry is not JsonValue jv || !jv.TryGetValue<string>(out string? typeName))
                    continue;

                if (typeName == "null")
                {
                    variants.Add("null");
                    continue;
                }

                var clone = (JsonObject)original.DeepClone();
                clone["type"] = typeName;
                variants.Add(EnsureRule(clone, $"{ruleName}_{NormalizeRuleName(typeName)}", $"{schemaKey}/type/{branch++}:{typeName}"));
            }

            return JoinDistinctAlternatives(variants);
        }

        private string BuildObjectBody(JsonObject obj, string ruleName, string schemaKey)
        {
            var properties = obj["properties"] as JsonObject;
            if (properties is null || properties.Count == 0)
                return WrapObject(Array.Empty<string>());

            var pairs = new List<string>(properties.Count);
            foreach (var property in properties)
            {
                string keyLiteral = GbnfJsonKey(property.Key);
                string valueRule = EnsureRule(property.Value, $"{ruleName}_{NormalizeRuleName(property.Key)}", $"{schemaKey}/properties/{property.Key}");
                pairs.Add($"{keyLiteral}{MaybeWs()}\":\"{MaybeWs()}{valueRule}");
            }

            return WrapObject(pairs);
        }

        private string BuildArrayBody(JsonObject obj, string ruleName, string schemaKey)
        {
            string itemRule = EnsureRule(obj["items"], $"{ruleName}_item", $"{schemaKey}/items");

            string tail = $"({GbnfLiteral(",")} {itemRule})*";
            string body = _options.AllowEmptyArrays
                ? $"({itemRule} {tail})?"
                : $"{itemRule} {tail}";

            return WrapArray(body);
        }

        private string BuildEnumBody(JsonArray enums)
        {
            var variants = new List<string>();
            foreach (var entry in enums)
            {
                if (entry is null)
                {
                    variants.Add("null");
                    continue;
                }

                if (entry is not JsonValue v)
                    continue;

                if (v.TryGetValue<string>(out string? s))
                {
                    variants.Add(GbnfLiteral($"\"{s}\""));
                    continue;
                }

                if (v.TryGetValue<bool>(out bool b))
                {
                    variants.Add(b ? "true" : "false");
                    continue;
                }

                if (v.GetValueKind() == JsonValueKind.Null)
                {
                    variants.Add("null");
                    continue;
                }

                // For numeric enums, emit the exact JSON literal.
                variants.Add(v.ToJsonString());
            }

            return JoinDistinctAlternatives(variants);
        }

        private string ResolveReference(string refPath)
        {
            string schemaKey = $"ref:{refPath}";
            if (_schemaKeyToRule.TryGetValue(schemaKey, out string? existing))
                return existing;

            JsonNode? target = ResolveJsonPointer(_rootSchema, refPath, _defs);
            if (target is null)
                return "string";

            string ruleName = RefPathToRuleName(refPath);
            return EnsureRule(target, ruleName, schemaKey);
        }

        private string EnsureUniqueName(string baseName)
        {
            string normalized = NormalizeRuleName(baseName);
            if (!Rules.ContainsKey(normalized))
                return normalized;

            int i = 1;
            while (Rules.ContainsKey($"{normalized}_{i}"))
                i++;

            return $"{normalized}_{i}";
        }

        private string NormalizeRuleName(string name)
        {
            var sb = new StringBuilder(name.Length + 8);
            bool prevUnderscore = false;

            foreach (char c in name)
            {
                char mapped;
                if (char.IsLetterOrDigit(c))
                {
                    mapped = char.ToLowerInvariant(c);
                }
                else
                {
                    mapped = _options.UseUnderscoreRuleNames ? '_' : '-';
                }

                if ((mapped == '_' || mapped == '-') && (sb.Length == 0 || prevUnderscore))
                    continue;

                sb.Append(mapped);
                prevUnderscore = mapped == '_' || mapped == '-';
            }

            if (sb.Length == 0)
                sb.Append("rule");

            return sb.ToString().TrimEnd('_', '-');
        }

        private string WrapObject(IReadOnlyList<string> pairs)
        {
            if (pairs.Count == 0)
                return UsesWhitespace ? "\"{\" ws \"}\"" : "\"{\" \"}\"";

            string sep = UsesWhitespace ? " ws \",\" ws " : " \",\" ";
            string inner = string.Join(sep, pairs);
            return UsesWhitespace ? $"\"{{\" ws {inner} ws \"}}\"" : $"\"{{\" {inner} \"}}\"";
        }

        private string WrapArray(string inner)
        {
            return UsesWhitespace
                ? $"{GbnfLiteral("[")} ws {inner} ws {GbnfLiteral("]")}"
                : $"{GbnfLiteral("[")} {inner} {GbnfLiteral("]")}";
        }

        private string CommaSeparatedTail(string itemRule)
        {
            return UsesWhitespace
                ? $"(ws \",\" ws {itemRule})*"
                : $"(\",\" {itemRule})*";
        }

        private string MaybeWs() => UsesWhitespace ? " ws " : " ";

        private static string JoinDistinctAlternatives(IEnumerable<string> variants)
        {
            string[] distinct = variants.Where(v => !string.IsNullOrWhiteSpace(v)).Distinct(StringComparer.Ordinal).ToArray();
            return distinct.Length switch
            {
                0 => "string",
                1 => distinct[0],
                _ => string.Join(" | ", distinct),
            };
        }

        private static JsonNode? ResolveJsonPointer(JsonNode root, string pointer, JsonObject? defs)
        {
            if (pointer == "#")
                return root;

            if (pointer.StartsWith("#/$defs/", StringComparison.Ordinal) && defs is not null)
            {
                string defKey = DecodePointerToken(pointer["#/$defs/".Length..]);
                return defs[defKey];
            }

            if (!pointer.StartsWith("#/", StringComparison.Ordinal))
                return null;

            string[] segments = pointer[2..].Split('/');
            JsonNode? current = root;
            foreach (string rawSegment in segments)
            {
                string segment = DecodePointerToken(rawSegment);
                if (current is JsonObject obj)
                {
                    current = obj[segment];
                }
                else if (current is JsonArray arr && int.TryParse(segment, out int index) && index >= 0 && index < arr.Count)
                {
                    current = arr[index];
                }
                else
                {
                    return null;
                }
            }

            return current;
        }

        private static string DecodePointerToken(string token) => token.Replace("~1", "/").Replace("~0", "~");

        private string RefPathToRuleName(string refPath)
        {
            if (refPath.StartsWith("#/$defs/", StringComparison.Ordinal))
            {
                string defKey = DecodePointerToken(refPath["#/$defs/".Length..]);
                return NormalizeRuleName(defKey);
            }

            if (!refPath.StartsWith("#/", StringComparison.Ordinal))
                return NormalizeRuleName(refPath);

            var meaningful = refPath[2..]
                .Split('/')
                .Select(DecodePointerToken)
                .Where(s => s is not "properties" and not "items" and not "$defs")
                .Select(NormalizeRuleName)
                .Where(s => s.Length > 0);

            return NormalizeRuleName("ref_" + string.Join(_options.UseUnderscoreRuleNames ? "_" : "-", meaningful));
        }
    }

    private static string GbnfLiteral(string rawText)
    {
        var sb = new StringBuilder(rawText.Length + 2);
        sb.Append('"');
        foreach (char c in rawText)
        {
            if (c is '"' or '\\')
                sb.Append('\\');
            sb.Append(c);
        }
        sb.Append('"');
        return sb.ToString();
    }

    private static string GbnfJsonKey(string propertyName) => GbnfLiteral('"' + propertyName + '"');

    private static bool ReferencesRule(string body, string ruleName)
    {
        int i = 0;
        while ((i = body.IndexOf(ruleName, i, StringComparison.Ordinal)) >= 0)
        {
            bool prevOk = i == 0 || !IsRuleChar(body[i - 1]);
            bool nextOk = i + ruleName.Length >= body.Length || !IsRuleChar(body[i + ruleName.Length]);
            if (prevOk && nextOk)
                return true;
            i++;
        }

        return false;
    }

    private static bool IsRuleChar(char c) => char.IsLetterOrDigit(c) || c is '_' or '-';
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
