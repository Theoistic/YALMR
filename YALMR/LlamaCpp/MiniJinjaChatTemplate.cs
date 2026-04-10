using YALMR.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace YALMR.LlamaCpp;

public static class MiniJinjaChatTemplate
{
    public static string Render(string template, IDictionary<string, object?> context)
    {
        var lexer = new TemplateLexer(template);
        var tokens = lexer.Tokenize();
        var parser = new TemplateParser(tokens);
        var nodes = parser.Parse();
        var runtime = new RuntimeContext(context);
        var sb = new StringBuilder();

        foreach (var node in nodes)
            node.Render(runtime, sb);

        return sb.ToString();
    }

    public static bool TryParseToolCalls(string text, out List<ToolCall> calls)
    {
        if (TryParseLfmToolCalls(text, out calls))
            return true;

        if (TryParseGemma4ToolCalls(text, out calls))
            return true;

        if (TryParseXmlToolCalls(text, out calls))
            return true;

        if (TryParseToolCode(text, out calls))
            return true;

        return TryParseJsonLike(text, out calls);
    }

    public static string StripToolCallMarkup(string text)
    {
        string cleaned = RemoveDelimitedBlocks(text, "<|tool_call_start|>", "<|tool_call_end|>");
        cleaned = RemoveDelimitedBlocks(cleaned, "<|tool_call>", "<tool_call|>");
        cleaned = RemoveDelimitedBlocks(cleaned, "<tool_call>", "</tool_call>");
        cleaned = Regex.Replace(cleaned, "<tool_code>\\s*.*?\\s*</tool_code>", string.Empty, RegexOptions.Singleline | RegexOptions.IgnoreCase);
        cleaned = cleaned.Replace("<think>", string.Empty, StringComparison.Ordinal);
        cleaned = cleaned.Replace("</think>", string.Empty, StringComparison.Ordinal);
        return cleaned.Trim();
    }

    // =========================================================
    // Template lexer
    // =========================================================

    private enum TemplateTokenKind
    {
        Text,
        Output,
        Statement
    }

    private readonly record struct TemplateToken(TemplateTokenKind Kind, string Value);

    private sealed class TemplateLexer
    {
        private readonly string _template;

        public TemplateLexer(string template)
        {
            _template = template ?? throw new ArgumentNullException(nameof(template));
        }

        public List<TemplateToken> Tokenize()
        {
            var result = new List<TemplateToken>();
            int i = 0;
            bool trimLeadingWhitespace = false;

            while (i < _template.Length)
            {
                int nextExpr = _template.IndexOf("{{", i, StringComparison.Ordinal);
                int nextStmt = _template.IndexOf("{%", i, StringComparison.Ordinal);

                int next;
                bool isExpr;

                if (nextExpr < 0 && nextStmt < 0)
                {
                    if (i < _template.Length)
                    {
                        string trailingText = _template[i..];
                        if (trimLeadingWhitespace)
                            trailingText = TrimLeadingWhitespace(trailingText);

                        if (trailingText.Length > 0)
                            result.Add(new TemplateToken(TemplateTokenKind.Text, trailingText));
                    }

                    break;
                }

                if (nextExpr >= 0 && (nextStmt < 0 || nextExpr < nextStmt))
                {
                    next = nextExpr;
                    isExpr = true;
                }
                else
                {
                    next = nextStmt;
                    isExpr = false;
                }

                bool trimBeforeTag = next + 2 < _template.Length && _template[next + 2] == '-';

                string text = _template[i..next];
                if (trimLeadingWhitespace)
                    text = TrimLeadingWhitespace(text);
                if (trimBeforeTag)
                    text = TrimTrailingWhitespace(text);

                if (text.Length > 0)
                    result.Add(new TemplateToken(TemplateTokenKind.Text, text));

                if (isExpr)
                {
                    int close = _template.IndexOf("}}", next + 2, StringComparison.Ordinal);
                    if (close < 0)
                        throw new InvalidOperationException("Unclosed {{ expression.");

                    int bodyStart = next + 2 + (trimBeforeTag ? 1 : 0);
                    bool trimAfterTag = close > bodyStart && _template[close - 1] == '-';
                    int bodyEnd = trimAfterTag ? close - 1 : close;

                    string body = NormalizeTag(_template[bodyStart..bodyEnd]);
                    result.Add(new TemplateToken(TemplateTokenKind.Output, body));
                    trimLeadingWhitespace = trimAfterTag;
                    i = close + 2;
                }
                else
                {
                    int close = _template.IndexOf("%}", next + 2, StringComparison.Ordinal);
                    if (close < 0)
                        throw new InvalidOperationException("Unclosed {% statement.");

                    int bodyStart = next + 2 + (trimBeforeTag ? 1 : 0);
                    bool trimAfterTag = close > bodyStart && _template[close - 1] == '-';
                    int bodyEnd = trimAfterTag ? close - 1 : close;

                    string body = NormalizeTag(_template[bodyStart..bodyEnd]);
                    result.Add(new TemplateToken(TemplateTokenKind.Statement, body));
                    trimLeadingWhitespace = trimAfterTag;
                    i = close + 2;
                }
            }

            return result;
        }

        private static string NormalizeTag(string s)
        {
            return s.Trim();
        }

        private static string TrimLeadingWhitespace(string s)
        {
            int start = 0;
            while (start < s.Length && char.IsWhiteSpace(s[start]))
                start++;

            return start == 0 ? s : s[start..];
        }

        private static string TrimTrailingWhitespace(string s)
        {
            int end = s.Length;
            while (end > 0 && char.IsWhiteSpace(s[end - 1]))
                end--;

            return end == s.Length ? s : s[..end];
        }
    }

    // =========================================================
    // Template AST
    // =========================================================

    private abstract class TemplateNode
    {
        public abstract void Render(RuntimeContext context, StringBuilder sb);
    }

    private sealed class TextNode : TemplateNode
    {
        private readonly string _text;

        public TextNode(string text)
        {
            _text = text;
        }

        public override void Render(RuntimeContext context, StringBuilder sb)
        {
            sb.Append(_text);
        }
    }

    private sealed class OutputNode : TemplateNode
    {
        private readonly Expr _expr;

        public OutputNode(Expr expr)
        {
            _expr = expr;
        }

        public override void Render(RuntimeContext context, StringBuilder sb)
        {
            object? value = _expr.Evaluate(context);
            sb.Append(Stringify(value));
        }
    }

    private sealed class SetNode : TemplateNode
    {
        private readonly AssignmentTarget _target;
        private readonly Expr _expr;

        public SetNode(AssignmentTarget target, Expr expr)
        {
            _target = target;
            _expr = expr;
        }

        public override void Render(RuntimeContext context, StringBuilder sb)
        {
            object? value = _expr.Evaluate(context);
            _target.Assign(context, value);
        }
    }

    private sealed class IfNode : TemplateNode
    {
        public readonly List<(Expr Condition, List<TemplateNode> Body)> Branches = [];
        public List<TemplateNode>? ElseBody;

        public override void Render(RuntimeContext context, StringBuilder sb)
        {
            foreach (var branch in Branches)
            {
                if (IsTruthy(branch.Condition.Evaluate(context)))
                {
                    foreach (var node in branch.Body)
                        node.Render(context, sb);
                    return;
                }
            }

            if (ElseBody is not null)
            {
                foreach (var node in ElseBody)
                    node.Render(context, sb);
            }
        }
    }

    private sealed class ForNode : TemplateNode
    {
        private readonly List<string> _variables;
        private readonly Expr _iterable;
        private readonly List<TemplateNode> _body;
        private readonly Expr? _filter;

        public ForNode(List<string> variables, Expr iterable, List<TemplateNode> body, Expr? filter = null)
        {
            _variables = variables;
            _iterable = iterable;
            _body = body;
            _filter = filter;
        }

        public override void Render(RuntimeContext context, StringBuilder sb)
        {
            object? value = _iterable.Evaluate(context);
            if (value is null || ReferenceEquals(value, Undefined.Value))
                return;

            if (value is string)
                throw new InvalidOperationException("Cannot iterate a string directly in for-loop.");

            if (value is not IEnumerable enumerable)
                throw new InvalidOperationException($"Object of type '{value.GetType().Name}' is not iterable.");

            var items = enumerable.Cast<object?>().ToList();

            if (_filter is not null)
            {
                var filtered = new List<object?>();
                foreach (var item in items)
                {
                    context.PushScope();
                    BindLoopVariables(context, item);
                    bool pass = IsTruthy(_filter.Evaluate(context));
                    context.PopScope();
                    if (pass)
                        filtered.Add(item);
                }
                items = filtered;
            }

            for (int i = 0; i < items.Count; i++)
            {
                object? item = items[i];

                context.PushScope();

                BindLoopVariables(context, item);

                context.Set(
                    "loop",
                    new Dictionary<string, object?>(StringComparer.Ordinal)
                    {
                        ["index0"] = i,
                        ["index"] = i + 1,
                        ["first"] = i == 0,
                        ["last"] = i == items.Count - 1,
                        ["length"] = items.Count,
                        ["previtem"] = i > 0 ? items[i - 1] : Undefined.Value,
                        ["nextitem"] = i + 1 < items.Count ? items[i + 1] : Undefined.Value
                    });

                foreach (var node in _body)
                    node.Render(context, sb);

                context.PopScope();
            }
        }

        private void BindLoopVariables(RuntimeContext context, object? item)
        {
            if (_variables.Count == 1)
            {
                context.Set(_variables[0], item);
                return;
            }

            if (item is IList list)
            {
                for (int i = 0; i < _variables.Count; i++)
                    context.Set(_variables[i], i < list.Count ? list[i] : Undefined.Value);
                return;
            }

            if (item is object?[] array)
            {
                for (int i = 0; i < _variables.Count; i++)
                    context.Set(_variables[i], i < array.Length ? array[i] : Undefined.Value);
                return;
            }

            if (item is KeyValuePair<string, object?> kvp)
            {
                context.Set(_variables[0], kvp.Key);
                if (_variables.Count > 1)
                    context.Set(_variables[1], kvp.Value);
                for (int i = 2; i < _variables.Count; i++)
                    context.Set(_variables[i], Undefined.Value);
                return;
            }

            throw new InvalidOperationException("Cannot destructure loop item.");
        }
    }

    private sealed class MacroNode : TemplateNode
    {
        private readonly string _name;
        private readonly List<MacroParameter> _parameters;
        private readonly List<TemplateNode> _body;

        public MacroNode(string name, List<MacroParameter> parameters, List<TemplateNode> body)
        {
            _name = name;
            _parameters = parameters;
            _body = body;
        }

        public override void Render(RuntimeContext context, StringBuilder sb)
        {
            context.Set(_name, new MacroValue(_name, _parameters, _body, context));
        }
    }

    private sealed class MacroParameter
    {
        public string Name { get; }
        public Expr? DefaultValue { get; }

        public MacroParameter(string name, Expr? defaultValue)
        {
            Name = name;
            DefaultValue = defaultValue;
        }
    }

    private sealed class AssignmentTarget
    {
        public string RootName { get; }
        public List<string> MemberPath { get; }

        public AssignmentTarget(string rootName, List<string> memberPath)
        {
            RootName = rootName;
            MemberPath = memberPath;
        }

        public void Assign(RuntimeContext context, object? value)
        {
            if (MemberPath.Count == 0)
            {
                context.SetInNearestScope(RootName, value);
                return;
            }

            object? target = context.Get(RootName);
            if (ReferenceEquals(target, Undefined.Value) || target is null)
                throw new InvalidOperationException($"Cannot assign to '{RootName}'.");

            object? current = target;
            for (int i = 0; i < MemberPath.Count - 1; i++)
            {
                current = GetMember(current, MemberPath[i]);
                if (ReferenceEquals(current, Undefined.Value) || current is null)
                    throw new InvalidOperationException($"Cannot assign to '{RootName}.{string.Join(".", MemberPath)}'.");
            }

            string leaf = MemberPath[^1];
            SetMember(current, leaf, value);
        }

        public static AssignmentTarget Parse(string text)
        {
            var parts = text.Split('.', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
                throw new InvalidOperationException($"Invalid assignment target '{text}'.");

            return new AssignmentTarget(parts[0], parts.Skip(1).ToList());
        }
    }

    // =========================================================
    // Template parser
    // =========================================================

    private sealed class TemplateParser
    {
        private readonly List<TemplateToken> _tokens;
        private int _pos;

        public TemplateParser(List<TemplateToken> tokens)
        {
            _tokens = tokens;
        }

        public List<TemplateNode> Parse()
        {
            return ParseUntil();
        }

        private List<TemplateNode> ParseUntil(params string[] stopStatements)
        {
            var nodes = new List<TemplateNode>();

            while (_pos < _tokens.Count)
            {
                var token = _tokens[_pos];

                if (token.Kind == TemplateTokenKind.Statement && stopStatements.Contains(Head(token.Value), StringComparer.Ordinal))
                    break;

                switch (token.Kind)
                {
                    case TemplateTokenKind.Text:
                        nodes.Add(new TextNode(token.Value));
                        _pos++;
                        break;

                    case TemplateTokenKind.Output:
                        nodes.Add(new OutputNode(ParseExpr(token.Value)));
                        _pos++;
                        break;

                    case TemplateTokenKind.Statement:
                        nodes.Add(ParseStatement(token.Value));
                        break;
                }
            }

            return nodes;
        }

        private TemplateNode ParseStatement(string statement)
        {
            string head = Head(statement);

            return head switch
            {
                "if" => ParseIf(),
                "for" => ParseFor(),
                "set" => ParseSet(),
                "macro" => ParseMacro(),
                _ => throw new InvalidOperationException($"Unsupported statement: {statement}")
            };
        }

        private TemplateNode ParseSet()
        {
            string statement = _tokens[_pos].Value;
            _pos++;

            string body = statement["set".Length..].Trim();
            int eq = body.IndexOf('=');
            if (eq < 0)
                throw new InvalidOperationException($"Invalid set statement: {statement}");

            string targetText = body[..eq].Trim();
            string exprText = body[(eq + 1)..].Trim();

            try
            {
                return new SetNode(AssignmentTarget.Parse(targetText), ParseExpr(exprText));
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Invalid set expression '{exprText}' in statement '{statement}'.", ex);
            }
        }

        private TemplateNode ParseFor()
        {
            string statement = _tokens[_pos].Value;
            _pos++;

            string body = statement["for".Length..].Trim();
            int inPos = IndexOfKeyword(body, "in");
            if (inPos < 0)
                throw new InvalidOperationException($"Invalid for statement: {statement}");

            string lhs = body[..inPos].Trim();
            string rhs = body[(inPos + 2)..].Trim();

            var variables = lhs.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToList();
            if (variables.Count == 0)
                throw new InvalidOperationException($"Invalid for statement: {statement}");

            Expr? filterExpr = null;
            int ifPos = IndexOfTopLevelKeyword(rhs, "if");
            if (ifPos >= 0)
            {
                filterExpr = ParseExpr(rhs[(ifPos + 2)..].Trim());
                rhs = rhs[..ifPos].Trim();
            }

            var loopBody = ParseUntil("endfor");

            ExpectStatement("endfor");
            _pos++;

            return new ForNode(variables, ParseExpr(rhs), loopBody, filterExpr);
        }

        private TemplateNode ParseIf()
        {
            var ifNode = new IfNode();

            while (true)
            {
                if (_pos >= _tokens.Count)
                    throw new InvalidOperationException("Unclosed if block.");

                string statement = _tokens[_pos].Value;
                string head = Head(statement);

                if (head == "if")
                {
                    string cond = statement["if".Length..].Trim();
                    _pos++;
                    ifNode.Branches.Add((ParseExpr(cond), ParseUntil("elif", "else", "endif")));
                }
                else if (head == "elif")
                {
                    string cond = statement["elif".Length..].Trim();
                    _pos++;
                    ifNode.Branches.Add((ParseExpr(cond), ParseUntil("elif", "else", "endif")));
                }
                else if (head == "else")
                {
                    _pos++;
                    ifNode.ElseBody = ParseUntil("endif");
                    ExpectStatement("endif");
                    _pos++;
                    return ifNode;
                }
                else if (head == "endif")
                {
                    _pos++;
                    return ifNode;
                }
                else
                {
                    throw new InvalidOperationException($"Unexpected statement in if block: {statement}");
                }
            }
        }

        private TemplateNode ParseMacro()
        {
            string statement = _tokens[_pos].Value;
            _pos++;

            string body = statement["macro".Length..].Trim();
            int open = body.IndexOf('(');
            int close = body.LastIndexOf(')');
            if (open < 0 || close < open)
                throw new InvalidOperationException($"Invalid macro statement: {statement}");

            string name = body[..open].Trim();
            string argsText = body[(open + 1)..close].Trim();

            var parameters = ParseMacroParameters(argsText);
            var macroBody = ParseUntil("endmacro");

            ExpectStatement("endmacro");
            _pos++;

            return new MacroNode(name, parameters, macroBody);
        }

        private List<MacroParameter> ParseMacroParameters(string text)
        {
            var parameters = new List<MacroParameter>();
            foreach (string part in SplitTopLevel(text, ','))
            {
                string p = part.Trim();
                if (p.Length == 0)
                    continue;

                int eq = FindTopLevelEquals(p);
                if (eq < 0)
                {
                    parameters.Add(new MacroParameter(p, null));
                }
                else
                {
                    string name = p[..eq].Trim();
                    string defaultExpr = p[(eq + 1)..].Trim();
                    parameters.Add(new MacroParameter(name, ParseExpr(defaultExpr)));
                }
            }
            return parameters;
        }

        private static int FindTopLevelEquals(string s)
        {
            int paren = 0;
            int bracket = 0;
            int brace = 0;
            char quote = '\0';

            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];

                if (quote != '\0')
                {
                    if (c == '\\' && i + 1 < s.Length)
                    {
                        i++;
                        continue;
                    }

                    if (c == quote)
                        quote = '\0';

                    continue;
                }

                switch (c)
                {
                    case '\'':
                    case '"':
                        quote = c;
                        break;
                    case '(':
                        paren++;
                        break;
                    case ')':
                        paren--;
                        break;
                    case '[':
                        bracket++;
                        break;
                    case ']':
                        bracket--;
                        break;
                    case '{':
                        brace++;
                        break;
                    case '}':
                        brace--;
                        break;
                    case '=':
                        if (paren == 0 && bracket == 0 && brace == 0)
                            return i;
                        break;
                }
            }

            return -1;
        }

        private static List<string> SplitTopLevel(string s, char separator)
        {
            var parts = new List<string>();
            int start = 0;
            int paren = 0;
            int bracket = 0;
            int brace = 0;
            char quote = '\0';

            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];

                if (quote != '\0')
                {
                    if (c == '\\' && i + 1 < s.Length)
                    {
                        i++;
                        continue;
                    }

                    if (c == quote)
                        quote = '\0';

                    continue;
                }

                switch (c)
                {
                    case '\'':
                    case '"':
                        quote = c;
                        break;
                    case '(':
                        paren++;
                        break;
                    case ')':
                        paren--;
                        break;
                    case '[':
                        bracket++;
                        break;
                    case ']':
                        bracket--;
                        break;
                    case '{':
                        brace++;
                        break;
                    case '}':
                        brace--;
                        break;
                    default:
                        if (c == separator && paren == 0 && bracket == 0 && brace == 0)
                        {
                            parts.Add(s[start..i]);
                            start = i + 1;
                        }
                        break;
                }
            }

            parts.Add(s[start..]);
            return parts;
        }

        private void ExpectStatement(string head)
        {
            if (_pos >= _tokens.Count
                || _tokens[_pos].Kind != TemplateTokenKind.Statement
                || Head(_tokens[_pos].Value) != head)
            {
                throw new InvalidOperationException($"Expected statement '{head}'.");
            }
        }

        private static string Head(string s)
        {
            int space = s.IndexOf(' ');
            return space < 0 ? s : s[..space];
        }

        private static int IndexOfKeyword(string s, string keyword)
        {
            for (int i = 0; i <= s.Length - keyword.Length; i++)
            {
                if (string.CompareOrdinal(s, i, keyword, 0, keyword.Length) != 0)
                    continue;

                bool leftOk = i == 0 || char.IsWhiteSpace(s[i - 1]);
                bool rightOk = i + keyword.Length == s.Length || char.IsWhiteSpace(s[i + keyword.Length]);

                if (leftOk && rightOk)
                    return i;
            }

            return -1;
        }

        private static int IndexOfTopLevelKeyword(string s, string keyword)
        {
            int depth = 0;
            bool inSingle = false;
            bool inDouble = false;

            for (int i = 0; i <= s.Length - keyword.Length; i++)
            {
                char c = s[i];

                if (inSingle)
                {
                    if (c == '\'' && (i == 0 || s[i - 1] != '\\')) inSingle = false;
                    continue;
                }
                if (inDouble)
                {
                    if (c == '"' && (i == 0 || s[i - 1] != '\\')) inDouble = false;
                    continue;
                }

                if (c == '\'') { inSingle = true; continue; }
                if (c == '"')  { inDouble = true; continue; }
                if (c is '(' or '[' or '{') { depth++; continue; }
                if (c is ')' or ']' or '}') { depth--; continue; }

                if (depth == 0 &&
                    string.CompareOrdinal(s, i, keyword, 0, keyword.Length) == 0 &&
                    (i == 0 || char.IsWhiteSpace(s[i - 1])) &&
                    (i + keyword.Length == s.Length || char.IsWhiteSpace(s[i + keyword.Length])))
                {
                    return i;
                }
            }

            return -1;
        }
    }

    // =========================================================
    // Expression lexer
    // =========================================================

    private enum ExprTokenKind
    {
        Identifier,
        String,
        Number,
        LParen,
        RParen,
        LBracket,
        RBracket,
        LBrace,
        RBrace,
        Dot,
        Comma,
        Colon,
        Pipe,
        Plus,
        Minus,
        Tilde,
        EqEq,
        NotEq,
        Lt,
        Lte,
        Gt,
        Gte,
        Assign,
        And,
        Or,
        Not,
        In,
        Is,
        If,
        Else,
        True,
        False,
        Null,
        Eof
    }

    private readonly record struct ExprToken(ExprTokenKind Kind, string Text);

    private sealed class ExprLexer
    {
        private readonly string _text;
        private int _pos;

        public ExprLexer(string text)
        {
            _text = text;
        }

        public List<ExprToken> Tokenize()
        {
            var tokens = new List<ExprToken>();

            while (true)
            {
                SkipWhitespace();

                if (_pos >= _text.Length)
                {
                    tokens.Add(new ExprToken(ExprTokenKind.Eof, ""));
                    return tokens;
                }

                char c = _text[_pos];

                switch (c)
                {
                    case '(':
                        tokens.Add(new ExprToken(ExprTokenKind.LParen, "("));
                        _pos++;
                        break;
                    case ')':
                        tokens.Add(new ExprToken(ExprTokenKind.RParen, ")"));
                        _pos++;
                        break;
                    case '[':
                        tokens.Add(new ExprToken(ExprTokenKind.LBracket, "["));
                        _pos++;
                        break;
                    case ']':
                        tokens.Add(new ExprToken(ExprTokenKind.RBracket, "]"));
                        _pos++;
                        break;
                    case '{':
                        tokens.Add(new ExprToken(ExprTokenKind.LBrace, "{"));
                        _pos++;
                        break;
                    case '}':
                        tokens.Add(new ExprToken(ExprTokenKind.RBrace, "}"));
                        _pos++;
                        break;
                    case '.':
                        tokens.Add(new ExprToken(ExprTokenKind.Dot, "."));
                        _pos++;
                        break;
                    case ',':
                        tokens.Add(new ExprToken(ExprTokenKind.Comma, ","));
                        _pos++;
                        break;
                    case ':':
                        tokens.Add(new ExprToken(ExprTokenKind.Colon, ":"));
                        _pos++;
                        break;
                    case '|':
                        tokens.Add(new ExprToken(ExprTokenKind.Pipe, "|"));
                        _pos++;
                        break;
                    case '+':
                        tokens.Add(new ExprToken(ExprTokenKind.Plus, "+"));
                        _pos++;
                        break;
                    case '-':
                        if (_pos + 1 < _text.Length && char.IsDigit(_text[_pos + 1]))
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.Number, ReadNumber()));
                        }
                        else
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.Minus, "-"));
                            _pos++;
                        }
                        break;
                    case '~':
                        tokens.Add(new ExprToken(ExprTokenKind.Tilde, "~"));
                        _pos++;
                        break;
                    case '=':
                        if (Peek("=="))
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.EqEq, "=="));
                            _pos += 2;
                        }
                        else
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.Assign, "="));
                            _pos++;
                        }
                        break;
                    case '!':
                        if (Peek("!="))
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.NotEq, "!="));
                            _pos += 2;
                        }
                        else
                        {
                            throw new InvalidOperationException("Unexpected '!'.");
                        }
                        break;
                    case '<':
                        if (Peek("<="))
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.Lte, "<="));
                            _pos += 2;
                        }
                        else
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.Lt, "<"));
                            _pos++;
                        }
                        break;
                    case '>':
                        if (Peek(">="))
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.Gte, ">="));
                            _pos += 2;
                        }
                        else
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.Gt, ">"));
                            _pos++;
                        }
                        break;
                    case '\'':
                    case '"':
                        tokens.Add(new ExprToken(ExprTokenKind.String, ReadString()));
                        break;
                    default:
                        if (char.IsDigit(c))
                        {
                            tokens.Add(new ExprToken(ExprTokenKind.Number, ReadNumber()));
                        }
                        else if (char.IsLetter(c) || c == '_')
                        {
                            string id = ReadIdentifier();
                            tokens.Add(id switch
                            {
                                "and" => new ExprToken(ExprTokenKind.And, id),
                                "or" => new ExprToken(ExprTokenKind.Or, id),
                                "not" => new ExprToken(ExprTokenKind.Not, id),
                                "in" => new ExprToken(ExprTokenKind.In, id),
                                "is" => new ExprToken(ExprTokenKind.Is, id),
                                "if" => new ExprToken(ExprTokenKind.If, id),
                                "else" => new ExprToken(ExprTokenKind.Else, id),
                                "true" => new ExprToken(ExprTokenKind.True, id),
                                "false" => new ExprToken(ExprTokenKind.False, id),
                                "none" => new ExprToken(ExprTokenKind.Null, id),
                                "null" => new ExprToken(ExprTokenKind.Null, id),
                                _ => new ExprToken(ExprTokenKind.Identifier, id)
                            });
                        }
                        else
                        {
                            throw new InvalidOperationException($"Unexpected character '{c}' in expression.");
                        }
                        break;
                }
            }
        }

        private void SkipWhitespace()
        {
            while (_pos < _text.Length && char.IsWhiteSpace(_text[_pos]))
                _pos++;
        }

        private bool Peek(string s)
        {
            return _pos + s.Length <= _text.Length
                && string.CompareOrdinal(_text, _pos, s, 0, s.Length) == 0;
        }

        private string ReadIdentifier()
        {
            int start = _pos;
            _pos++;
            while (_pos < _text.Length && (char.IsLetterOrDigit(_text[_pos]) || _text[_pos] == '_'))
                _pos++;
            return _text[start.._pos];
        }

        private string ReadNumber()
        {
            int start = _pos;
            if (_text[_pos] == '-')
                _pos++;

            while (_pos < _text.Length && char.IsDigit(_text[_pos]))
                _pos++;

            if (_pos < _text.Length && _text[_pos] == '.')
            {
                _pos++;
                while (_pos < _text.Length && char.IsDigit(_text[_pos]))
                    _pos++;
            }

            return _text[start.._pos];
        }

        private string ReadString()
        {
            char quote = _text[_pos++];
            var sb = new StringBuilder();

            while (_pos < _text.Length)
            {
                char c = _text[_pos++];
                if (c == quote)
                    return sb.ToString();

                if (c == '\\')
                {
                    if (_pos >= _text.Length)
                        throw new InvalidOperationException("Invalid string escape.");

                    char e = _text[_pos++];
                    sb.Append(e switch
                    {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        _ => e
                    });
                }
                else
                {
                    sb.Append(c);
                }
            }

            throw new InvalidOperationException("Unterminated string literal.");
        }
    }

    // =========================================================
    // Expressions
    // =========================================================

    private abstract class Expr
    {
        public abstract object? Evaluate(RuntimeContext context);
    }

    private sealed class LiteralExpr : Expr
    {
        private readonly object? _value;

        public LiteralExpr(object? value)
        {
            _value = value;
        }

        public override object? Evaluate(RuntimeContext context) => _value;
    }

    private sealed class VarExpr : Expr
    {
        private readonly string _name;

        public VarExpr(string name)
        {
            _name = name;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            return context.Get(_name);
        }
    }

    private sealed class MemberExpr : Expr
    {
        private readonly Expr _target;
        private readonly string _member;

        public MemberExpr(Expr target, string member)
        {
            _target = target;
            _member = member;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            object? target = _target.Evaluate(context);
            return GetMember(target, _member);
        }
    }

    private sealed class IndexExpr : Expr
    {
        private readonly Expr _target;
        private readonly Expr _index;

        public IndexExpr(Expr target, Expr index)
        {
            _target = target;
            _index = index;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            object? target = _target.Evaluate(context);
            object? index = _index.Evaluate(context);
            return GetIndex(target, index);
        }
    }

    private sealed class SliceExpr : Expr
    {
        private readonly Expr _target;
        private readonly Expr? _start;
        private readonly Expr? _end;
        private readonly Expr? _step;

        public SliceExpr(Expr target, Expr? start, Expr? end, Expr? step)
        {
            _target = target;
            _start = start;
            _end = end;
            _step = step;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            object? target = _target.Evaluate(context);

            int? start = ToNullableInt(_start?.Evaluate(context));
            int? end = ToNullableInt(_end?.Evaluate(context));
            int? step = ToNullableInt(_step?.Evaluate(context));

            return ApplySlice(target, start, end, step);
        }
    }

    private sealed class CallArg
    {
        public string? Name { get; }
        public Expr Value { get; }

        public CallArg(string? name, Expr value)
        {
            Name = name;
            Value = value;
        }
    }

    private sealed class CallExpr : Expr
    {
        private readonly Expr _callee;
        private readonly List<CallArg> _args;

        public CallExpr(Expr callee, List<CallArg> args)
        {
            _callee = callee;
            _args = args;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            object? callee = _callee.Evaluate(context);

            var positional = new List<object?>();
            var named = new Dictionary<string, object?>(StringComparer.Ordinal);

            foreach (var arg in _args)
            {
                object? value = arg.Value.Evaluate(context);
                if (arg.Name is null)
                    positional.Add(value);
                else
                    named[arg.Name] = value;
            }

            return InvokeCallable(callee, context, positional, named);
        }
    }

    private sealed class ListExpr : Expr
    {
        private readonly List<Expr> _items;

        public ListExpr(List<Expr> items)
        {
            _items = items;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            var list = new List<object?>(_items.Count);
            foreach (var item in _items)
                list.Add(item.Evaluate(context));
            return list;
        }
    }

    private sealed class DictExpr : Expr
    {
        private readonly List<(Expr Key, Expr Value)> _items;

        public DictExpr(List<(Expr Key, Expr Value)> items)
        {
            _items = items;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            var dict = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var (k, v) in _items)
                dict[Stringify(k.Evaluate(context))] = v.Evaluate(context);
            return dict;
        }
    }

    private sealed class UnaryExpr : Expr
    {
        private readonly string _op;
        private readonly Expr _expr;

        public UnaryExpr(string op, Expr expr)
        {
            _op = op;
            _expr = expr;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            object? value = _expr.Evaluate(context);

            return _op switch
            {
                "not" => !IsTruthy(value),
                "-" => Negate(value),
                _ => throw new InvalidOperationException($"Unsupported unary operator '{_op}'.")
            };
        }
    }

    private sealed class BinaryExpr : Expr
    {
        private readonly Expr _left;
        private readonly string _op;
        private readonly Expr _right;

        public BinaryExpr(Expr left, string op, Expr right)
        {
            _left = left;
            _op = op;
            _right = right;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            if (_op == "and")
            {
                object? l = _left.Evaluate(context);
                return IsTruthy(l) ? _right.Evaluate(context) : l;
            }

            if (_op == "or")
            {
                object? l = _left.Evaluate(context);
                return IsTruthy(l) ? l : _right.Evaluate(context);
            }

            object? left = _left.Evaluate(context);
            object? right = _right.Evaluate(context);

            return _op switch
            {
                "==" => EqualsNormalized(left, right),
                "!=" => !EqualsNormalized(left, right),
                "<" => Compare(left, right) < 0,
                "<=" => Compare(left, right) <= 0,
                ">" => Compare(left, right) > 0,
                ">=" => Compare(left, right) >= 0,
                "in" => Contains(right, left),
                "+" => Add(left, right),
                "-" => Subtract(left, right),
                "~" => Stringify(left) + Stringify(right),
                _ => throw new InvalidOperationException($"Unsupported operator '{_op}'.")
            };
        }
    }

    private sealed class TestExpr : Expr
    {
        private readonly Expr _value;
        private readonly string _testName;
        private readonly bool _negated;

        public TestExpr(Expr value, string testName, bool negated)
        {
            _value = value;
            _testName = testName;
            _negated = negated;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            bool result = ApplyTest(_testName, _value.Evaluate(context));
            return _negated ? !result : result;
        }
    }

    private sealed class ConditionalExpr : Expr
    {
        private readonly Expr _whenTrue;
        private readonly Expr _condition;
        private readonly Expr _whenFalse;

        public ConditionalExpr(Expr whenTrue, Expr condition, Expr whenFalse)
        {
            _whenTrue = whenTrue;
            _condition = condition;
            _whenFalse = whenFalse;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            return IsTruthy(_condition.Evaluate(context))
                ? _whenTrue.Evaluate(context)
                : _whenFalse.Evaluate(context);
        }
    }

    private sealed class FilterExpr : Expr
    {
        private readonly Expr _value;
        private readonly string _filterName;
        private readonly List<Expr> _args;

        public FilterExpr(Expr value, string filterName, List<Expr> args)
        {
            _value = value;
            _filterName = filterName;
            _args = args;
        }

        public override object? Evaluate(RuntimeContext context)
        {
            object? value = _value.Evaluate(context);
            var args = _args.Select(a => a.Evaluate(context)).ToList();
            return ApplyFilter(_filterName, value, args);
        }
    }

    private sealed class ExprParser
    {
        private readonly List<ExprToken> _tokens;
        private int _pos;

        public ExprParser(List<ExprToken> tokens)
        {
            _tokens = tokens;
        }

        public Expr ParseExpression()
        {
            return ParseConditional();
        }

        private Expr ParseConditional()
        {
            Expr expr = ParseOr();

            if (Match(ExprTokenKind.If))
            {
                Expr condition = ParseExpression();
                Expr whenFalse = Match(ExprTokenKind.Else)
                    ? ParseExpression()
                    : new LiteralExpr(Undefined.Value);
                return new ConditionalExpr(expr, condition, whenFalse);
            }

            return expr;
        }

        private Expr ParseFilter()
        {
            Expr expr = ParsePostfix();

            while (Match(ExprTokenKind.Pipe))
            {
                string name = Expect(ExprTokenKind.Identifier).Text;
                var args = new List<Expr>();

                if (Match(ExprTokenKind.LParen))
                {
                    if (!Check(ExprTokenKind.RParen))
                    {
                        do
                        {
                            args.Add(ParseExpression());
                        }
                        while (Match(ExprTokenKind.Comma));
                    }

                    Expect(ExprTokenKind.RParen);
                }

                expr = new FilterExpr(expr, name, args);
            }

            return expr;
        }

        private Expr ParseOr()
        {
            Expr expr = ParseAnd();
            while (Match(ExprTokenKind.Or))
                expr = new BinaryExpr(expr, "or", ParseAnd());
            return expr;
        }

        private Expr ParseAnd()
        {
            Expr expr = ParseCompare();
            while (Match(ExprTokenKind.And))
                expr = new BinaryExpr(expr, "and", ParseCompare());
            return expr;
        }

        private Expr ParseCompare()
        {
            Expr expr = ParseAdditive();

            while (true)
            {
                if (Match(ExprTokenKind.Is))
                {
                    bool negated = Match(ExprTokenKind.Not);
                    string testName = ParseTestName();
                    expr = new TestExpr(expr, testName, negated);
                }
                else if (Match(ExprTokenKind.EqEq))
                {
                    expr = new BinaryExpr(expr, "==", ParseAdditive());
                }
                else if (Match(ExprTokenKind.NotEq))
                {
                    expr = new BinaryExpr(expr, "!=", ParseAdditive());
                }
                else if (Match(ExprTokenKind.Lte))
                {
                    expr = new BinaryExpr(expr, "<=", ParseAdditive());
                }
                else if (Match(ExprTokenKind.Gte))
                {
                    expr = new BinaryExpr(expr, ">=", ParseAdditive());
                }
                else if (Match(ExprTokenKind.Lt))
                {
                    expr = new BinaryExpr(expr, "<", ParseAdditive());
                }
                else if (Match(ExprTokenKind.Gt))
                {
                    expr = new BinaryExpr(expr, ">", ParseAdditive());
                }
                else if (Match(ExprTokenKind.In))
                {
                    expr = new BinaryExpr(expr, "in", ParseAdditive());
                }
                else
                {
                    break;
                }
            }

            return expr;
        }

        private string ParseTestName()
        {
            if (Match(ExprTokenKind.Identifier, out var identifier))
                return identifier.Text;

            if (Match(ExprTokenKind.Null))
                return "none";

            if (Match(ExprTokenKind.True))
                return "true";

            if (Match(ExprTokenKind.False))
                return "false";

            throw new InvalidOperationException($"Expected Identifier, got {Peek().Kind}.");
        }

        private Expr ParseAdditive()
        {
            Expr expr = ParseUnary();

            while (true)
            {
                if (Match(ExprTokenKind.Plus))
                    expr = new BinaryExpr(expr, "+", ParseUnary());
                else if (Match(ExprTokenKind.Minus))
                    expr = new BinaryExpr(expr, "-", ParseUnary());
                else if (Match(ExprTokenKind.Tilde))
                    expr = new BinaryExpr(expr, "~", ParseUnary());
                else
                    break;
            }

            return expr;
        }

        private Expr ParseUnary()
        {
            if (Match(ExprTokenKind.Not))
                return new UnaryExpr("not", ParseUnary());

            if (Match(ExprTokenKind.Minus))
                return new UnaryExpr("-", ParseUnary());

            return ParseFilter();
        }

        private Expr ParsePostfix()
        {
            Expr expr = ParsePrimary();

            while (true)
            {
                if (Match(ExprTokenKind.Dot))
                {
                    string member = Expect(ExprTokenKind.Identifier).Text;
                    expr = new MemberExpr(expr, member);
                }
                else if (Match(ExprTokenKind.LBracket))
                {
                    expr = ParseBracketSuffix(expr);
                }
                else if (Match(ExprTokenKind.LParen))
                {
                    expr = ParseCallSuffix(expr);
                }
                else
                {
                    break;
                }
            }

            return expr;
        }

        private Expr ParseBracketSuffix(Expr target)
        {
            if (Match(ExprTokenKind.Colon))
            {
                Expr? end = null;
                Expr? step = null;

                if (!Check(ExprTokenKind.Colon) && !Check(ExprTokenKind.RBracket))
                    end = ParseExpression();

                if (Match(ExprTokenKind.Colon))
                {
                    if (!Check(ExprTokenKind.RBracket))
                        step = ParseExpression();
                }

                Expect(ExprTokenKind.RBracket);
                return new SliceExpr(target, null, end, step);
            }

            Expr first = ParseExpression();

            if (Match(ExprTokenKind.Colon))
            {
                Expr? end = null;
                Expr? step = null;

                if (!Check(ExprTokenKind.Colon) && !Check(ExprTokenKind.RBracket))
                    end = ParseExpression();

                if (Match(ExprTokenKind.Colon))
                {
                    if (!Check(ExprTokenKind.RBracket))
                        step = ParseExpression();
                }

                Expect(ExprTokenKind.RBracket);
                return new SliceExpr(target, first, end, step);
            }

            Expect(ExprTokenKind.RBracket);
            return new IndexExpr(target, first);
        }

        private Expr ParseCallSuffix(Expr callee)
        {
            var args = new List<CallArg>();

            if (!Check(ExprTokenKind.RParen))
            {
                do
                {
                    if (Check(ExprTokenKind.Identifier) && PeekNext().Kind == ExprTokenKind.Assign)
                    {
                        string name = Expect(ExprTokenKind.Identifier).Text;
                        Expect(ExprTokenKind.Assign);
                        args.Add(new CallArg(name, ParseExpression()));
                    }
                    else
                    {
                        args.Add(new CallArg(null, ParseExpression()));
                    }
                }
                while (Match(ExprTokenKind.Comma));
            }

            Expect(ExprTokenKind.RParen);
            return new CallExpr(callee, args);
        }

        private Expr ParsePrimary()
        {
            if (Match(ExprTokenKind.String, out var s))
                return new LiteralExpr(s.Text);

            if (Match(ExprTokenKind.Number, out var n))
            {
                if (n.Text.Contains('.', StringComparison.Ordinal))
                    return new LiteralExpr(double.Parse(n.Text, CultureInfo.InvariantCulture));
                return new LiteralExpr(long.Parse(n.Text, CultureInfo.InvariantCulture));
            }

            if (Match(ExprTokenKind.True))
                return new LiteralExpr(true);

            if (Match(ExprTokenKind.False))
                return new LiteralExpr(false);

            if (Match(ExprTokenKind.Null))
                return new LiteralExpr(null);

            if (Match(ExprTokenKind.Identifier, out var id))
                return new VarExpr(id.Text);

            if (Match(ExprTokenKind.LParen))
            {
                Expr expr = ParseExpression();
                Expect(ExprTokenKind.RParen);
                return expr;
            }

            if (Match(ExprTokenKind.LBracket))
            {
                var items = new List<Expr>();
                if (!Check(ExprTokenKind.RBracket))
                {
                    do
                    {
                        items.Add(ParseExpression());
                    }
                    while (Match(ExprTokenKind.Comma));
                }
                Expect(ExprTokenKind.RBracket);
                return new ListExpr(items);
            }

            if (Match(ExprTokenKind.LBrace))
            {
                var items = new List<(Expr Key, Expr Value)>();
                if (!Check(ExprTokenKind.RBrace))
                {
                    do
                    {
                        Expr key = ParseExpression();
                        Expect(ExprTokenKind.Colon);
                        Expr value = ParseExpression();
                        items.Add((key, value));
                    }
                    while (Match(ExprTokenKind.Comma));
                }
                Expect(ExprTokenKind.RBrace);
                return new DictExpr(items);
            }

            throw new InvalidOperationException($"Unexpected token: {Peek().Kind}");
        }

        private ExprToken Peek() => _tokens[_pos];
        private ExprToken PeekNext() => _pos + 1 < _tokens.Count ? _tokens[_pos + 1] : _tokens[^1];

        private bool Check(ExprTokenKind kind) => Peek().Kind == kind;

        private bool Match(ExprTokenKind kind)
        {
            if (Check(kind))
            {
                _pos++;
                return true;
            }

            return false;
        }

        private bool Match(ExprTokenKind kind, out ExprToken token)
        {
            token = Peek();
            if (token.Kind == kind)
            {
                _pos++;
                return true;
            }

            return false;
        }

        private ExprToken Expect(ExprTokenKind kind)
        {
            var token = Peek();
            if (token.Kind != kind)
                throw new InvalidOperationException($"Expected {kind}, got {token.Kind}.");
            _pos++;
            return token;
        }
    }

    private static Expr ParseExpr(string text)
    {
        var lexer = new ExprLexer(text);
        var tokens = lexer.Tokenize();
        var parser = new ExprParser(tokens);
        return parser.ParseExpression();
    }

    // =========================================================
    // Runtime
    // =========================================================

    private sealed class RuntimeContext
    {
        private readonly Stack<Dictionary<string, object?>> _scopes = new();

        public RuntimeContext(IDictionary<string, object?> globals)
        {
            var global = new Dictionary<string, object?>(StringComparer.Ordinal);

            foreach (var kv in globals)
                global[kv.Key] = NormalizeInput(kv.Value);

            global["namespace"] = new BuiltinCallable("namespace", (_, pos, named) =>
            {
                var ns = new NamespaceValue();
                for (int i = 0; i < pos.Count; i++)
                    ns.Set($"arg{i}", pos[i]);

                foreach (var kv in named)
                    ns.Set(kv.Key, kv.Value);

                return ns;
            });

            global["raise_exception"] = new BuiltinCallable("raise_exception", (_, pos, _) =>
            {
                string message = pos.Count > 0 ? Stringify(pos[0]) : "Template raised an exception.";
                throw new InvalidOperationException(message);
            });

            _scopes.Push(global);
        }

        public void PushScope()
        {
            _scopes.Push(new Dictionary<string, object?>(StringComparer.Ordinal));
        }

        public void PopScope()
        {
            _scopes.Pop();
        }

        public void Set(string name, object? value)
        {
            _scopes.Peek()[name] = NormalizeInput(value);
        }

        public void SetInNearestScope(string name, object? value)
        {
            foreach (var scope in _scopes)
            {
                if (scope.ContainsKey(name))
                {
                    scope[name] = NormalizeInput(value);
                    return;
                }
            }

            Set(name, value);
        }

        public object? Get(string name)
        {
            foreach (var scope in _scopes)
            {
                if (scope.TryGetValue(name, out var value))
                    return value;
            }

            return Undefined.Value;
        }
    }

    private interface ICallable
    {
        object? Invoke(RuntimeContext context, List<object?> positional, Dictionary<string, object?> named);
    }

    private sealed class BuiltinCallable : ICallable
    {
        private readonly string _name;
        private readonly Func<RuntimeContext, List<object?>, Dictionary<string, object?>, object?> _impl;

        public BuiltinCallable(
            string name,
            Func<RuntimeContext, List<object?>, Dictionary<string, object?>, object?> impl)
        {
            _name = name;
            _impl = impl;
        }

        public object? Invoke(RuntimeContext context, List<object?> positional, Dictionary<string, object?> named)
        {
            return _impl(context, positional, named);
        }

        public override string ToString() => $"<builtin {_name}>";
    }

    private sealed class BoundMethod : ICallable
    {
        private readonly object? _target;
        private readonly string _name;

        public BoundMethod(object? target, string name)
        {
            _target = target;
            _name = name;
        }

        public object? Invoke(RuntimeContext context, List<object?> positional, Dictionary<string, object?> named)
        {
            if (named.Count != 0)
                throw new InvalidOperationException($"Method '{_name}' does not support named arguments.");

            if (_target is IDictionary<string, object?> dict)
            {
                return _name switch
                {
                    "get" => DictGet(dict, positional),
                    "items" => dict.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList(),
                    "keys" => dict.Keys.Cast<object?>().ToList(),
                    "values" => dict.Values.ToList(),
                    _ => throw new InvalidOperationException($"Unsupported dict method '{_name}'.")
                };
            }

            string s = Stringify(_target);

            return _name switch
            {
                "startswith" => positional.Count >= 1 && s.StartsWith(Stringify(positional[0]), StringComparison.Ordinal),
                "endswith" => positional.Count >= 1 && s.EndsWith(Stringify(positional[0]), StringComparison.Ordinal),
                "split" => SplitMethod(s, positional),
                "rstrip" => TrimRightMethod(s, positional),
                "lstrip" => TrimLeftMethod(s, positional),
                "strip" => TrimMethod(s, positional),
                _ => throw new InvalidOperationException($"Unsupported method '{_name}'.")
            };
        }

        private static object SplitMethod(string s, List<object?> positional)
        {
            if (positional.Count == 0)
                return s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Cast<object?>().ToList();

            string sep = Stringify(positional[0]);
            return s.Split([sep], StringSplitOptions.None).Cast<object?>().ToList();
        }

        private static object TrimRightMethod(string s, List<object?> positional)
        {
            if (positional.Count == 0)
                return s.TrimEnd();

            string chars = Stringify(positional[0]);
            return s.TrimEnd(chars.ToCharArray());
        }

        private static object TrimLeftMethod(string s, List<object?> positional)
        {
            if (positional.Count == 0)
                return s.TrimStart();

            string chars = Stringify(positional[0]);
            return s.TrimStart(chars.ToCharArray());
        }

        private static object TrimMethod(string s, List<object?> positional)
        {
            if (positional.Count == 0)
                return s.Trim();

            string chars = Stringify(positional[0]);
            return s.Trim(chars.ToCharArray());
        }

        private static object? DictGet(IDictionary<string, object?> dict, List<object?> args)
        {
            if (args.Count == 0)
                throw new InvalidOperationException("dict.get() requires at least one argument.");

            string key = Stringify(args[0]);
            if (dict.TryGetValue(key, out var val))
                return val;

            return args.Count > 1 ? args[1] : Undefined.Value;
        }

        public override string ToString() => $"<method {_name}>";
    }

    private sealed class MacroValue : ICallable
    {
        private readonly string _name;
        private readonly List<MacroParameter> _parameters;
        private readonly List<TemplateNode> _body;
        private readonly RuntimeContext _ownerContext;

        public MacroValue(
            string name,
            List<MacroParameter> parameters,
            List<TemplateNode> body,
            RuntimeContext ownerContext)
        {
            _name = name;
            _parameters = parameters;
            _body = body;
            _ownerContext = ownerContext;
        }

        public object? Invoke(RuntimeContext context, List<object?> positional, Dictionary<string, object?> named)
        {
            _ownerContext.PushScope();

            try
            {
                for (int i = 0; i < _parameters.Count; i++)
                {
                    var p = _parameters[i];

                    object? value;
                    if (i < positional.Count)
                    {
                        value = positional[i];
                    }
                    else if (named.TryGetValue(p.Name, out var namedValue))
                    {
                        value = namedValue;
                    }
                    else if (p.DefaultValue is not null)
                    {
                        value = p.DefaultValue.Evaluate(_ownerContext);
                    }
                    else
                    {
                        value = Undefined.Value;
                    }

                    _ownerContext.Set(p.Name, value);
                }

                var sb = new StringBuilder();
                foreach (var node in _body)
                    node.Render(_ownerContext, sb);

                return sb.ToString();
            }
            finally
            {
                _ownerContext.PopScope();
            }
        }

        public override string ToString() => $"<macro {_name}>";
    }

    private sealed class NamespaceValue
    {
        private readonly Dictionary<string, object?> _values = new(StringComparer.Ordinal);

        public object? Get(string name)
        {
            return _values.TryGetValue(name, out var value) ? value : Undefined.Value;
        }

        public void Set(string name, object? value)
        {
            _values[name] = NormalizeInput(value);
        }
    }

    private sealed class Undefined
    {
        public static readonly Undefined Value = new();
        private Undefined() { }
        public override string ToString() => "";
    }

    // =========================================================
    // Helpers
    // =========================================================

    private static object? NormalizeInput(object? value)
    {
        if (value is null)
            return null;

        if (ReferenceEquals(value, Undefined.Value))
            return value;

        if (value is JsonElement e)
            return NormalizeJsonElement(e);

        if (value is IDictionary<string, object?>)
            return value;

        if (value is IDictionary<string, object> dictObj)
        {
            var dict = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var kv in dictObj)
                dict[kv.Key] = NormalizeInput(kv.Value);
            return dict;
        }

        if (value is IEnumerable enumerable && value is not string)
        {
            if (value is IList || value is object?[])
            {
                var list = new List<object?>();
                foreach (var item in enumerable)
                    list.Add(NormalizeInput(item));
                return list;
            }
        }

        return value;
    }

    private static object? NormalizeJsonElement(JsonElement e)
    {
        return e.ValueKind switch
        {
            JsonValueKind.Object => e.EnumerateObject()
                .ToDictionary(
                    p => p.Name,
                    p => NormalizeJsonElement(p.Value),
                    StringComparer.Ordinal),
            JsonValueKind.Array => e.EnumerateArray()
                .Select(NormalizeJsonElement)
                .ToList(),
            JsonValueKind.String => e.GetString(),
            JsonValueKind.Number => e.TryGetInt64(out long l) ? l : e.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => null,
            JsonValueKind.Undefined => Undefined.Value,
            _ => e.GetRawText()
        };
    }

    private static object? GetMember(object? obj, string member)
    {
        if (obj is null)
            return Undefined.Value;

        if (ReferenceEquals(obj, Undefined.Value))
            return Undefined.Value;

        if (obj is NamespaceValue ns)
            return ns.Get(member);

        if (obj is IDictionary<string, object?> dict && dict.TryGetValue(member, out var value))
            return value;

        if (obj is IDictionary<string, object> dict2 && dict2.TryGetValue(member, out var value2))
            return value2;

        if (obj is string && IsSupportedStringMethod(member))
            return new BoundMethod(obj, member);

        if ((obj is IDictionary<string, object?> || obj is IDictionary<string, object>) && IsSupportedDictMethod(member))
            return new BoundMethod(obj, member);

        var prop = obj.GetType().GetProperty(member);
        if (prop is not null)
            return prop.GetValue(obj) ?? Undefined.Value;

        var field = obj.GetType().GetField(member);
        if (field is not null)
            return field.GetValue(obj) ?? Undefined.Value;

        return Undefined.Value;
    }

    private static void SetMember(object? obj, string member, object? value)
    {
        if (obj is NamespaceValue ns)
        {
            ns.Set(member, value);
            return;
        }

        if (obj is IDictionary<string, object?> dict)
        {
            dict[member] = NormalizeInput(value);
            return;
        }

        var prop = obj?.GetType().GetProperty(member);
        if (prop is not null && prop.CanWrite)
        {
            prop.SetValue(obj, value);
            return;
        }

        var field = obj?.GetType().GetField(member);
        if (field is not null)
        {
            field.SetValue(obj, value);
            return;
        }

        throw new InvalidOperationException($"Cannot assign member '{member}'.");
    }

    private static object? GetIndex(object? obj, object? index)
    {
        if (obj is null || ReferenceEquals(obj, Undefined.Value) || index is null || ReferenceEquals(index, Undefined.Value))
            return Undefined.Value;

        if (obj is IList list && index is IConvertible)
        {
            int i = Convert.ToInt32(index, CultureInfo.InvariantCulture);
            if (i < 0)
                i = list.Count + i;

            return i >= 0 && i < list.Count ? list[i] : Undefined.Value;
        }

        if (obj is string s && index is IConvertible)
        {
            int i = Convert.ToInt32(index, CultureInfo.InvariantCulture);
            if (i < 0)
                i = s.Length + i;

            return i >= 0 && i < s.Length ? s[i].ToString() : Undefined.Value;
        }

        if (obj is IDictionary<string, object?> dict)
        {
            string key = Stringify(index);
            return dict.TryGetValue(key, out var value) ? value : Undefined.Value;
        }

        if (obj is IDictionary<string, object> dict2)
        {
            string key = Stringify(index);
            return dict2.TryGetValue(key, out var value2) ? value2 : Undefined.Value;
        }

        return Undefined.Value;
    }

    private static object? ApplySlice(object? target, int? start, int? end, int? step)
    {
        int st = step ?? 1;
        if (st == 0)
            throw new InvalidOperationException("Slice step cannot be zero.");

        if (target is string s)
        {
            var chars = s.Select(c => (object?)c.ToString()).ToList();
            return string.Concat(ApplySliceToList(chars, start, end, st).Select(Stringify));
        }

        if (target is IEnumerable enumerable && target is not string)
        {
            var list = enumerable.Cast<object?>().ToList();
            return ApplySliceToList(list, start, end, st);
        }

        return Undefined.Value;
    }

    private static List<object?> ApplySliceToList(List<object?> list, int? start, int? end, int step)
    {
        int count = list.Count;

        int actualStart;
        int actualEnd;

        if (step > 0)
        {
            actualStart = start ?? 0;
            actualEnd = end ?? count;
            if (actualStart < 0) actualStart += count;
            if (actualEnd < 0) actualEnd += count;
            actualStart = Math.Clamp(actualStart, 0, count);
            actualEnd = Math.Clamp(actualEnd, 0, count);

            var result = new List<object?>();
            for (int i = actualStart; i < actualEnd; i += step)
                result.Add(list[i]);
            return result;
        }
        else
        {
            actualStart = start ?? (count - 1);
            actualEnd = end ?? -1;
            if (start.HasValue && actualStart < 0) actualStart += count;
            if (end.HasValue && actualEnd < 0) actualEnd += count;
            actualStart = Math.Clamp(actualStart, -1, count - 1);
            actualEnd = Math.Clamp(actualEnd, -1, count - 1);

            var result = new List<object?>();
            for (int i = actualStart; i > actualEnd; i += step)
            {
                if (i >= 0 && i < count)
                    result.Add(list[i]);
            }
            return result;
        }
    }

    private static int? ToNullableInt(object? value)
    {
        if (value is null || ReferenceEquals(value, Undefined.Value))
            return null;

        return value switch
        {
            int i => i,
            long l => checked((int)l),
            short s => s,
            byte b => b,
            double d => checked((int)d),
            float f => checked((int)f),
            decimal m => checked((int)m),
            _ => int.TryParse(Stringify(value), NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
                ? parsed
                : null
        };
    }

    private static object? InvokeCallable(
        object? callee,
        RuntimeContext context,
        List<object?> positional,
        Dictionary<string, object?> named)
    {
        if (callee is ICallable callable)
            return callable.Invoke(context, positional, named);

        throw new InvalidOperationException($"Object '{Stringify(callee)}' is not callable.");
    }

    private static bool ApplyTest(string name, object? value)
    {
        return name switch
        {
            "true" => value is true,
            "false" => value is false,
            "boolean" => value is bool,
            "string" => value is string,
            "number" => value is sbyte or byte or short or ushort or int or uint or long or ulong or float or double or decimal,
            "integer" => value is sbyte or byte or short or ushort or int or uint or long or ulong,
            "float" => value is float or double,
            "mapping" => value is IDictionary || value is NamespaceValue,
            "iterable" => value is IEnumerable && value is not string,
            "sequence" => (value is IEnumerable && value is not string && value is not IDictionary && value is not NamespaceValue),
            "defined" => !ReferenceEquals(value, Undefined.Value),
            "undefined" => ReferenceEquals(value, Undefined.Value),
            "none" => value is null,
            _ => throw new InvalidOperationException($"Unsupported test '{name}'.")
        };
    }

    private static bool IsSupportedStringMethod(string name)
    {
        return name is "startswith" or "endswith" or "split" or "rstrip" or "lstrip" or "strip";
    }

    private static bool IsSupportedDictMethod(string name)
    {
        return name is "get" or "items" or "keys" or "values";
    }

    private static bool IsTruthy(object? value)
    {
        return value switch
        {
            null => false,
            Undefined => false,
            bool b => b,
            string s => s.Length > 0,
            ICollection c => c.Count > 0,
            IEnumerable e when value is not string => e.Cast<object?>().Any(),
            sbyte v => v != 0,
            byte v => v != 0,
            short v => v != 0,
            ushort v => v != 0,
            int v => v != 0,
            uint v => v != 0,
            long v => v != 0,
            ulong v => v != 0,
            float v => Math.Abs(v) > float.Epsilon,
            double v => Math.Abs(v) > double.Epsilon,
            decimal v => v != 0m,
            _ => true
        };
    }

    private static bool EqualsNormalized(object? a, object? b)
    {
        if (ReferenceEquals(a, Undefined.Value))
            a = null;
        if (ReferenceEquals(b, Undefined.Value))
            b = null;

        if (a is null && b is null)
            return true;
        if (a is null || b is null)
            return false;

        if (TryNumber(a, out double da) && TryNumber(b, out double db))
            return Math.Abs(da - db) < 1e-12;

        return string.Equals(Stringify(a), Stringify(b), StringComparison.Ordinal);
    }

    private static int Compare(object? a, object? b)
    {
        if (TryNumber(a, out double da) && TryNumber(b, out double db))
            return da.CompareTo(db);

        return string.CompareOrdinal(Stringify(a), Stringify(b));
    }

    private static bool Contains(object? container, object? item)
    {
        if (container is null || ReferenceEquals(container, Undefined.Value))
            return false;

        if (container is string s)
            return s.Contains(Stringify(item), StringComparison.Ordinal);

        if (container is IDictionary<string, object?> dict)
            return dict.ContainsKey(Stringify(item));

        if (container is IDictionary dictObj)
            return dictObj.Contains(Stringify(item));

        if (container is IEnumerable e)
        {
            foreach (var x in e)
            {
                if (EqualsNormalized(x, item))
                    return true;
            }
        }

        return false;
    }

    private static object? Add(object? a, object? b)
    {
        if (TryNumber(a, out double da) && TryNumber(b, out double db))
        {
            if (a is long or int or short or byte && b is long or int or short or byte)
                return Convert.ToInt64(da + db, CultureInfo.InvariantCulture);

            return da + db;
        }

        if (a is IList la && b is IList lb)
        {
            var result = new List<object?>();
            foreach (var x in la) result.Add(x);
            foreach (var x in lb) result.Add(x);
            return result;
        }

        return Stringify(a) + Stringify(b);
    }

    private static object? Subtract(object? a, object? b)
    {
        if (TryNumber(a, out double da) && TryNumber(b, out double db))
        {
            if (a is long or int or short or byte && b is long or int or short or byte)
                return Convert.ToInt64(da - db, CultureInfo.InvariantCulture);

            return da - db;
        }

        throw new InvalidOperationException("Subtraction is only supported for numbers.");
    }

    private static object? Negate(object? value)
    {
        if (TryNumber(value, out double d))
        {
            if (value is long or int or short or byte)
                return Convert.ToInt64(-d, CultureInfo.InvariantCulture);

            return -d;
        }

        throw new InvalidOperationException("Unary minus is only supported for numbers.");
    }

    private static bool TryNumber(object? value, out double number)
    {
        switch (value)
        {
            case byte v: number = v; return true;
            case sbyte v: number = v; return true;
            case short v: number = v; return true;
            case ushort v: number = v; return true;
            case int v: number = v; return true;
            case uint v: number = v; return true;
            case long v: number = v; return true;
            case ulong v: number = v; return true;
            case float v: number = v; return true;
            case double v: number = v; return true;
            case decimal v: number = (double)v; return true;
            default:
                number = 0;
                return false;
        }
    }

    private static object? ApplyFilter(string name, object? value, List<object?> args)
    {
        return name switch
        {
            "trim" => Stringify(value).Trim(),
            "lower" => Stringify(value).ToLowerInvariant(),
            "upper" => Stringify(value).ToUpperInvariant(),
            "length" => GetLength(value),
            "join" => JoinFilter(value, args),
            "default" => DefaultFilter(value, args),
            "tojson" => JsonSerializer.Serialize(value is Undefined ? null : value),
            "safe" => value,
            "string" => Stringify(value),
            "items" => ItemsFilter(value),
            "dictsort" => DictSortFilter(value),
            _ => throw new InvalidOperationException($"Unsupported filter '{name}'.")
        };
    }

    private static object GetLength(object? value)
    {
        return value switch
        {
            null => 0,
            Undefined => 0,
            string s => s.Length,
            ICollection c => c.Count,
            IEnumerable e when value is not string => e.Cast<object?>().Count(),
            _ => Stringify(value).Length
        };
    }

    private static object JoinFilter(object? value, List<object?> args)
    {
        string sep = args.Count > 0 ? Stringify(args[0]) : "";

        if (value is IEnumerable e && value is not string)
            return string.Join(sep, e.Cast<object?>().Select(Stringify));

        return Stringify(value);
    }

    private static object? DefaultFilter(object? value, List<object?> args)
    {
        return IsTruthy(value) ? value : (args.Count > 0 ? args[0] : null);
    }

    private static object ItemsFilter(object? value)
    {
        if (value is IDictionary<string, object?> dict)
            return dict.Select(kv => new object?[] { kv.Key, kv.Value }).ToList();

        if (value is IDictionary dictObj)
        {
            var result = new List<object?[]>();
            foreach (DictionaryEntry entry in dictObj)
                result.Add([entry.Key?.ToString() ?? "", entry.Value]);
            return result;
        }

        throw new InvalidOperationException("items filter requires a mapping.");
    }

    private static object DictSortFilter(object? value)
    {
        if (value is IDictionary<string, object?> dict)
            return dict.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                       .Select(kv => (object?)(new object?[] { kv.Key, kv.Value }))
                       .ToList();

        if (value is IDictionary dictObj)
        {
            var result = new List<object?>();
            var sorted = dictObj.Keys.Cast<object?>()
                               .OrderBy(k => k?.ToString() ?? "", StringComparer.Ordinal);
            foreach (var key in sorted)
                result.Add(new object?[] { key?.ToString() ?? "", dictObj[key] });
            return result;
        }

        throw new InvalidOperationException("dictsort filter requires a mapping.");
    }

    private static string Stringify(object? value)
    {
        return value switch
        {
            null => "",
            Undefined => "",
            string s => s,
            bool b => b ? "true" : "false",
            JsonElement je => JsonElementToString(je),
            _ => Convert.ToString(value, CultureInfo.InvariantCulture) ?? ""
        };
    }

    private static string JsonElementToString(JsonElement e)
    {
        return e.ValueKind switch
        {
            JsonValueKind.String => e.GetString() ?? "",
            JsonValueKind.Number => e.ToString(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            JsonValueKind.Null => "",
            _ => e.GetRawText()
        };
    }

    // Parses the LFM / LiquidAI format: <|tool_call_start|>[func(arg=val, ...)]<|tool_call_end|>
    private static bool TryParseLfmToolCalls(string text, out List<ToolCall> calls)
    {
        calls = new();

        const string open = "<|tool_call_start|>[";
        const string close = "]<|tool_call_end|>";

        int startIdx = text.IndexOf(open, StringComparison.Ordinal);
        if (startIdx < 0)
            return false;

        int contentStart = startIdx + open.Length;
        int endIdx = text.IndexOf(close, contentStart, StringComparison.Ordinal);
        if (endIdx < 0)
            return false;

        string content = text[contentStart..endIdx].Trim();
        if (string.IsNullOrEmpty(content))
            return false;

        foreach (string item in SplitTopLevelArguments(content))
        {
            string callText = item.Trim();
            if (string.IsNullOrEmpty(callText))
                continue;

            if (!TryParsePythonLikeCall(callText, out var call))
            {
                calls.Clear();
                return false;
            }

            calls.Add(call);
        }

        return calls.Count > 0;
    }

    // Parses the Gemma 4 format: <|tool_call>call:name{key:<|"|>str<|"|>,key2:42}<tool_call|>
    private static bool TryParseGemma4ToolCalls(string text, out List<ToolCall> calls)
    {
        calls = new();

        const string open = "<|tool_call>";
        const string close = "<tool_call|>";
        const string callPrefix = "call:";

        int pos = 0;
        while (true)
        {
            int tcStart = text.IndexOf(open, pos, StringComparison.Ordinal);
            if (tcStart < 0) break;

            int tcEnd = text.IndexOf(close, tcStart + open.Length, StringComparison.Ordinal);
            if (tcEnd < 0) return false;

            string body = text.Substring(tcStart + open.Length, tcEnd - (tcStart + open.Length)).Trim();

            if (!body.StartsWith(callPrefix, StringComparison.Ordinal))
                return false;

            body = body[callPrefix.Length..];

            int braceStart = body.IndexOf('{');
            if (braceStart < 0) return false;

            string funcName = body[..braceStart].Trim();
            if (string.IsNullOrEmpty(funcName)) return false;

            int braceEnd = body.LastIndexOf('}');
            if (braceEnd <= braceStart) return false;

            string argsBody = body.Substring(braceStart + 1, braceEnd - braceStart - 1);
            calls.Add(new ToolCall(funcName, ParseGemma4Args(argsBody)));

            pos = tcEnd + close.Length;
        }

        return calls.Count > 0;
    }

    private static IReadOnlyDictionary<string, object?> ParseGemma4Args(string body)
    {
        const string strDelim = "<|\"|>";
        var result = new Dictionary<string, object?>(StringComparer.Ordinal);
        int i = 0;

        while (i < body.Length)
        {
            while (i < body.Length && (body[i] == ',' || char.IsWhiteSpace(body[i])))
                i++;
            if (i >= body.Length) break;

            int colonIdx = body.IndexOf(':', i);
            if (colonIdx < 0) break;

            string key = body[i..colonIdx].Trim();
            i = colonIdx + 1;

            while (i < body.Length && char.IsWhiteSpace(body[i]))
                i++;

            result[key] = ParseGemma4ArgValue(body, strDelim, ref i);
        }

        return result;
    }

    private static object? ParseGemma4ArgValue(string body, string strDelim, ref int i)
    {
        if (i >= body.Length) return null;

        if (i + strDelim.Length <= body.Length &&
            body.AsSpan(i, strDelim.Length).SequenceEqual(strDelim.AsSpan()))
        {
            i += strDelim.Length;
            int end = body.IndexOf(strDelim, i, StringComparison.Ordinal);
            if (end < 0) { i = body.Length; return null; }
            string val = body[i..end];
            i = end + strDelim.Length;
            return val;
        }

        if (body[i] == '{') return Gemma4ConsumeNested(body, '{', '}', strDelim, ref i);
        if (body[i] == '[') return Gemma4ConsumeNested(body, '[', ']', strDelim, ref i);

        int scalarEnd = i;
        while (scalarEnd < body.Length && body[scalarEnd] != ',' && body[scalarEnd] != '}' && body[scalarEnd] != ']')
            scalarEnd++;

        string scalar = body[i..scalarEnd].Trim();
        i = scalarEnd;

        if (scalar.Equals("true", StringComparison.OrdinalIgnoreCase)) return true;
        if (scalar.Equals("false", StringComparison.OrdinalIgnoreCase)) return false;
        if (long.TryParse(scalar, out long lv)) return lv;
        if (double.TryParse(scalar, NumberStyles.Float, CultureInfo.InvariantCulture, out double dv)) return dv;
        return scalar;
    }

    private static string Gemma4ConsumeNested(string body, char open, char close, string strDelim, ref int i)
    {
        int start = i;
        int depth = 0;
        while (i < body.Length)
        {
            if (i + strDelim.Length <= body.Length &&
                body.AsSpan(i, strDelim.Length).SequenceEqual(strDelim.AsSpan()))
            {
                i += strDelim.Length;
                int strEnd = body.IndexOf(strDelim, i, StringComparison.Ordinal);
                i = strEnd >= 0 ? strEnd + strDelim.Length : body.Length;
                continue;
            }
            if (body[i] == open) depth++;
            else if (body[i] == close) { depth--; if (depth == 0) { i++; break; } }
            i++;
        }
        return body[start..i];
    }

    private static bool TryParseXmlToolCalls(string text, out List<ToolCall> calls)
    {
        calls = new();

        const string toolCallOpen = "<tool_call>";
        const string toolCallClose = "</tool_call>";
        const string functionOpenPrefix = "<function=";
        const string functionOpenSuffix = ">";
        const string functionClose = "</function>";
        const string parameterOpenPrefix = "<parameter=";
        const string parameterOpenSuffix = ">";
        const string parameterClose = "</parameter>";

        int pos = 0;
        while (true)
        {
            int tcStart = text.IndexOf(toolCallOpen, pos, StringComparison.Ordinal);
            if (tcStart < 0)
                break;

            int tcEnd = text.IndexOf(toolCallClose, tcStart + toolCallOpen.Length, StringComparison.Ordinal);
            if (tcEnd < 0)
                return false;

            string body = text.Substring(tcStart + toolCallOpen.Length, tcEnd - (tcStart + toolCallOpen.Length));

            int fnStart = body.IndexOf(functionOpenPrefix, StringComparison.Ordinal);
            if (fnStart < 0)
                return false;

            fnStart += functionOpenPrefix.Length;

            int fnNameEnd = body.IndexOf(functionOpenSuffix, fnStart, StringComparison.Ordinal);
            if (fnNameEnd < 0)
                return false;

            string fnName = body.Substring(fnStart, fnNameEnd - fnStart).Trim();

            int fnClose = body.IndexOf(functionClose, fnNameEnd + functionOpenSuffix.Length, StringComparison.Ordinal);
            if (fnClose < 0)
                return false;

            string fnBody = body.Substring(fnNameEnd + functionOpenSuffix.Length, fnClose - (fnNameEnd + functionOpenSuffix.Length));

            var args = new Dictionary<string, object?>();
            int p = 0;

            while (true)
            {
                int ps = fnBody.IndexOf(parameterOpenPrefix, p, StringComparison.Ordinal);
                if (ps < 0)
                    break;

                ps += parameterOpenPrefix.Length;

                int pnEnd = fnBody.IndexOf(parameterOpenSuffix, ps, StringComparison.Ordinal);
                if (pnEnd < 0)
                    return false;

                string paramName = fnBody.Substring(ps, pnEnd - ps).Trim();

                int pvStart = pnEnd + parameterOpenSuffix.Length;
                int pe = fnBody.IndexOf(parameterClose, pvStart, StringComparison.Ordinal);
                if (pe < 0)
                    return false;

                string paramValue = fnBody.Substring(pvStart, pe - pvStart);
                args[paramName] = paramValue.Trim();

                p = pe + parameterClose.Length;
            }

            calls.Add(new ToolCall(fnName, args));
            pos = tcEnd + toolCallClose.Length;
        }

        return calls.Count > 0;
    }

    private static bool TryParseJsonLike(string text, out List<ToolCall> calls)
    {
        calls = new();

        foreach (string candidate in EnumerateJsonCandidates(text))
        {
            if (TryParseJsonDocument(candidate, out calls))
                return true;
        }

        return false;
    }

    private static bool TryParseToolCode(string text, out List<ToolCall> calls)
    {
        calls = new();

        foreach (Match match in Regex.Matches(text, "<tool_code>\\s*(.*?)\\s*</tool_code>", RegexOptions.Singleline | RegexOptions.IgnoreCase))
        {
            foreach (string line in match.Groups[1].Value.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (!TryParsePythonLikeCall(line, out var call))
                {
                    calls.Clear();
                    return false;
                }

                calls.Add(call);
            }
        }

        return calls.Count > 0;
    }

    private static IEnumerable<string> EnumerateJsonCandidates(string text)
    {
        foreach (Match match in Regex.Matches(text, "```(?:json)?\\s*(.*?)```", RegexOptions.Singleline | RegexOptions.IgnoreCase))
        {
            string block = match.Groups[1].Value.Trim();
            if (!string.IsNullOrWhiteSpace(block) && LooksLikeCompleteJson(block))
                yield return block;
        }

        int firstBrace = text.IndexOf('{');
        int lastBrace = text.LastIndexOf('}');
        if (firstBrace >= 0 && lastBrace > firstBrace)
        {
            string body = text.Substring(firstBrace, lastBrace - firstBrace + 1).Trim();
            if (!string.IsNullOrWhiteSpace(body) && LooksLikeCompleteJson(body))
                yield return body;
        }
    }

    private static bool LooksLikeCompleteJson(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return false;

        text = text.Trim();
        if (!((text[0] == '{' && text[^1] == '}') || (text[0] == '[' && text[^1] == ']')))
            return false;

        int braceDepth = 0;
        int bracketDepth = 0;
        bool escaping = false;
        char quote = '\0';

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];

            if (quote != '\0')
            {
                if (escaping)
                {
                    escaping = false;
                    continue;
                }

                if (c == '\\')
                {
                    escaping = true;
                    continue;
                }

                if (c == quote)
                    quote = '\0';

                continue;
            }

            if (c == '"' || c == '\'')
            {
                quote = c;
                continue;
            }

            switch (c)
            {
                case '{':
                    braceDepth++;
                    break;
                case '}':
                    braceDepth--;
                    if (braceDepth < 0)
                        return false;
                    break;
                case '[':
                    bracketDepth++;
                    break;
                case ']':
                    bracketDepth--;
                    if (bracketDepth < 0)
                        return false;
                    break;
            }
        }

        return quote == '\0' && !escaping && braceDepth == 0 && bracketDepth == 0;
    }

    private static bool TryParseJsonDocument(string json, out List<ToolCall> calls)
    {
        calls = new();

        try
        {
            using var document = JsonDocument.Parse(json);

            switch (document.RootElement.ValueKind)
            {
                case JsonValueKind.Object:
                    if (TryParseToolCallObject(document.RootElement, out var singleCall))
                    {
                        calls.Add(singleCall);
                        return true;
                    }

                    if (document.RootElement.TryGetProperty("tool_calls", out var toolCallsElement)
                        && toolCallsElement.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var item in toolCallsElement.EnumerateArray())
                        {
                            if (!TryParseToolCallObject(item, out var call))
                            {
                                calls.Clear();
                                return false;
                            }

                            calls.Add(call);
                        }

                        return calls.Count > 0;
                    }

                    break;

                case JsonValueKind.Array:
                    foreach (var item in document.RootElement.EnumerateArray())
                    {
                        if (!TryParseToolCallObject(item, out var call))
                        {
                            calls.Clear();
                            return false;
                        }

                        calls.Add(call);
                    }

                    return calls.Count > 0;
            }
        }
        catch (JsonException)
        {
        }

        calls.Clear();
        return false;
    }

    private static bool TryParseToolCallObject(JsonElement element, out ToolCall call)
    {
        call = default!;

        if (element.ValueKind != JsonValueKind.Object)
            return false;

        JsonElement functionElement = element;
        if (element.TryGetProperty("function", out var nestedFunction))
            functionElement = nestedFunction;

        if (!functionElement.TryGetProperty("name", out var nameElement) || nameElement.ValueKind != JsonValueKind.String)
            return false;

        string name = nameElement.GetString() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(name))
            return false;

        var args = new Dictionary<string, object?>();
        if (functionElement.TryGetProperty("arguments", out var argsElement))
        {
            if (argsElement.ValueKind == JsonValueKind.String)
            {
                string rawArgs = argsElement.GetString() ?? string.Empty;
                if (!string.IsNullOrWhiteSpace(rawArgs))
                {
                    try
                    {
                        using var argsDocument = JsonDocument.Parse(rawArgs);
                        if (argsDocument.RootElement.ValueKind == JsonValueKind.Object)
                            args = ParseArgumentsObject(argsDocument.RootElement);
                    }
                    catch (JsonException)
                    {
                    }
                }
            }
            else if (argsElement.ValueKind == JsonValueKind.Object)
            {
                args = ParseArgumentsObject(argsElement);
            }
        }

        string callId = $"call_{Guid.NewGuid():N}";

        if (element.TryGetProperty("call_id", out var callIdElement) && callIdElement.ValueKind == JsonValueKind.String)
            callId = callIdElement.GetString() ?? callId;
        else if (element.TryGetProperty("id", out var idElement) && idElement.ValueKind == JsonValueKind.String)
            callId = idElement.GetString() ?? callId;

        call = new ToolCall(name, args, callId);
        return true;
    }

    private static bool TryParsePythonLikeCall(string text, out ToolCall call)
    {
        call = default!;

        Match match = Regex.Match(text, "^([A-Za-z_][A-Za-z0-9_]*)\\s*\\((.*)\\)\\s*;?$", RegexOptions.Singleline);
        if (!match.Success)
            return false;

        string name = match.Groups[1].Value;
        string argsText = match.Groups[2].Value.Trim();
        var args = new Dictionary<string, object?>();

        if (!string.IsNullOrEmpty(argsText))
        {
            foreach (string argument in SplitTopLevelArguments(argsText))
            {
                int equalsIndex = FindTopLevelEqualsInArguments(argument);
                if (equalsIndex < 0)
                    return false;

                string argName = argument[..equalsIndex].Trim();
                string argValue = argument[(equalsIndex + 1)..].Trim();

                if (string.IsNullOrWhiteSpace(argName))
                    return false;

                args[argName] = ParsePythonLikeValue(argValue);
            }
        }

        call = new ToolCall(name, args);
        return true;
    }

    private static string RemoveDelimitedBlocks(string text, string open, string close)
    {
        var sb = new StringBuilder();
        int position = 0;

        while (true)
        {
            int start = text.IndexOf(open, position, StringComparison.Ordinal);
            if (start < 0)
            {
                sb.Append(text, position, text.Length - position);
                break;
            }

            sb.Append(text, position, start - position);

            int end = text.IndexOf(close, start + open.Length, StringComparison.Ordinal);
            if (end < 0)
                break;

            position = end + close.Length;
        }

        return sb.ToString();
    }

    private static List<string> SplitTopLevelArguments(string text)
    {
        var arguments = new List<string>();
        var current = new StringBuilder();
        int parenDepth = 0;
        int bracketDepth = 0;
        int braceDepth = 0;
        char quote = '\0';

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];

            if (quote != '\0')
            {
                current.Append(c);

                if (c == '\\' && i + 1 < text.Length)
                {
                    current.Append(text[++i]);
                    continue;
                }

                if (c == quote)
                    quote = '\0';

                continue;
            }

            switch (c)
            {
                case '\'':
                case '"':
                    quote = c;
                    current.Append(c);
                    break;
                case '(':
                    parenDepth++;
                    current.Append(c);
                    break;
                case ')':
                    parenDepth--;
                    current.Append(c);
                    break;
                case '[':
                    bracketDepth++;
                    current.Append(c);
                    break;
                case ']':
                    bracketDepth--;
                    current.Append(c);
                    break;
                case '{':
                    braceDepth++;
                    current.Append(c);
                    break;
                case '}':
                    braceDepth--;
                    current.Append(c);
                    break;
                case ',':
                    if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0)
                    {
                        arguments.Add(current.ToString().Trim());
                        current.Clear();
                    }
                    else
                    {
                        current.Append(c);
                    }
                    break;
                default:
                    current.Append(c);
                    break;
            }
        }

        if (current.Length > 0)
            arguments.Add(current.ToString().Trim());

        return arguments;
    }

    private static int FindTopLevelEqualsInArguments(string text)
    {
        int parenDepth = 0;
        int bracketDepth = 0;
        int braceDepth = 0;
        char quote = '\0';

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];

            if (quote != '\0')
            {
                if (c == '\\' && i + 1 < text.Length)
                {
                    i++;
                    continue;
                }

                if (c == quote)
                    quote = '\0';

                continue;
            }

            switch (c)
            {
                case '\'':
                case '"':
                    quote = c;
                    break;
                case '(':
                    parenDepth++;
                    break;
                case ')':
                    parenDepth--;
                    break;
                case '[':
                    bracketDepth++;
                    break;
                case ']':
                    bracketDepth--;
                    break;
                case '{':
                    braceDepth++;
                    break;
                case '}':
                    braceDepth--;
                    break;
                case '=':
                    if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0)
                        return i;
                    break;
            }
        }

        return -1;
    }

    private static object? ParsePythonLikeValue(string text)
    {
        if (string.Equals(text, "true", StringComparison.OrdinalIgnoreCase))
            return true;

        if (string.Equals(text, "false", StringComparison.OrdinalIgnoreCase))
            return false;

        if (string.Equals(text, "null", StringComparison.OrdinalIgnoreCase)
            || string.Equals(text, "none", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        if (text.Length >= 2 && ((text[0] == '\'' && text[^1] == '\'') || (text[0] == '"' && text[^1] == '"')))
            return text[1..^1];

        if (long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long intValue))
            return intValue;

        if (double.TryParse(text, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out double doubleValue))
            return doubleValue;

        if ((text.StartsWith('{') && text.EndsWith('}')) || (text.StartsWith('[') && text.EndsWith(']')))
        {
            try
            {
                using var document = JsonDocument.Parse(text);
                return ConvertJsonValue(document.RootElement);
            }
            catch (JsonException)
            {
            }
        }

        return text;
    }

    private static Dictionary<string, object?> ParseArgumentsObject(JsonElement element)
    {
        var args = new Dictionary<string, object?>();
        foreach (var property in element.EnumerateObject())
            args[property.Name] = ConvertJsonValue(property.Value);

        return args;
    }

    private static object? ConvertJsonValue(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number when element.TryGetInt64(out long intValue) => intValue,
            JsonValueKind.Number => element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => null,
            JsonValueKind.Object => ParseArgumentsObject(element),
            JsonValueKind.Array => ParseJsonArray(element),
            _ => element.ToString()
        };
    }

    private static object?[] ParseJsonArray(JsonElement element)
    {
        var items = new List<object?>();
        foreach (var item in element.EnumerateArray())
            items.Add(ConvertJsonValue(item));

        return [.. items];
    }
}