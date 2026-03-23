using System.Reflection;
using System.Text;

namespace YALMR.Runtime;

/// <summary>
/// Marks a public method as a tool exposed to the model.
/// The method name is converted to snake_case automatically; override with <see cref="Name"/>.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class ToolAttribute(string description) : Attribute
{
    public string Description { get; } = description;
    public string? Name { get; init; }
}

/// <summary>
/// Provides a description for a tool method parameter.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter)]
public sealed class ToolParamAttribute(string description) : Attribute
{
    public string Description { get; } = description;
}

/// <summary>
/// Reflection-based tool registration for <see cref="ToolRegistry"/>.
/// </summary>
public static class ToolRegistryReflectionExtensions
{
    /// <summary>
    /// Scans <paramref name="instance"/> for public methods marked with <see cref="ToolAttribute"/>
    /// and registers each one as an <see cref="AgentTool"/>.
    /// </summary>
    public static void Register(this ToolRegistry registry, object instance)
    {
        ArgumentNullException.ThrowIfNull(instance);

        foreach (var method in instance.GetType()
            .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            var attr = method.GetCustomAttribute<ToolAttribute>();
            if (attr is null) continue;

            registry.Register(BuildTool(method, attr, instance));
        }
    }

    /// <summary>
    /// Resolves <typeparamref name="T"/> from <paramref name="services"/> and registers its tools.
    /// </summary>
    public static void Register<T>(this ToolRegistry registry, IServiceProvider services) where T : class
    {
        var instance = services.GetService(typeof(T)) as T
            ?? throw new InvalidOperationException(
                $"{typeof(T).Name} is not registered in the service provider.");
        registry.Register(instance);
    }

    // -------------------------------------------------------------------------

    private static AgentTool BuildTool(MethodInfo method, ToolAttribute attr, object instance)
    {
        string name       = attr.Name ?? ToSnakeCase(method.Name);
        var    parameters = BuildParameters(method);

        return new AgentTool(name, attr.Description, parameters,
            args => InvokeMethod(method, instance, args));
    }

    private static IReadOnlyList<ToolParameter> BuildParameters(MethodInfo method)
    {
        var result = new List<ToolParameter>();
        foreach (var p in method.GetParameters())
        {
            string description = p.GetCustomAttribute<ToolParamAttribute>()?.Description
                                 ?? p.Name
                                 ?? string.Empty;
            string schemaType  = MapSchemaType(p.ParameterType);
            bool   required    = !p.HasDefaultValue && !IsNullable(p);
            result.Add(new ToolParameter(p.Name!, schemaType, description, required));
        }
        return result;
    }

    private static string InvokeMethod(
        MethodInfo method,
        object instance,
        IReadOnlyDictionary<string, object?> args)
    {
        var parms      = method.GetParameters();
        var invokeArgs = new object?[parms.Length];

        for (int i = 0; i < parms.Length; i++)
        {
            var p = parms[i];
            invokeArgs[i] = args.TryGetValue(p.Name!, out var raw)
                ? CoerceValue(raw, p.ParameterType)
                : p.HasDefaultValue ? p.DefaultValue : DefaultFor(p.ParameterType);
        }

        object? result;
        try
        {
            result = method.Invoke(instance, invokeArgs);
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            throw ex.InnerException;
        }

        if (result is Task<string> ts) return ts.GetAwaiter().GetResult();
        if (result is Task t) { t.GetAwaiter().GetResult(); return string.Empty; }

        return result switch
        {
            string s => s,
            null     => string.Empty,
            _        => result.ToString() ?? string.Empty,
        };
    }

    private static object? CoerceValue(object? value, Type target)
    {
        if (value is null) return DefaultFor(target);

        var inner = Nullable.GetUnderlyingType(target) ?? target;

        if (inner.IsInstanceOfType(value)) return value;

        return inner switch
        {
            _ when inner == typeof(string)  => value.ToString(),
            _ when inner == typeof(bool)    => Convert.ToBoolean(value),
            _ when inner == typeof(int)     => value is long l   ? (int)l    : Convert.ToInt32(value),
            _ when inner == typeof(short)   => value is long l   ? (short)l  : Convert.ToInt16(value),
            _ when inner == typeof(long)    => value is long l2  ? l2        : Convert.ToInt64(value),
            _ when inner == typeof(float)   => value is double d ? (float)d  : Convert.ToSingle(value),
            _ when inner == typeof(decimal) => value is double d ? (decimal)d : Convert.ToDecimal(value),
            _ when inner.IsArray            => CoerceArray(value, inner),
            _                               => value,
        };
    }

    private static object CoerceArray(object value, Type arrayType)
    {
        var raw         = value as object?[] ?? [];
        var elementType = arrayType.GetElementType()!;
        var arr         = Array.CreateInstance(elementType, raw.Length);
        for (int i = 0; i < raw.Length; i++)
            arr.SetValue(CoerceValue(raw[i], elementType), i);
        return arr;
    }

    private static object? DefaultFor(Type type) =>
        type.IsValueType ? Activator.CreateInstance(type) : null;

    private static bool IsNullable(ParameterInfo p)
    {
        if (Nullable.GetUnderlyingType(p.ParameterType) is not null) return true;
        return new NullabilityInfoContext().Create(p).WriteState == NullabilityState.Nullable;
    }

    private static string MapSchemaType(Type type)
    {
        type = Nullable.GetUnderlyingType(type) ?? type;

        if (type == typeof(string))  return "string";
        if (type == typeof(bool))    return "boolean";
        if (type == typeof(int)  || type == typeof(long)  ||
            type == typeof(short) || type == typeof(byte)) return "integer";
        if (type == typeof(double) || type == typeof(float) ||
            type == typeof(decimal)) return "number";
        if (type.IsArray || IsCollectionType(type)) return "array";

        return "object";
    }

    private static bool IsCollectionType(Type type) =>
        type.IsGenericType && type.GetGenericTypeDefinition() is { } def &&
        (def == typeof(List<>)          || def == typeof(IList<>)         ||
         def == typeof(IReadOnlyList<>) || def == typeof(IEnumerable<>)   ||
         def == typeof(ICollection<>));

    private static string ToSnakeCase(string name)
    {
        var sb = new StringBuilder(name.Length + 4);
        for (int i = 0; i < name.Length; i++)
        {
            char c = name[i];
            if (char.IsUpper(c) && i > 0) sb.Append('_');
            sb.Append(char.ToLowerInvariant(c));
        }
        return sb.ToString();
    }
}
