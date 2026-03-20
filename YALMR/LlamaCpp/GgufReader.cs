using System.Collections.Generic;
using System.IO;
using System.Text;

namespace YALMR.LlamaCpp;

/// <summary>
/// Reads GGUF metadata needed for prompt templates and tokenizer details.
/// </summary>
public static class GgufReader
{
    /// <summary>
    /// Reads all top-level GGUF metadata key-value pairs from a model file.
    /// </summary>
    public static IReadOnlyDictionary<string, object?> ReadMetadata(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);

        var magic = br.ReadUInt32();
        if (magic != 0x46554747)
            throw new InvalidDataException("Not a GGUF file.");

        var version = br.ReadUInt32();
        if (version != 2 && version != 3)
            throw new NotSupportedException($"Unsupported GGUF version: {version}");

        _ = br.ReadUInt64();
        ulong kvCount = br.ReadUInt64();

        var metadata = new Dictionary<string, object?>(StringComparer.Ordinal);
        for (ulong i = 0; i < kvCount; i++)
        {
            string key = ReadGgufString(br);
            uint valueType = br.ReadUInt32();
            metadata[key] = ReadValue(br, valueType);
        }

        return metadata;
    }

    /// <summary>
    /// Reads the tokenizer chat template from a GGUF file.
    /// </summary>
    public static string? ReadChatTemplate(string path)
    {
        var metadata = ReadMetadata(path);
        return GetString(metadata, "tokenizer.chat_template");
    }

    /// <summary>
    /// Gets a string metadata value by key.
    /// </summary>
    public static string? GetString(IReadOnlyDictionary<string, object?> metadata, string key)
    {
        return metadata.TryGetValue(key, out var value) ? value as string : null;
    }

    /// <summary>
    /// Gets a string-array metadata value by key.
    /// </summary>
    public static string[] GetStringArray(IReadOnlyDictionary<string, object?> metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var value) || value is not object?[] items)
            return [];

        var strings = new List<string>(items.Length);
        foreach (var item in items)
        {
            if (item is string text)
                strings.Add(text);
        }

        return [.. strings];
    }

    /// <summary>
    /// Gets an integer metadata value by key when it can be losslessly converted to <see cref="int"/>.
    /// </summary>
    public static int? GetInt32(IReadOnlyDictionary<string, object?> metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var value) || value is null)
            return null;

        return value switch
        {
            byte v => v,
            sbyte v => v,
            short v => v,
            ushort v => v,
            int v => v,
            uint v => checked((int)v),
            long v => checked((int)v),
            ulong v => checked((int)v),
            _ => null
        };
    }

    /// <summary>
    /// Resolves a token string from a token-id metadata key.
    /// </summary>
    public static string? ResolveTokenById(
        IReadOnlyDictionary<string, object?> metadata,
        string tokenIdKey,
        string tokensKey = "tokenizer.ggml.tokens")
    {
        int? tokenId = GetInt32(metadata, tokenIdKey);
        if (tokenId is null || tokenId < 0)
            return null;

        string[] tokens = GetStringArray(metadata, tokensKey);
        return tokenId.Value < tokens.Length ? tokens[tokenId.Value] : null;
    }

    private static string ReadGgufString(BinaryReader br)
    {
        ulong len = br.ReadUInt64();
        byte[] bytes = br.ReadBytes(checked((int)len));
        return Encoding.UTF8.GetString(bytes);
    }

    private static object? ReadValue(BinaryReader br, uint type)
    {
        // GGUF metadata value types
        // 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool
        // 8=string 9=array 10=u64 11=i64 12=f64
        return type switch
        {
            0 => br.ReadByte(),
            1 => br.ReadSByte(),
            2 => br.ReadUInt16(),
            3 => br.ReadInt16(),
            4 => br.ReadUInt32(),
            5 => br.ReadInt32(),
            6 => br.ReadSingle(),
            7 => br.ReadByte() != 0,
            8 => ReadGgufString(br),
            9 => ReadArray(br),
            10 => br.ReadUInt64(),
            11 => br.ReadInt64(),
            12 => br.ReadDouble(),
            _ => throw new NotSupportedException($"Unsupported GGUF metadata type: {type}")
        };
    }

    private static object? ReadArray(BinaryReader br)
    {
        uint elementType = br.ReadUInt32();
        ulong count = br.ReadUInt64();
        var items = new object?[checked((int)count)];

        for (int i = 0; i < items.Length; i++)
            items[i] = ReadValue(br, elementType);

        return items;
    }
}