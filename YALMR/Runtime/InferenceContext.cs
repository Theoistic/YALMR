using System.Runtime.CompilerServices;
using System.Text;
using YALMR.Diagnostics;
using YALMR.LlamaCpp;

namespace YALMR.Runtime;

/// <summary>
/// Per-session native inference state owning its own <see cref="Llama.Context"/> and KV cache.
/// Each <see cref="Session"/> creates a dedicated instance so sessions can run truly in
/// parallel without sharing native context state.
/// </summary>
public sealed class InferenceContext : IAsyncDisposable, IDisposable
{
    private readonly Engine _engine;
    private readonly SessionOptions _options;
    private readonly List<int> _cachedTokens = [];
    private readonly Random _random;
    private readonly int _nBatch;

    private Llama.Context _context;
    private bool _cacheContainsVision;
    private bool _disposed;

    // Stateful UTF-8 decoder for token streaming — buffers incomplete multi-byte sequences
    // (e.g. emojis split across tokens) so they are emitted whole instead of as U+FFFD.
    private readonly Decoder _utf8Decoder = Encoding.UTF8.GetDecoder();
    private readonly byte[] _tokenBuf = new byte[256];
    private readonly char[] _charBuf = new char[257]; // UTF8.GetMaxCharCount(256) == 257

    internal InferenceContext(
        Engine engine,
        SessionOptions options,
        Llama.Context context,
        int nBatch,
        Random random)
    {
        _engine = engine;
        _options = options;
        _context = context;
        _nBatch = nBatch;
        _random = random;
    }

    /// <summary>
    /// Encodes a prompt into the native context, using the vision pipeline when images are present.
    /// </summary>
    public async Task EncodePromptAsync(string prompt, int[] promptTokens, List<string> imageBase64s, CancellationToken ct)
    {
        bool hasImages = _engine.VisionEnabled
            && imageBase64s.Count > 0
            && !string.IsNullOrEmpty(_engine.ImageToken)
            && prompt.Contains(_engine.ImageToken, StringComparison.Ordinal);

        if (hasImages)
        {
            ResetCacheInternal();
            string mtmdPrompt = prompt.Replace(_engine.ImageToken, Llama.Vision.DefaultMarker, StringComparison.Ordinal);
            int imageMarkerCount = CountOccurrences(mtmdPrompt, Llama.Vision.DefaultMarker);

            if (imageMarkerCount != imageBase64s.Count)
                throw new InvalidOperationException($"Prompt expects {imageMarkerCount} image(s), but {imageBase64s.Count} image payload(s) were collected from the fitted history.");

            await _engine.EvalVisionPromptAsync(_context, mtmdPrompt, imageBase64s, _nBatch, ct);
            _cacheContainsVision = true;
        }
        else
        {
            await Task.Run(() => DecodePromptWithCacheContinuation(promptTokens), ct);
        }
    }

    /// <summary>
    /// Streams tokens from the model for the current context state.
    /// </summary>
    public async IAsyncEnumerable<InferenceToken> GenerateTokensAsync(
        InferenceOptions options,
        int maxOutputTokens,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        using var activity = RuntimeTelemetry.StartActivity("yalmr.context.generate_tokens");
        activity?.SetTag("yalmr.max_output_tokens", maxOutputTokens);

        var samplerChain = Llama.CreateSamplerChain(options, _random);
        int emittedTokens = 0;
        _utf8Decoder.Reset();
        try
        {
            for (int i = 0; i < maxOutputTokens; i++)
            {
                ct.ThrowIfCancellationRequested();

                int token = Llama.SampleWithChain(samplerChain, _context);

                if (_engine.IsEndOfGeneration(token))
                {
                    _options.Logger?.Log(LogLevel.Debug, "generation", $"stop reason=eog token={token} emitted={emittedTokens} max={maxOutputTokens}");
                    activity?.SetTag("yalmr.tokens_emitted", emittedTokens);
                    activity?.SetTag("yalmr.stop_reason", "eog");
                    RuntimeTelemetry.TokensGenerated.Add(emittedTokens);
                    yield break;
                }

                int byteCount = _engine.TokenToBytes(token, _tokenBuf);
                int charCount = _utf8Decoder.GetChars(_tokenBuf, 0, byteCount, _charBuf, 0, flush: false);
                string piece = new string(_charBuf, 0, charCount);
                emittedTokens++;

                yield return new InferenceToken(token, piece, false);

                DecodeToken(token);
            }

            _options.Logger?.Log(LogLevel.Debug, "generation", $"stop reason=max_output_tokens emitted={emittedTokens} max={maxOutputTokens}");
            activity?.SetTag("yalmr.tokens_emitted", emittedTokens);
            activity?.SetTag("yalmr.stop_reason", "max_output_tokens");
            RuntimeTelemetry.TokensGenerated.Add(emittedTokens);
        }
        finally
        {
            Llama.FreeSamplerChain(samplerChain);
        }
    }

    /// <summary>
    /// Generates an embedding vector as <see cref="float"/> values.
    /// </summary>
    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        ResetCacheInternal();
        int[] tokens = _engine.Tokenize(text);
        await Task.Run(() => DecodePromptWithCacheContinuation(tokens), ct);
        return await Task.Run(() => Llama.GetEmbeddings(_context, _engine.NativeModel), ct);
    }

    /// <summary>
    /// Generates an embedding vector as <see cref="double"/> values.
    /// </summary>
    public async Task<double[]> EmbedAsDoubleAsync(string text, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        ResetCacheInternal();
        int[] tokens = _engine.Tokenize(text);
        await Task.Run(() => DecodePromptWithCacheContinuation(tokens), ct);
        return await Task.Run(() => Llama.GetEmbeddingsAsDouble(_context, _engine.NativeModel), ct);
    }

    private void DecodeToken(int token)
    {
        using var batch = Llama.CreateBatch([token]);
        int rc = Llama.Decode(_context, batch);
        if (rc != 0) throw new InvalidOperationException($"llama_decode failed: {rc}");
        _cachedTokens.Add(token);
    }

    private void DecodePromptWithCacheContinuation(int[] promptTokens)
    {
        if (_cacheContainsVision) ResetCacheInternal();

        int prefixLength = GetCommonPrefixLength(promptTokens, _cachedTokens);

        if (prefixLength < _cachedTokens.Count)
        {
            if (!Llama.TryRemoveCacheRange(_context, 0, prefixLength))
            {
                ResetCacheInternal();
                prefixLength = 0;
            }
            else
            {
                _cachedTokens.RemoveRange(prefixLength, _cachedTokens.Count - prefixLength);
            }
        }

        if (prefixLength < promptTokens.Length)
        {
            int[] suffix = promptTokens[prefixLength..];
            for (int offset = 0; offset < suffix.Length; offset += _options.BatchTokens)
            {
                int count = Math.Min(_options.BatchTokens, suffix.Length - offset);
                int[] chunk = new int[count];
                Array.Copy(suffix, offset, chunk, 0, count);

                using var batch = Llama.CreateBatch(chunk);
                if (Llama.Decode(_context, batch) != 0) throw new InvalidOperationException("Decode failed.");
            }
            _cachedTokens.AddRange(suffix);
        }
    }

    private void ResetCacheInternal()
    {
        Llama.FreeContext(_context);
        _context = Llama.CreateContext(
            _engine.NativeModel,
            nCtx: _options.ContextTokens,
            nBatch: _nBatch,
            nUbatch: _options.MicroBatchTokens,
            nThreads: _options.Threads,
            embeddings: true,
            unifiedKvCache: _options.UnifiedKvCache,
            ropeFreqBase: _options.RopeFrequencyBase,
            ropeFreqScale: _options.RopeFrequencyScale,
            offloadKvCacheToGpu: _options.OffloadKvCacheToGpu,
            flashAttention: _options.FlashAttention,
            kvCacheTypeK: _options.KvCacheTypeK,
            kvCacheTypeV: _options.KvCacheTypeV);
        _cachedTokens.Clear();
        _cacheContainsVision = false;
    }

    private static int GetCommonPrefixLength(int[] prompt, List<int> cache)
    {
        int limit = Math.Min(prompt.Length, cache.Count);
        int i = 0;
        while (i < limit && prompt[i] == cache[i]) i++;
        return i;
    }

    private static int CountOccurrences(string text, string value)
    {
        if (string.IsNullOrEmpty(text) || string.IsNullOrEmpty(value))
            return 0;

        int count = 0;
        int index = 0;

        while ((index = text.IndexOf(value, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += value.Length;
        }

        return count;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        await Task.Run(() =>
        {
            if (!_context.IsNull) Llama.FreeContext(_context);
        });
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();
}
