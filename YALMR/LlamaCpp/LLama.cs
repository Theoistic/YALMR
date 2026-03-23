using YALMR.Runtime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using YALMR.Diagnostics;

namespace YALMR.LlamaCpp;

public enum KvCacheQuantization { F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 }

/// <summary>
/// Thin native interop wrapper over llama.cpp and related multimodal helpers.
/// </summary>
public static class Llama
{
    private const string LLAMA_LIB = "llama";
    private const string GGML_LIB = "ggml"; // if your build exports from ggml.dll instead, change this to "ggml"

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void NativeLogCallback(int level, IntPtr text, IntPtr userData);

    private static readonly NativeLogCallback s_llamaLogCallback = HandleLlamaLog;
    private static readonly NativeLogCallback s_ggmlLogCallback = HandleGgmlLog;
    private static ILogger s_logger = NullLogger.Instance;

    /// <summary>
    /// Handle to a loaded llama model.
    /// </summary>
    public readonly struct Model
    {
        public readonly IntPtr Ptr;
        public Model(IntPtr ptr) => Ptr = ptr;
        public bool IsNull => Ptr == IntPtr.Zero;
    }

    /// <summary>
    /// Handle to an active llama inference context.
    /// </summary>
    public readonly struct Context
    {
        public readonly IntPtr Ptr;
        public Context(IntPtr ptr) => Ptr = ptr;
        public bool IsNull => Ptr == IntPtr.Zero;
    }

    /// <summary>
    /// Handle to a model vocabulary.
    /// </summary>
    public readonly struct Vocab
    {
        public readonly IntPtr Ptr;
        public Vocab(IntPtr ptr) => Ptr = ptr;
        public bool IsNull => Ptr == IntPtr.Zero;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct llama_model_params
    {
        public IntPtr devices;
        public IntPtr tensor_buft_overrides;
        public int n_gpu_layers;
        public int split_mode;
        public int main_gpu;
        public IntPtr tensor_split;
        public IntPtr progress_callback;
        public IntPtr progress_callback_user_data;
        public IntPtr kv_overrides;

        [MarshalAs(UnmanagedType.I1)] public bool vocab_only;
        [MarshalAs(UnmanagedType.I1)] public bool use_mmap;
        [MarshalAs(UnmanagedType.I1)] public bool use_direct_io;
        [MarshalAs(UnmanagedType.I1)] public bool use_mlock;
        [MarshalAs(UnmanagedType.I1)] public bool check_tensors;
        [MarshalAs(UnmanagedType.I1)] public bool use_extra_bufts;
        [MarshalAs(UnmanagedType.I1)] public bool no_host;
        [MarshalAs(UnmanagedType.I1)] public bool no_alloc;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct llama_context_params
    {
        public uint n_ctx;
        public uint n_batch;
        public uint n_ubatch;
        public uint n_seq_max;

        public int n_threads;
        public int n_threads_batch;

        public int rope_scaling_type;
        public int pooling_type;
        public int attention_type;
        public int flash_attn_type;

        public float rope_freq_base;
        public float rope_freq_scale;
        public float yarn_ext_factor;
        public float yarn_attn_factor;
        public float yarn_beta_fast;
        public float yarn_beta_slow;
        public uint yarn_orig_ctx;
        public float defrag_thold;

        public IntPtr cb_eval;
        public IntPtr cb_eval_user_data;

        public int type_k;
        public int type_v;

        public IntPtr abort_callback;
        public IntPtr abort_callback_data;

        [MarshalAs(UnmanagedType.I1)] public bool embeddings;
        [MarshalAs(UnmanagedType.I1)] public bool offload_kqv;
        [MarshalAs(UnmanagedType.I1)] public bool no_perf;
        [MarshalAs(UnmanagedType.I1)] public bool op_offload;
        [MarshalAs(UnmanagedType.I1)] public bool swa_full;
        [MarshalAs(UnmanagedType.I1)] public bool kv_unified;

        public IntPtr samplers;
        public UIntPtr n_samplers;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct llama_batch
    {
        public int n_tokens;
        public IntPtr token;
        public IntPtr embd;
        public IntPtr pos;
        public IntPtr n_seq_id;
        public IntPtr seq_id;
        public IntPtr logits;
    }

    /// <summary>
    /// Managed wrapper for a pinned llama batch.
    /// </summary>
    public sealed class Batch : IDisposable
    {
        internal GCHandle TokensHandle;
        internal llama_batch Native;
        internal bool Pinned;

        public int Count => Native.n_tokens;

        internal Batch(GCHandle tokensHandle, llama_batch native)
        {
            TokensHandle = tokensHandle;
            Native = native;
            Pinned = true;
        }

        public void Dispose()
        {
            if (Pinned)
            {
                TokensHandle.Free();
                Pinned = false;
            }
        }
    }

    // Initializes the global llama backend before any models or contexts are created.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void llama_backend_init();

    // Releases the global llama backend state.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void llama_backend_free();

    // Loads every available ggml backend from the default search path.
    [DllImport(GGML_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void ggml_backend_load_all();

    // Loads ggml backends from a specific directory.
    [DllImport(GGML_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void ggml_backend_load_all_from_path(IntPtr dirPath);

    // Returns the native default model parameters.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern llama_model_params llama_model_default_params();

    // Returns the native default context parameters.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern llama_context_params llama_context_default_params();

    // Registers a native log callback for llama.cpp messages.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void llama_log_set(NativeLogCallback callback, IntPtr userData);

    // Registers a native log callback for ggml messages.
    [DllImport(GGML_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void ggml_log_set(NativeLogCallback callback, IntPtr userData);

    // Opens a GGUF model from disk.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_model_load_from_file(
        IntPtr pathModel,
        llama_model_params parameters);

    // Creates an inference context bound to an already loaded model.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_init_from_model(
        IntPtr model,
        llama_context_params parameters);

    // Releases a loaded model.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void llama_model_free(IntPtr model);

    // Releases an inference context.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void llama_free(IntPtr ctx);

    // Fetches the vocabulary handle for a loaded model.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_model_get_vocab(IntPtr model);

    // Returns the vocabulary size.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern int llama_vocab_n_tokens(IntPtr vocab);

    // Tokenizes UTF-8 text into model token ids.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern int llama_tokenize(
        IntPtr vocab,
        IntPtr text,
        int text_len,
        IntPtr tokens,
        int n_tokens_max,
        [MarshalAs(UnmanagedType.I1)] bool add_special,
        [MarshalAs(UnmanagedType.I1)] bool parse_special);

    // Builds a simple batch over a contiguous token buffer.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern llama_batch llama_batch_get_one(
        IntPtr tokens,
        int n_tokens);

    // Decodes one batch into the context.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern int llama_decode(
        IntPtr ctx,
        llama_batch batch);

    // Removes cached KV entries for a sequence in the specified position range.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern bool llama_kv_cache_seq_rm(
        IntPtr ctx,
        int seq_id,
        int p0,
        int p1);

    // Returns the logits buffer for a decoded position.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_get_logits_ith(
        IntPtr ctx,
        int i);

    // Returns the embedding dimension for the model.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern int llama_n_embd(IntPtr model);

    // Returns the pooled sequence embeddings for the given sequence id when available.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_get_embeddings_seq(
        IntPtr ctx,
        int seqId);

    // Returns the current embeddings buffer when pooled sequence embeddings are unavailable.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_get_embeddings(IntPtr ctx);

    // Converts one token id back into its text piece.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern int llama_token_to_piece(
        IntPtr vocab,
        int token,
        byte[] buf,
        int length,
        int lstrip,
        [MarshalAs(UnmanagedType.I1)] bool special);

    // Reports whether a token should terminate generation.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern bool llama_vocab_is_eog(
        IntPtr vocab,
        int token);

    [StructLayout(LayoutKind.Sequential)]
    private struct llama_sampler_chain_params
    {
        [MarshalAs(UnmanagedType.I1)] public bool no_perf;
    }

    // Returns default sampler chain parameters.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern llama_sampler_chain_params llama_sampler_chain_default_params();

    // Creates a new sampler chain.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_sampler_chain_init(llama_sampler_chain_params parms);

    // Appends a sampler to the chain (chain takes ownership).
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void llama_sampler_chain_add(IntPtr chain, IntPtr smpl);

    // Frees any sampler, including chains.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void llama_sampler_free(IntPtr smpl);

    // Applies the sampler chain to the logits and returns the sampled token.
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern int llama_sampler_sample(IntPtr smpl, IntPtr ctx, int idx);

    // Notifies the chain that a token was accepted (updates penalty window).
    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern void llama_sampler_accept(IntPtr smpl, int token);

    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_sampler_init_greedy();

    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_sampler_init_dist(uint seed);

    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_sampler_init_top_k(int k);

    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_sampler_init_top_p(float p, nuint min_keep);

    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_sampler_init_temp(float t);

    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_sampler_init_penalties(
        int penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);

    [DllImport(LLAMA_LIB, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_sampler_init_grammar(IntPtr vocab, IntPtr grammar_str, IntPtr grammar_root);

    private static IntPtr AllocUtf8(string s)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(s + "\0");
        IntPtr ptr = Marshal.AllocHGlobal(utf8.Length);
        Marshal.Copy(utf8, 0, ptr, utf8.Length);
        return ptr;
    }

    /// <summary>
    /// Initializes native llama and ggml backends.
    /// </summary>
    public static void Init(string? backendDirectory = null, ILogger? logger = null)
    {
        try
        {
            SetLogger(logger);
            llama_backend_init();

            if (!string.IsNullOrWhiteSpace(backendDirectory))
            {
                if (!Directory.Exists(backendDirectory))
                    throw new DirectoryNotFoundException(backendDirectory);

                IntPtr dirPtr = AllocUtf8(backendDirectory);
                try
                {
                    ggml_backend_load_all_from_path(dirPtr);
                }
                finally
                {
                    Marshal.FreeHGlobal(dirPtr);
                }
            }
            else
            {
                ggml_backend_load_all();
            }
        }
        catch (BadImageFormatException ex)
        {
            throw new InvalidOperationException(
                $"Failed to load the llama.cpp native library — architecture mismatch. " +
                $"Process architecture: {RuntimeInformation.ProcessArchitecture}. " +
                $"Backend directory: '{backendDirectory ?? "(none)"}'. " +
                $"Delete the cached runtime at '{LlamaRuntimeInstaller.DefaultInstallRoot}' " +
                $"and restart to trigger a fresh download for the correct architecture.", ex);
        }
    }

    /// <summary>
    /// Shuts down the global llama backend state.
    /// </summary>
    public static void Shutdown()
    {
        llama_backend_free();
    }

    /// <summary>
    /// Registers the managed logger with the native backends when supported.
    /// </summary>
    public static void SetLogger(ILogger? logger)
    {
        s_logger = logger ?? NullLogger.Instance;

        try
        {
            llama_log_set(s_llamaLogCallback, IntPtr.Zero);
        }
        catch (EntryPointNotFoundException)
        {
        }

        try
        {
            ggml_log_set(s_ggmlLogCallback, IntPtr.Zero);
        }
        catch (EntryPointNotFoundException)
        {
        }
    }

    /// <summary>
    /// Loads a GGUF model from disk.
    /// </summary>
    public static Model LoadModel(
        string path,
        bool useMmap = true,
        bool useMlock = false,
        bool checkTensors = false,
        int nGpuLayers = 0)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Model path is empty.", nameof(path));

        if (!File.Exists(path))
            throw new FileNotFoundException("Model file not found.", path);

        var mp = llama_model_default_params();
        mp.n_gpu_layers = nGpuLayers;
        mp.use_mmap = useMmap;
        mp.use_mlock = useMlock;
        mp.check_tensors = checkTensors;

        IntPtr pathPtr = AllocUtf8(path);
        try
        {
            IntPtr model = llama_model_load_from_file(pathPtr, mp);
            if (model == IntPtr.Zero)
                throw new InvalidOperationException(
                    "llama_model_load_from_file returned null. " +
                    "Make sure Init(backendDirectory) was called first, " +
                    "the GGUF exists, and the backend DLLs are in that directory.");
            return new Model(model);
        }
        finally
        {
            Marshal.FreeHGlobal(pathPtr);
        }
    }

    /// <summary>
    /// Frees a previously loaded model.
    /// </summary>
    public static void FreeModel(Model model)
    {
        if (!model.IsNull)
            llama_model_free(model.Ptr);
    }

    /// <summary>
    /// Creates an inference context for the specified model.
    /// </summary>
    public static Context CreateContext(
        Model model,
        int nCtx = 2048,
        int? nBatch = null,
        int? nUbatch = null,
        int? nThreads = null,
        bool embeddings = false,
        bool unifiedKvCache = false,
        float? ropeFreqBase = null,
        float? ropeFreqScale = null,
        bool offloadKvCacheToGpu = true,
        bool flashAttention = false,
        KvCacheQuantization? kvCacheTypeK = null,
        KvCacheQuantization? kvCacheTypeV = null)
    {
        var cp = llama_context_default_params();

        cp.n_ctx = (uint)nCtx;

        int batch = nBatch ?? 512;
        int ubatch = nUbatch ?? Math.Min(batch, 512);

        cp.n_batch = (uint)batch;
        cp.n_ubatch = (uint)ubatch;
        cp.embeddings = embeddings;
        cp.kv_unified = unifiedKvCache;
        cp.offload_kqv = offloadKvCacheToGpu;

        cp.n_threads = nThreads ?? Environment.ProcessorCount;
        cp.n_threads_batch = nThreads ?? Environment.ProcessorCount;

        if (ropeFreqBase is float baseValue)
            cp.rope_freq_base = baseValue;

        if (ropeFreqScale is float scaleValue)
            cp.rope_freq_scale = scaleValue;

        if (flashAttention)
            cp.flash_attn_type = 1;

        if (kvCacheTypeK is KvCacheQuantization cacheTypeK)
            cp.type_k = ToGgmlType(cacheTypeK);

        if (kvCacheTypeV is KvCacheQuantization cacheTypeV)
            cp.type_v = ToGgmlType(cacheTypeV);

        IntPtr ctx = llama_init_from_model(model.Ptr, cp);
        if (ctx == IntPtr.Zero)
            throw new InvalidOperationException("llama_init_from_model returned null.");

        return new Context(ctx);
    }

    private static int ToGgmlType(KvCacheQuantization quantization)
    {
        return quantization switch
        {
            KvCacheQuantization.F16 => 1,
            KvCacheQuantization.Q4_0 => 2,
            KvCacheQuantization.Q4_1 => 3,
            KvCacheQuantization.Q5_0 => 6,
            KvCacheQuantization.Q5_1 => 7,
            KvCacheQuantization.Q8_0 => 8,
            _ => throw new ArgumentOutOfRangeException(nameof(quantization), quantization, null)
        };
    }

    /// <summary>
    /// Frees an inference context.
    /// </summary>
    public static void FreeContext(Context ctx)
    {
        if (!ctx.IsNull)
            llama_free(ctx.Ptr);
    }

    /// <summary>
    /// Returns the model embedding dimension.
    /// </summary>
    public static int GetEmbeddingDimension(Model model)
    {
        int dimension = llama_n_embd(model.Ptr);
        if (dimension <= 0)
            throw new InvalidOperationException("llama_n_embd returned an invalid embedding dimension.");
        return dimension;
    }

    /// <summary>
    /// Reads embeddings for the current sequence as <see cref="float"/> values.
    /// </summary>
    public static float[] GetEmbeddings(Context ctx, Model model, int seqId = 0)
    {
        IntPtr embeddingsPtr;

        try
        {
            embeddingsPtr = llama_get_embeddings_seq(ctx.Ptr, seqId);
        }
        catch (EntryPointNotFoundException)
        {
            embeddingsPtr = IntPtr.Zero;
        }

        if (embeddingsPtr == IntPtr.Zero)
            embeddingsPtr = llama_get_embeddings(ctx.Ptr);

        if (embeddingsPtr == IntPtr.Zero)
            throw new InvalidOperationException("The current context did not expose embeddings. Ensure embeddings are enabled on the context and the model supports them.");

        float[] result = new float[GetEmbeddingDimension(model)];
        Marshal.Copy(embeddingsPtr, result, 0, result.Length);
        return result;
    }

    /// <summary>
    /// Reads embeddings for the current sequence as <see cref="double"/> values.
    /// </summary>
    public static double[] GetEmbeddingsAsDouble(Context ctx, Model model, int seqId = 0)
    {
        float[] embeddings = GetEmbeddings(ctx, model, seqId);
        var result = new double[embeddings.Length];

        for (int i = 0; i < embeddings.Length; i++)
            result[i] = embeddings[i];

        return result;
    }

    /// <summary>
    /// Gets the vocabulary handle for a model.
    /// </summary>
    public static Vocab GetVocab(Model model)
    {
        IntPtr vocab = llama_model_get_vocab(model.Ptr);
        if (vocab == IntPtr.Zero)
            throw new InvalidOperationException("llama_model_get_vocab returned null.");
        return new Vocab(vocab);
    }

    /// <summary>
    /// Tokenizes text using the model vocabulary.
    /// </summary>
    public static int[] Tokenize(Model model, string text, bool addSpecial = true, bool parseSpecial = true)
    {
        Vocab vocab = GetVocab(model);
        IntPtr textPtr = AllocUtf8(text);

        try
        {
            int textLen = Encoding.UTF8.GetByteCount(text);

            int needed = llama_tokenize(
                vocab.Ptr,
                textPtr,
                textLen,
                IntPtr.Zero,
                0,
                addSpecial,
                parseSpecial);

            if (needed >= 0)
                throw new InvalidOperationException("Expected negative probe result from llama_tokenize.");

            int count = -needed;
            int[] tokens = new int[count];

            unsafe
            {
                fixed (int* pTokens = tokens)
                {
                    int written = llama_tokenize(
                        vocab.Ptr,
                        textPtr,
                        textLen,
                        (IntPtr)pTokens,
                        tokens.Length,
                        addSpecial,
                        parseSpecial);

                    if (written < 0)
                        throw new InvalidOperationException("llama_tokenize failed on second pass.");

                    if (written != tokens.Length)
                        Array.Resize(ref tokens, written);
                }
            }

            return tokens;
        }
        finally
        {
            Marshal.FreeHGlobal(textPtr);
        }
    }

    /// <summary>
    /// Creates a native batch from token ids.
    /// </summary>
    public static Batch CreateBatch(int[] tokens)
    {
        if (tokens == null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Length == 0) throw new ArgumentException("tokens must not be empty", nameof(tokens));

        GCHandle handle = GCHandle.Alloc(tokens, GCHandleType.Pinned);
        try
        {
            IntPtr tokenPtr = handle.AddrOfPinnedObject();
            var native = llama_batch_get_one(tokenPtr, tokens.Length);
            return new Batch(handle, native);
        }
        catch
        {
            handle.Free();
            throw;
        }
    }

    /// <summary>
    /// Decodes a batch into the provided context.
    /// </summary>
    public static int Decode(Context ctx, Batch batch)
    {
        if (batch == null) throw new ArgumentNullException(nameof(batch));
        return llama_decode(ctx.Ptr, batch.Native);
    }

    /// <summary>
    /// Frees a managed batch wrapper.
    /// </summary>
    public static void FreeBatch(Batch batch)
    {
        batch?.Dispose();
    }

    /// <summary>
    /// Attempts to remove a cache range from the KV cache.
    /// </summary>
    public static bool TryRemoveCacheRange(Context ctx, int seqId, int startPos, int endPos = int.MaxValue)
    {
        try
        {
            return llama_kv_cache_seq_rm(ctx.Ptr, seqId, startPos, endPos);
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    /// <summary>
    /// Samples the highest-logit token.
    /// </summary>
    public static unsafe int SampleGreedy(Context ctx, Model model)
    {
        Vocab vocab = GetVocab(model);
        int nVocab = llama_vocab_n_tokens(vocab.Ptr);
        IntPtr logitsPtr = llama_get_logits_ith(ctx.Ptr, -1);

        if (logitsPtr == IntPtr.Zero)
            throw new InvalidOperationException("llama_get_logits_ith returned null.");

        float* logits = (float*)logitsPtr;

        int bestToken = 0;
        float bestLogit = logits[0];

        for (int i = 1; i < nVocab; i++)
        {
            if (logits[i] > bestLogit)
            {
                bestLogit = logits[i];
                bestToken = i;
            }
        }

        return bestToken;
    }

    /// <summary>
    /// Samples a token using the configured inference options.
    /// </summary>
    public static unsafe int Sample(
        Context ctx,
        Model model,
        InferenceOptions options,
        IReadOnlyDictionary<int, int>? tokenCounts,
        Random random)
    {
        float temperature = options.Temperature.GetValueOrDefault();
        float presencePenalty = options.PresencePenalty.GetValueOrDefault();
        float frequencyPenalty = options.FrequencyPenalty.GetValueOrDefault();
        int topK = options.TopK.GetValueOrDefault();
        float topP = options.TopP.GetValueOrDefault(1.0f);

        if (temperature <= 0)
            return SampleGreedy(ctx, model);

        Vocab vocab = GetVocab(model);
        int nVocab = llama_vocab_n_tokens(vocab.Ptr);
        IntPtr logitsPtr = llama_get_logits_ith(ctx.Ptr, -1);

        if (logitsPtr == IntPtr.Zero)
            throw new InvalidOperationException("llama_get_logits_ith returned null.");

        float* logits = (float*)logitsPtr;
        var candidates = new List<(int Token, double Logit)>(nVocab);

        for (int token = 0; token < nVocab; token++)
        {
            double adjusted = logits[token];

            if (tokenCounts is not null && tokenCounts.TryGetValue(token, out int count))
            {
                if (count > 0)
                    adjusted -= presencePenalty;

                adjusted -= frequencyPenalty * count;
            }

            adjusted /= Math.Max(temperature, 1e-6f);
            candidates.Add((token, adjusted));
        }

        candidates.Sort((a, b) => b.Logit.CompareTo(a.Logit));

        if (topK > 0 && topK < candidates.Count)
            candidates.RemoveRange(topK, candidates.Count - topK);

        double maxLogit = candidates[0].Logit;
        double sum = 0;
        var weighted = new List<(int Token, double Probability)>(candidates.Count);
        foreach (var candidate in candidates)
        {
            double probability = Math.Exp(candidate.Logit - maxLogit);
            weighted.Add((candidate.Token, probability));
            sum += probability;
        }

        for (int i = 0; i < weighted.Count; i++)
            weighted[i] = (weighted[i].Token, weighted[i].Probability / sum);

        if (topP > 0 && topP < 1)
        {
            double cumulative = 0;
            int keep = 0;

            for (; keep < weighted.Count; keep++)
            {
                cumulative += weighted[keep].Probability;
                if (cumulative >= topP)
                {
                    keep++;
                    break;
                }
            }

            keep = Math.Max(1, Math.Min(keep, weighted.Count));
            weighted.RemoveRange(keep, weighted.Count - keep);

            double renormalized = weighted.Sum(x => x.Probability);
            for (int i = 0; i < weighted.Count; i++)
                weighted[i] = (weighted[i].Token, weighted[i].Probability / renormalized);
        }

        double sample = random.NextDouble();
        double running = 0;
        foreach (var candidate in weighted)
        {
            running += candidate.Probability;
            if (sample <= running)
                return candidate.Token;
        }

        return weighted[^1].Token;
    }

    /// <summary>
    /// Handle to a native llama_sampler_chain.
    /// </summary>
    public readonly record struct SamplerChain(IntPtr Ptr)
    {
        public bool IsNull => Ptr == IntPtr.Zero;
    }

    /// <summary>
    /// Creates a native sampler chain configured from the given inference options.
    /// The chain holds no managed references and must be freed with
    /// <see cref="FreeSamplerChain"/> when generation is complete.
    /// </summary>
    public static SamplerChain CreateSamplerChain(InferenceOptions options, Random random, Model? model = null)
    {
        float temperature = options.Temperature.GetValueOrDefault();
        IntPtr chain = llama_sampler_chain_init(llama_sampler_chain_default_params());

        if (!string.IsNullOrWhiteSpace(options.Grammar) && model.HasValue)
        {
            IntPtr grammarPtr = AllocUtf8(options.Grammar);
            IntPtr rootPtr    = AllocUtf8("root");
            Vocab  vocab      = GetVocab(model.Value);
            try
            {
                llama_sampler_chain_add(chain, llama_sampler_init_grammar(vocab.Ptr, grammarPtr, rootPtr));
            }
            finally
            {
                Marshal.FreeHGlobal(grammarPtr);
                Marshal.FreeHGlobal(rootPtr);
            }
        }

        if (temperature <= 0)
        {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            return new SamplerChain(chain);
        }

        float presencePenalty  = options.PresencePenalty.GetValueOrDefault();
        float frequencyPenalty = options.FrequencyPenalty.GetValueOrDefault();
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(64, 1.0f, frequencyPenalty, presencePenalty));

        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));

        int topK = options.TopK.GetValueOrDefault();
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK > 0 ? topK : 0));

        float topP = options.TopP.GetValueOrDefault(1.0f);
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));

        uint seed = options.Seed.HasValue ? (uint)options.Seed.Value : (uint)random.Next();
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

        return new SamplerChain(chain);
    }

    /// <summary>
    /// Samples the next token using a native sampler chain and accepts it,
    /// updating the chain's internal penalty window.
    /// </summary>
    public static int SampleWithChain(SamplerChain chain, Context ctx)
    {
        int token = llama_sampler_sample(chain.Ptr, ctx.Ptr, -1);
        llama_sampler_accept(chain.Ptr, token);
        return token;
    }

    /// <summary>
    /// Frees a sampler chain created with <see cref="CreateSamplerChain"/>.
    /// </summary>
    public static void FreeSamplerChain(SamplerChain chain)
    {
        if (!chain.IsNull)
            llama_sampler_free(chain.Ptr);
    }

    /// <summary>
    /// Converts a token id back into text.
    /// </summary>
    public static string TokenToString(Model model, int token, bool special = true)
    {
        Vocab vocab = GetVocab(model);

        byte[] buf = new byte[256];
        int n = llama_token_to_piece(vocab.Ptr, token, buf, buf.Length, 0, special);

        if (n < 0)
        {
            buf = new byte[-n];
            n = llama_token_to_piece(vocab.Ptr, token, buf, buf.Length, 0, special);
        }

        if (n < 0)
            throw new InvalidOperationException("llama_token_to_piece failed.");

        return Encoding.UTF8.GetString(buf, 0, n);
    }

    /// <summary>
    /// Writes the raw UTF-8 bytes for a token into <paramref name="buf"/> and returns the byte count.
    /// Use this with a stateful <see cref="System.Text.Decoder"/> to correctly handle multi-byte
    /// characters that are split across consecutive tokens.
    /// </summary>
    public static int TokenToBytes(Model model, int token, byte[] buf, bool special = true)
    {
        Vocab vocab = GetVocab(model);
        int n = llama_token_to_piece(vocab.Ptr, token, buf, buf.Length, 0, special);
        if (n < 0)
            throw new InvalidOperationException($"llama_token_to_piece failed: buffer too small (need {-n} bytes).");
        return n;
    }

    /// <summary>
    /// Returns whether a token terminates generation.
    /// </summary>
    public static bool IsEndOfGeneration(Model model, int token)
    {
        Vocab vocab = GetVocab(model);
        return llama_vocab_is_eog(vocab.Ptr, token);
    }

    private static void HandleLlamaLog(int level, IntPtr text, IntPtr userData)
    {
        WriteNativeLog("llama", level, text);
    }

    private static void HandleGgmlLog(int level, IntPtr text, IntPtr userData)
    {
        WriteNativeLog("ggml", level, text);
    }

    private static void WriteNativeLog(string category, int level, IntPtr text)
    {
        string? message = Marshal.PtrToStringUTF8(text);
        if (string.IsNullOrEmpty(message))
            return;

        s_logger.Log(MapLogLevel(level), category, message);
    }

    private static LogLevel MapLogLevel(int nativeLevel)
    {
        return nativeLevel switch
        {
            2 => LogLevel.Warning,
            3 => LogLevel.Error,
            4 => LogLevel.Debug,
            5 => LogLevel.Trace,
            _ => LogLevel.Information
        };
    }

    public static class Vision
    {
        private const string MTMD_LIB = "mtmd";

        // If your backend folder has mtmd-helper.dll, change this to "mtmd-helper".
        private const string MTMD_HELPER_LIB = "mtmd";

        private static void LogDiagnostic(string message)
        {
            s_logger.Log(LogLevel.Debug, "vision", message);
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

        public readonly struct Context
        {
            public readonly IntPtr Ptr;
            public Context(IntPtr ptr) => Ptr = ptr;
            public bool IsNull => Ptr == IntPtr.Zero;
        }

        public readonly struct Bitmap
        {
            public readonly IntPtr Ptr;
            public Bitmap(IntPtr ptr) => Ptr = ptr;
            public bool IsNull => Ptr == IntPtr.Zero;
        }

        public readonly struct InputChunks
        {
            public readonly IntPtr Ptr;
            public InputChunks(IntPtr ptr) => Ptr = ptr;
            public bool IsNull => Ptr == IntPtr.Zero;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct mtmd_context_params
        {
            [MarshalAs(UnmanagedType.I1)] public bool use_gpu;
            [MarshalAs(UnmanagedType.I1)] public bool print_timings;
            public int n_threads;
            public IntPtr image_marker;
            public IntPtr media_marker;
            public int flash_attn_type;
            [MarshalAs(UnmanagedType.I1)] public bool warmup;
            public int image_min_tokens;
            public int image_max_tokens;
            public IntPtr cb_eval;
            public IntPtr cb_eval_user_data;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct mtmd_input_text
        {
            public IntPtr text;
            [MarshalAs(UnmanagedType.I1)] public bool add_special;
            [MarshalAs(UnmanagedType.I1)] public bool parse_special;
        }

        // Returns the native placeholder token that mtmd expects in multimodal prompts.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtmd_default_marker();

        // Returns the native default mtmd parameters.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern mtmd_context_params mtmd_context_params_default();

        // Creates an mtmd context using the projector and the already loaded text model.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtmd_init_from_file(
            IntPtr mmproj_fname,
            IntPtr text_model,
            mtmd_context_params ctx_params);

        // Releases the mtmd context.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern void mtmd_free(IntPtr ctx);

        // Confirms that the created mtmd context actually supports vision.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool mtmd_support_vision(IntPtr ctx);

        // Allocates a chunk container used by mtmd tokenization.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtmd_input_chunks_init();

        // Releases a chunk container.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern void mtmd_input_chunks_free(IntPtr chunks);

        // Tokenizes prompt text plus bitmap handles into mtmd input chunks.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtmd_tokenize(
            IntPtr ctx,
            IntPtr outputChunks,
            ref mtmd_input_text text,
            IntPtr[] bitmaps,
            UIntPtr n_bitmaps);

        // Decodes an image buffer into an mtmd bitmap handle.
        [DllImport(MTMD_HELPER_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtmd_helper_bitmap_init_from_buf(
            IntPtr ctx,
            byte[] buf,
            UIntPtr len);

        // Releases a decoded bitmap.
        [DllImport(MTMD_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern void mtmd_bitmap_free(IntPtr bitmap);

        // Evaluates multimodal chunks directly into a llama context.
        [DllImport(MTMD_HELPER_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtmd_helper_eval_chunks(
            IntPtr ctx,
            IntPtr lctx,
            IntPtr chunks,
            int n_past,
            int seq_id,
            int n_batch,
            [MarshalAs(UnmanagedType.I1)] bool logits_last,
            out int new_n_past);

        // Returns the chunk token count for diagnostics and batching.
        [DllImport(MTMD_HELPER_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern UIntPtr mtmd_helper_get_n_tokens(IntPtr chunks);

        // Returns the chunk position count used when picking an effective batch size.
        [DllImport(MTMD_HELPER_LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtmd_helper_get_n_pos(IntPtr chunks);

        /// <summary>
        /// Gets the marker string that should replace the model's rendered image token before mtmd evaluation.
        /// </summary>
        public static string DefaultMarker
        {
            get
            {
                IntPtr ptr = mtmd_default_marker();
                return Marshal.PtrToStringUTF8(ptr) ?? "<__media__>";
            }
        }

        /// <summary>
        /// Creates a vision context bound to an already loaded llama model.
        /// </summary>
        public static Context Load(
            Model model,
            string mmprojPath,
            bool useGpu = false,
            bool printTimings = false,
            int nThreads = 0,
            string? mediaMarker = null,
            bool warmup = false,
            int imageMinTokens = 0,
            int imageMaxTokens = 0,
            int flashAttnType = 0)
        {
            if (model.IsNull)
                throw new ArgumentException("Model is null.", nameof(model));
            if (string.IsNullOrWhiteSpace(mmprojPath))
                throw new ArgumentException("mmproj path is empty.", nameof(mmprojPath));

            IntPtr mmprojPtr = AllocUtf8(mmprojPath);
            IntPtr markerPtr = IntPtr.Zero;

            try
            {
                var p = mtmd_context_params_default();
                p.use_gpu = useGpu;
                p.print_timings = printTimings;
                p.n_threads = nThreads > 0 ? nThreads : Environment.ProcessorCount;
                p.warmup = warmup;
                p.image_min_tokens = imageMinTokens;
                p.image_max_tokens = imageMaxTokens;
                p.flash_attn_type = flashAttnType;

                if (!string.IsNullOrWhiteSpace(mediaMarker))
                {
                    markerPtr = AllocUtf8(mediaMarker);
                    p.media_marker = markerPtr;
                }

                IntPtr ctx = mtmd_init_from_file(mmprojPtr, model.Ptr, p);
                if (ctx == IntPtr.Zero)
                    throw new InvalidOperationException(
                        "mtmd_init_from_file returned null. Check mtmd DLLs and mmproj/model compatibility.");

                if (!mtmd_support_vision(ctx))
                {
                    mtmd_free(ctx);
                    throw new InvalidOperationException("Loaded mtmd context does not report vision support.");
                }

                return new Context(ctx);
            }
            finally
            {
                if (markerPtr != IntPtr.Zero)
                    Marshal.FreeHGlobal(markerPtr);

                Marshal.FreeHGlobal(mmprojPtr);
            }
        }

        /// <summary>
        /// Releases a vision context.
        /// </summary>
        public static void Free(Context ctx)
        {
            if (!ctx.IsNull)
                mtmd_free(ctx.Ptr);
        }

        /// <summary>
        /// Decodes a base64-encoded image into an mtmd bitmap handle.
        /// </summary>
        public static Bitmap CreateBitmapFromBase64(Context vision, string base64)
        {
            if (vision.IsNull)
                throw new ArgumentException("Vision context is null.", nameof(vision));
            if (string.IsNullOrWhiteSpace(base64))
                throw new ArgumentException("Base64 image is empty.", nameof(base64));

            int comma = base64.IndexOf(',');
            if (base64.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && comma >= 0)
                base64 = base64[(comma + 1)..];

            byte[] bytes = Convert.FromBase64String(base64);
            IntPtr bmp = mtmd_helper_bitmap_init_from_buf(vision.Ptr, bytes, (UIntPtr)bytes.Length);
            if (bmp == IntPtr.Zero)
                throw new InvalidOperationException("mtmd_helper_bitmap_init_from_buf returned null.");

            LogDiagnostic($"bitmap decoded: bytes={bytes.Length}, ptr=0x{bmp.ToString("X")}");

            return new Bitmap(bmp);
        }

        /// <summary>
        /// Releases a decoded bitmap.
        /// </summary>
        public static void FreeBitmap(Bitmap bmp)
        {
            if (!bmp.IsNull)
                mtmd_bitmap_free(bmp.Ptr);
        }

        /// <summary>
        /// Tokenizes a prompt and its attached images into mtmd chunks.
        /// </summary>
        public static InputChunks Tokenize(
            Context vision,
            string prompt,
            IReadOnlyList<Bitmap> bitmaps,
            bool addSpecial,
            bool parseSpecial = true)
        {
            if (vision.IsNull)
                throw new ArgumentException("Vision context is null.", nameof(vision));
            if (prompt is null)
                throw new ArgumentNullException(nameof(prompt));

            IntPtr textPtr = AllocUtf8(prompt);
            IntPtr chunks = IntPtr.Zero;
            int promptMarkerCount = CountOccurrences(prompt, DefaultMarker);

            try
            {
                var text = new mtmd_input_text
                {
                    text = textPtr,
                    add_special = addSpecial,
                    parse_special = parseSpecial
                };

                chunks = mtmd_input_chunks_init();
                if (chunks == IntPtr.Zero)
                    throw new InvalidOperationException("mtmd_input_chunks_init returned null.");

                IntPtr[] bitmapPtrs = new IntPtr[bitmaps.Count];
                for (int i = 0; i < bitmaps.Count; i++)
                {
                    if (bitmaps[i].IsNull)
                        throw new ArgumentException($"Bitmap at index {i} is null.", nameof(bitmaps));

                    bitmapPtrs[i] = bitmaps[i].Ptr;
                }

                int rc = mtmd_tokenize(
                    vision.Ptr,
                    chunks,
                    ref text,
                    bitmapPtrs,
                    (UIntPtr)bitmapPtrs.Length);

                if (rc != 0)
                    throw new InvalidOperationException($"mtmd_tokenize failed with code {rc}.");

                UIntPtr chunkTokens = mtmd_helper_get_n_tokens(chunks);
                int chunkPos = mtmd_helper_get_n_pos(chunks);
                LogDiagnostic(
                    $"mtmd_tokenize ok: markers={promptMarkerCount}, bitmaps={bitmapPtrs.Length}, chunk_tokens={(ulong)chunkTokens}, chunk_pos={chunkPos}, add_special={addSpecial}, parse_special={parseSpecial}");

                return new InputChunks(chunks);
            }
            catch
            {
                if (chunks != IntPtr.Zero)
                    mtmd_input_chunks_free(chunks);

                throw;
            }
            finally
            {
                Marshal.FreeHGlobal(textPtr);
            }
        }

        /// <summary>
        /// Releases tokenized mtmd chunks.
        /// </summary>
        public static void FreeChunks(InputChunks chunks)
        {
            if (!chunks.IsNull)
                mtmd_input_chunks_free(chunks.Ptr);
        }

        /// <summary>
        /// Helper that converts base64 images, tokenizes the multimodal prompt, and evaluates it into a llama context.
        /// </summary>
        public static void EvalPromptWithBase64Images(
            Context vision,
            Llama.Context llamaCtx,
            string prompt,
            IReadOnlyList<string> base64Images,
            ref int nPast,
            int seqId = 0,
            int nBatch = 4096,
            bool addSpecial = true,
            bool parseSpecial = true,
            bool logitsLast = true)
        {
            if (vision.IsNull)
                throw new ArgumentException("Vision context is null.", nameof(vision));
            if (llamaCtx.IsNull)
                throw new ArgumentException("Llama context is null.", nameof(llamaCtx));
            if (base64Images is null)
                throw new ArgumentNullException(nameof(base64Images));

            var bitmaps = new List<Bitmap>(base64Images.Count);
            InputChunks chunks = default;

            try
            {
                int promptMarkerCount = CountOccurrences(prompt, DefaultMarker);
                LogDiagnostic($"eval start: markers={promptMarkerCount}, images={base64Images.Count}, n_past={nPast}, n_batch={nBatch}, seq_id={seqId}");

                foreach (string image in base64Images)
                    bitmaps.Add(CreateBitmapFromBase64(vision, image));

                chunks = Tokenize(vision, prompt, bitmaps, addSpecial, parseSpecial);

                int chunkPos = mtmd_helper_get_n_pos(chunks.Ptr);
                if (chunkPos <= 0)
                    throw new InvalidOperationException("mtmd produced zero positions for the input chunks.");

                UIntPtr chunkTokens = mtmd_helper_get_n_tokens(chunks.Ptr);
                LogDiagnostic($"eval chunks: chunk_tokens={(ulong)chunkTokens}, chunk_pos={chunkPos}, bitmaps={bitmaps.Count}");

                int rc = mtmd_helper_eval_chunks(
                    vision.Ptr,
                    llamaCtx.Ptr,
                    chunks.Ptr,
                    nPast,
                    seqId,
                    nBatch,
                    logitsLast,
                    out int newNPast);

                if (rc != 0)
                    throw new InvalidOperationException(
                        $"mtmd_helper_eval_chunks failed with code {rc}. n_past={nPast}, chunk_pos={chunkPos}, chunk_tokens={(ulong)chunkTokens}, n_batch={nBatch}");

                nPast = newNPast;
                LogDiagnostic($"eval complete: new_n_past={newNPast}, chunk_tokens={(ulong)chunkTokens}, chunk_pos={chunkPos}");
            }
            finally
            {
                if (!chunks.IsNull)
                    FreeChunks(chunks);

                foreach (var bmp in bitmaps)
                    FreeBitmap(bmp);
            }
        }
    }
}