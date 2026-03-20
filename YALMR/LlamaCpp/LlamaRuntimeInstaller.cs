using System.IO.Compression;
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace YALMR.LlamaCpp;

/// <summary>
/// The accelerator backend to use for llama.cpp inference.
/// </summary>
public enum LlamaBackend
{
    /// <summary>CPU-only inference (AVX2 preferred).</summary>
    Cpu,
    /// <summary>NVIDIA CUDA GPU inference.</summary>
    Cuda,
    /// <summary>Vulkan GPU inference (AMD / Intel / NVIDIA).</summary>
    Vulkan,
}

/// <summary>
/// Downloads and installs llama.cpp native runtime binaries from GitHub releases on demand.
/// </summary>
public static class LlamaRuntimeInstaller
{
    private const string Owner = "ggml-org";
    private const string Repo = "llama.cpp";

    private static string ReleaseApiUrl(string? releaseTag) =>
        releaseTag is null
            ? $"https://api.github.com/repos/{Owner}/{Repo}/releases/latest"
            : $"https://api.github.com/repos/{Owner}/{Repo}/releases/tags/{releaseTag}";

    private static readonly string NativeLib =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "llama.dll" : "libllama.so";

    /// <summary>
    /// Default directory under which backend-specific sub-directories are installed.
    /// Windows: <c>%LOCALAPPDATA%\YALMR\llama-runtime</c>
    /// Linux/macOS: <c>~/.local/share/YALMR/llama-runtime</c>
    /// </summary>
    public static string DefaultInstallRoot { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "YALMR",
        "llama-runtime");

    /// <summary>
    /// Returns the path to an already-installed runtime for <paramref name="backend"/>,
    /// or <see langword="null"/> when nothing is installed yet.
    /// When <paramref name="releaseTag"/> is supplied the match is exact (e.g. <c>"b8269"</c>);
    /// otherwise the newest installed version for that backend is returned.
    /// </summary>
    public static string? FindInstalled(LlamaBackend backend, string? releaseTag = null, string? installRoot = null)
    {
        string root = installRoot ?? DefaultInstallRoot;
        if (!Directory.Exists(root))
            return null;

        string prefix = BackendPrefix(backend, CurrentArchToken());

        if (releaseTag is not null)
        {
            string exact = Path.Combine(root, $"{prefix}-{releaseTag}");
            return Directory.Exists(exact) && File.Exists(Path.Combine(exact, NativeLib)) ? exact : null;
        }

        return Directory
            .EnumerateDirectories(root)
            .Where(d => Path.GetFileName(d).StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(d => d, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault(d => File.Exists(Path.Combine(d, NativeLib)));
    }

    /// <summary>
    /// Ensures the llama.cpp runtime for <paramref name="backend"/> is available,
    /// downloading and extracting from GitHub when needed.
    /// </summary>
    /// <param name="backend">Desired inference backend.</param>
    /// <param name="cudaVersion">
    /// Preferred CUDA version, e.g. <c>"12.4"</c>. When <see langword="null"/> the
    /// highest available CUDA 12.x release asset is chosen automatically.
    /// </param>
    /// <param name="releaseTag">
    /// Pin to a specific release tag, e.g. <c>"b8269"</c>. When <see langword="null"/> the
    /// latest published release is used.
    /// </param>
    public static async Task<string> EnsureInstalledAsync(
        LlamaBackend backend = LlamaBackend.Cpu,
        string? cudaVersion = null,
        string? releaseTag = null,
        string? installRoot = null,
        bool forceReinstall = false,
        IProgress<(string message, double percent)>? progress = null,
        CancellationToken ct = default)
    {
        string root = installRoot ?? DefaultInstallRoot;

        if (!forceReinstall)
        {
            string? existing = FindInstalled(backend, releaseTag, root);
            if (existing is not null)
            {
                progress?.Report(($"Using installed runtime: {existing}", 100));
                return existing;
            }
        }

        bool isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        string archToken = CurrentArchToken();

        string fetchLabel = releaseTag is not null ? $"release {releaseTag}" : "latest release";
        progress?.Report(($"Fetching llama.cpp {fetchLabel} info from GitHub...", 0));

        using var http = new HttpClient();
        http.DefaultRequestHeaders.Add("User-Agent", "YALMR/1.0");

        var release = await http.GetFromJsonAsync<GitHubRelease>(ReleaseApiUrl(releaseTag), ct)
            ?? throw new InvalidOperationException($"Failed to fetch llama.cpp {fetchLabel} from GitHub.");

        string tag = release.TagName ?? "unknown";

        var asset = PickAsset(release.Assets, backend, cudaVersion, isWindows, archToken)
            ?? throw new InvalidOperationException(
                $"No suitable llama.cpp asset found for backend={backend}, " +
                $"platform={(isWindows ? "Windows" : "Linux")}, cudaVersion={cudaVersion ?? "any"}. " +
                $"Available assets: {string.Join(", ", release.Assets.Select(a => a.Name))}");

        progress?.Report(($"Downloading {asset.Name} ({tag})...", 3));

        string tempArchive = Path.Combine(Path.GetTempPath(), asset.Name ?? $"llama-{Guid.NewGuid()}.tmp");

        try
        {
            await DownloadAsync(http, asset.BrowserDownloadUrl!, tempArchive, progress, ct);

            progress?.Report(("Extracting...", 92));

            string tempDir = Path.Combine(Path.GetTempPath(), $"llama-extract-{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDir);

            try
            {
                await ExtractAsync(tempArchive, tempDir, ct);

                string binDir = FindBinDir(tempDir)
                    ?? throw new InvalidOperationException(
                        $"Could not find {NativeLib} in the extracted archive. " +
                        $"Check that the downloaded asset is a valid llama.cpp release.");

                string installDir = Path.Combine(root, $"{BackendPrefix(backend, archToken)}-{tag}");
                Directory.CreateDirectory(root);

                if (Directory.Exists(installDir))
                    Directory.Delete(installDir, recursive: true);

                Directory.Move(binDir, installDir);

                progress?.Report(($"Runtime installed: {installDir}", 100));
                return installDir;
            }
            finally
            {
                if (Directory.Exists(tempDir))
                    Directory.Delete(tempDir, recursive: true);
            }
        }
        finally
        {
            if (File.Exists(tempArchive))
                File.Delete(tempArchive);
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private static async Task DownloadAsync(
        HttpClient http,
        string url,
        string destPath,
        IProgress<(string message, double percent)>? progress,
        CancellationToken ct)
    {
        using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();

        long total = response.Content.Headers.ContentLength ?? -1;
        await using var src = await response.Content.ReadAsStreamAsync(ct);
        await using var dst = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, FileOptions.Asynchronous);

        byte[] buffer = new byte[81920];
        long downloaded = 0;
        int read;

        while ((read = await src.ReadAsync(buffer, ct)) > 0)
        {
            await dst.WriteAsync(buffer.AsMemory(0, read), ct);
            downloaded += read;

            if (total > 0)
            {
                double pct = 3 + downloaded * 87.0 / total;
                string msg = $"Downloading... {downloaded / 1_048_576.0:F1} / {total / 1_048_576.0:F1} MB";
                progress?.Report((msg, pct));
            }
        }
    }

    private static async Task ExtractAsync(string archivePath, string destDir, CancellationToken ct)
    {
        if (archivePath.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            await Task.Run(() => ZipFile.ExtractToDirectory(archivePath, destDir, overwriteFiles: true), ct);
            return;
        }

        if (archivePath.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase) ||
            archivePath.EndsWith(".tgz", StringComparison.OrdinalIgnoreCase))
        {
            await using var fs = File.OpenRead(archivePath);
            await using var gz = new GZipStream(fs, CompressionMode.Decompress);
            await System.Formats.Tar.TarFile.ExtractToDirectoryAsync(gz, destDir, overwriteFiles: true, ct);
            return;
        }

        throw new NotSupportedException($"Unsupported archive format: {Path.GetFileName(archivePath)}");
    }

    /// <summary>
    /// Finds the directory within <paramref name="root"/> that directly contains the native lib.
    /// Handles both flat zips (DLLs at archive root) and zips with a single top-level folder.
    /// </summary>
    private static string? FindBinDir(string root)
    {
        if (File.Exists(Path.Combine(root, NativeLib)))
            return root;

        foreach (string sub in Directory.EnumerateDirectories(root))
            if (File.Exists(Path.Combine(sub, NativeLib)))
                return sub;

        foreach (string file in Directory.EnumerateFiles(root, NativeLib, SearchOption.AllDirectories))
            return Path.GetDirectoryName(file);

        return null;
    }

    private static GitHubAsset? PickAsset(
        IReadOnlyList<GitHubAsset> assets,
        LlamaBackend backend,
        string? cudaVersion,
        bool isWindows,
        string arch)
    {
        string os = isWindows ? "win" : "ubuntu";

        switch (backend)
        {
            case LlamaBackend.Cuda:
            {
                var cudaAssets = assets
                    .Where(a => Has(a, os) && Has(a, arch) && Has(a, "cuda") && !Has(a, "vulkan") && IsArchive(a)
                             && !(a.Name?.StartsWith("cudart", StringComparison.OrdinalIgnoreCase) ?? false))
                    .ToList();

                if (cudaVersion is not null)
                {
                    // Exact substring match (e.g. "12.4" in asset name)
                    var exact = cudaAssets.FirstOrDefault(a => Has(a, cudaVersion));
                    if (exact is not null)
                        return exact;

                    // No exact match — pick nearest compatible version:
                    // same major, highest minor ≤ requested; else lowest minor > requested
                    if (TryParseMajorMinor(cudaVersion, out int reqMajor, out int reqMinor))
                    {
                        var versioned = cudaAssets
                            .Select(a => (Asset: a, Ver: ExtractCudaVersion(a.Name)))
                            .Where(x => x.Ver.HasValue)
                            .Select(x => (x.Asset, x.Ver!.Value.Major, x.Ver!.Value.Minor))
                            .ToList();

                        var below = versioned
                            .Where(x => x.Major == reqMajor && x.Minor <= reqMinor)
                            .OrderByDescending(x => x.Minor)
                            .FirstOrDefault();
                        if (below.Asset is not null)
                            return below.Asset;

                        var above = versioned
                            .Where(x => x.Major == reqMajor && x.Minor > reqMinor)
                            .OrderBy(x => x.Minor)
                            .FirstOrDefault();
                        if (above.Asset is not null)
                            return above.Asset;
                    }
                }

                // Prefer highest CUDA 12.x, then any CUDA
                return cudaAssets
                    .Where(a => Has(a, "12."))
                    .OrderByDescending(a => a.Name, StringComparer.OrdinalIgnoreCase)
                    .FirstOrDefault()
                    ?? cudaAssets
                       .OrderByDescending(a => a.Name, StringComparer.OrdinalIgnoreCase)
                       .FirstOrDefault();
            }

            case LlamaBackend.Vulkan:
                return assets.FirstOrDefault(a => Has(a, os) && Has(a, arch) && Has(a, "vulkan") && IsArchive(a));

            default: // Cpu — prefer avx2, then avx, then noavx, then any non-GPU asset
                return
                    assets.FirstOrDefault(a => Has(a, os) && Has(a, arch) && Has(a, "avx2") && !Has(a, "cuda") && !Has(a, "vulkan") && IsArchive(a)) ??
                    assets.FirstOrDefault(a => Has(a, os) && Has(a, arch) && Has(a, "avx")  && !Has(a, "cuda") && !Has(a, "vulkan") && IsArchive(a)) ??
                    assets.FirstOrDefault(a => Has(a, os) && Has(a, arch) && !Has(a, "cuda") && !Has(a, "vulkan") && !Has(a, "opencl") && !Has(a, "sycl") && IsArchive(a));
        }
    }

    private static bool Has(GitHubAsset a, string value) =>
        a.Name?.Contains(value, StringComparison.OrdinalIgnoreCase) == true;

    private static bool IsArchive(GitHubAsset a) =>
        a.Name?.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) == true ||
        a.Name?.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase) == true ||
        a.Name?.EndsWith(".tgz", StringComparison.OrdinalIgnoreCase) == true;

    // Extracts the CUDA version embedded in an asset name, e.g. "cuda-12.4" → (12, 4).
    private static (int Major, int Minor)? ExtractCudaVersion(string? name)
    {
        if (name is null) return null;
        int idx = name.IndexOf("cuda-", StringComparison.OrdinalIgnoreCase);
        if (idx < 0) return null;
        int start = idx + 5;
        int end = start;
        while (end < name.Length && (char.IsDigit(name[end]) || name[end] == '.'))
            end++;
        string ver = name[start..end];
        int dot = ver.IndexOf('.');
        if (dot < 0) return null;
        return int.TryParse(ver[..dot], out int major) && int.TryParse(ver[(dot + 1)..], out int minor)
            ? (major, minor)
            : null;
    }

    private static bool TryParseMajorMinor(string version, out int major, out int minor)
    {
        major = minor = 0;
        int dot = version.IndexOf('.');
        return dot > 0
            && int.TryParse(version[..dot], out major)
            && int.TryParse(version[(dot + 1)..], out minor);
    }

    private static string BackendPrefix(LlamaBackend backend, string arch) => $"{backend switch
    {
        LlamaBackend.Cuda   => "cuda",
        LlamaBackend.Vulkan => "vulkan",
        _                   => "cpu",
    }}-{arch}";

    private static string CurrentArchToken() => RuntimeInformation.ProcessArchitecture switch
    {
        Architecture.X64   => "x64",
        Architecture.Arm64 => "arm64",
        Architecture.X86   => "x86",
        _                  => "x64",
    };

    // -------------------------------------------------------------------------
    // GitHub API models
    // -------------------------------------------------------------------------

    private sealed class GitHubRelease
    {
        [JsonPropertyName("tag_name")]
        public string? TagName { get; set; }

        [JsonPropertyName("assets")]
        public List<GitHubAsset> Assets { get; set; } = [];
    }

    private sealed class GitHubAsset
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("browser_download_url")]
        public string? BrowserDownloadUrl { get; set; }

        [JsonPropertyName("size")]
        public long Size { get; set; }
    }
}
