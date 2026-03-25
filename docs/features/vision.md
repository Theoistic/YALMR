---
layout: page
title: Vision
permalink: /features/vision/
parent: Features
nav_order: 4
---

# Vision

YALMR supports multimodal models that can process both text and images. Images are attached to messages as `ImagePart` values alongside `TextPart` values, and a separate multimodal projector (mmproj) model encodes them into the token space.

---

## Requirements

- A multimodal GGUF model (e.g. LLaVA, Moondream, Qwen-VL, or similar)
- The corresponding mmproj GGUF file

---

## Session setup

Provide the `MmprojPath` option when creating the session:

```csharp
await using var session = await Session.CreateAsync(new SessionOptions
{
    BackendDirectory = backendDir,
    ModelPath        = "path/to/vision-model.gguf",
    MmprojPath       = "path/to/mmproj.gguf",    // multimodal projector
    ToolRegistry     = new ToolRegistry(),
    Compaction       = new ConversationCompactionOptions(MaxInputTokens: 8192),
});
```

---

## Sending images

### From a file path

```csharp
var message = new ChatMessage("user", Parts:
[
    ImagePart.FromFile("photo.jpg"),
    new TextPart("What is in this image?"),
]);

await foreach (var chunk in session.GenerateAsync(message))
    Console.Write(chunk.Text);
```

### From raw bytes

```csharp
byte[] imageData = await File.ReadAllBytesAsync("diagram.png");

var message = new ChatMessage("user", Parts:
[
    ImagePart.FromBytes(imageData),
    new TextPart("Describe the diagram."),
]);
```

### From a base-64 string

```csharp
string base64 = Convert.ToBase64String(await File.ReadAllBytesAsync("chart.png"));

var message = new ChatMessage("user", Parts:
[
    new ImagePart(Base64: base64),
    new TextPart("What does this chart show?"),
]);
```

---

## Image retention policy

Re-encoding images on every turn is expensive. Use `ImageRetentionPolicy` to control how images are kept in the conversation history:

| Policy | Behaviour |
|---|---|
| `KeepAllImages` | Every image is re-encoded on every turn (default) |
| `KeepLatestImage` | Only the single most-recent image is retained across the whole conversation |
| `DropProcessedImages` | Images are dropped from user messages once an assistant reply follows them |

```csharp
new SessionOptions
{
    ...
    ImageRetentionPolicy = ImageRetentionPolicy.DropProcessedImages,
}
```

`DropProcessedImages` is the best choice for most interactive applications: once the assistant has responded to an image the model has already "seen" it, so re-encoding it on every subsequent turn is wasteful.

---

## GPU acceleration for vision

Image encoding runs on a background thread. By default it uses the GPU when one is available. Opt out with:

```csharp
new SessionOptions
{
    ...
    UseGpuForVision = false,
    VisionThreads   = 4,    // CPU threads to use instead
}
```

### Vision token budget

Control how many tokens each image is allowed to occupy:

```csharp
new SessionOptions
{
    ...
    VisionImageMinTokens = 256,
    VisionImageMaxTokens = 1024,
}
```
