---
layout: page
title: Structured Output
permalink: /features/structured-output/
parent: Features
nav_order: 3
---

# Structured Output

`AskAsync<T>()` constrains the model's token sampling with a GBNF grammar derived from `T`'s JSON schema, so the output is always valid JSON that deserializes cleanly into a .NET type — no prompt-engineering or retry logic required.

---

## Basic usage

Define a record or class that represents the shape you want, then call `AskAsync<T>()`:

```csharp
public record MovieReview(string Title, int ReleaseYear, double Score, string Summary);

MovieReview review = await session.AskAsync<MovieReview>(
    "Review the movie Interstellar.");

Console.WriteLine($"{review.Title} ({review.ReleaseYear}) — {review.Score}/10");
Console.WriteLine(review.Summary);
```

---

## Supported types

Any JSON-serializable type works:

- **Records** — `record Person(string Name, int Age)`
- **Classes** — `class Product { public string Name { get; set; } ... }`
- **Collections** — `List<string>`, `string[]`, `IReadOnlyList<T>`
- **Nested types** — records or classes that contain other records/classes
- **Nullable / optional properties** — may be returned as `null` by the model when the property is nullable or has a default value

### Example — list of items

```csharp
public record Product(string Name, decimal Price);

List<Product> products = await session.AskAsync<List<Product>>(
    "Name three popular smartphones with their approximate prices in USD.");

foreach (var p in products)
    Console.WriteLine($"{p.Name}: ${p.Price}");
```

### Example — nested types

```csharp
public record Address(string Street, string City, string Country);
public record Person(string FullName, int Age, Address HomeAddress);

Person person = await session.AskAsync<Person>(
    "Invent a fictional person and describe them.");

Console.WriteLine($"{person.FullName}, {person.Age}, lives in {person.HomeAddress.City}");
```

---

## How it works

1. YALMR reflects over `T` and generates a GBNF grammar that accepts only JSON valid against `T`'s schema.
2. The grammar is passed to llama.cpp's sampler, which rejects any token that would produce invalid JSON.
3. The raw JSON output is deserialized with `System.Text.Json` into a `T` instance.

This means **the model cannot produce malformed or out-of-schema output** — the guarantee is enforced at the sampling level, not by post-processing.

---

## Using a custom grammar

If you need a grammar not derived from a C# type — for example, a restricted vocabulary or a domain-specific language — supply it directly through `InferenceOptions`:

```csharp
// Allow only "yes" or "no"
string grammar = """
  root ::= ("yes" | "no")
""";

var options = new InferenceOptions { Grammar = grammar };

string answer = await session.AskAsync<string>(
    "Is Paris the capital of France?",
    options);
```

---

## Limitations

- **Enum values** — use `[JsonConverter]` or string-backed enums for best results.
- **Polymorphic types** — discriminated unions require a custom grammar.
- **Very large schemas** — deeply nested or very wide types increase grammar complexity. For best performance keep output types focused.
