Here’s a very small HTTP API that works well over a `llama.cpp`-style engine and still supports streaming, sessions, and tool-ready evolution later.

## Minimal shape

### `GET /v1/health`

Check server status.

Response:

```json
{
  "ok": true,
  "engine": "llama.cpp",
  "model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M"
}
```

---

### `GET /v1/models`

List loaded or available models.

Response:

```json
{
  "data": [
    {
      "id": "llama-3.1-8b",
      "object": "model",
      "loaded": true,
      "context_length": 8192
    }
  ]
}
```

---

### `POST /v1/generate`

Single-turn text generation.

Request:

```json
{
  "model": "llama-3.1-8b",
  "prompt": "Write a haiku about APIs.",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

Response:

```json
{
  "id": "gen_123",
  "object": "text_completion",
  "model": "llama-3.1-8b",
  "text": "Quiet endpoints hum\nPackets drift through midnight sockets\nMeaning blooms in bytes",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 17,
    "total_tokens": 23
  }
}
```

---

### `POST /v1/chat`

Basic conversation endpoint.

Request:

```json
{
  "model": "llama-3.1-8b",
  "messages": [
    { "role": "system", "content": "Be concise." },
    { "role": "user", "content": "Explain HTTP in one sentence." }
  ],
  "max_tokens": 128,
  "temperature": 0.2,
  "stream": false
}
```

Response:

```json
{
  "id": "chat_123",
  "object": "chat.completion",
  "model": "llama-3.1-8b",
  "message": {
    "role": "assistant",
    "content": "HTTP is a request-response protocol that lets clients and servers exchange resources over the web."
  },
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 21,
    "total_tokens": 41
  }
}
```

---

## Streaming version

Same `POST /v1/chat`, but with:

```json
{
  "model": "llama-3.1-8b",
  "messages": [
    { "role": "user", "content": "Count from 1 to 5." }
  ],
  "stream": true
}
```

Use `text/event-stream` with tiny SSE events:

```text
event: start
data: {"id":"chat_124","model":"llama-3.1-8b"}

event: token
data: {"delta":"1"}

event: token
data: {"delta":", 2"}

event: token
data: {"delta":", 3"}

event: token
data: {"delta":", 4"}

event: token
data: {"delta":", 5"}

event: end
data: {"finish_reason":"stop","usage":{"prompt_tokens":8,"completion_tokens":9,"total_tokens":17}}
```

That is enough for most agents/UI clients.

---

## Optional stateful sessions

If you want agents without resending full history every time:

### `POST /v1/sessions`

Create a session.

Request:

```json
{
  "model": "llama-3.1-8b",
  "system": "Be concise."
}
```

Response:

```json
{
  "id": "sess_abc",
  "object": "session",
  "model": "llama-3.1-8b"
}
```

### `POST /v1/sessions/{id}/chat`

Append user message and generate reply.

Request:

```json
{
  "message": "What is HTTP?",
  "stream": false
}
```

Response:

```json
{
  "id": "chat_125",
  "session_id": "sess_abc",
  "message": {
    "role": "assistant",
    "content": "HTTP is the standard protocol for web request-response communication."
  },
  "finish_reason": "stop"
}
```

### `DELETE /v1/sessions/{id}`

Free server-side context.

Response:

```json
{
  "ok": true
}
```

---

## Tiny OpenAPI-style schema

```yaml
openapi: 3.1.0
info:
  title: Tiny LLM API
  version: 1.0.0

paths:
  /v1/health:
    get:
      operationId: health
      responses:
        "200":
          description: OK

  /v1/models:
    get:
      operationId: listModels
      responses:
        "200":
          description: Model list

  /v1/generate:
    post:
      operationId: generate
      requestBody:
        required: true
      responses:
        "200":
          description: Text completion

  /v1/chat:
    post:
      operationId: chat
      requestBody:
        required: true
      responses:
        "200":
          description: Chat completion or SSE stream

  /v1/sessions:
    post:
      operationId: createSession
      requestBody:
        required: true
      responses:
        "200":
          description: Session created

  /v1/sessions/{id}/chat:
    post:
      operationId: sessionChat
      parameters:
        - in: path
          name: id
          required: true
          schema: { type: string }
      requestBody:
        required: true
      responses:
        "200":
          description: Session chat completion

  /v1/sessions/{id}:
    delete:
      operationId: deleteSession
      parameters:
        - in: path
          name: id
          required: true
          schema: { type: string }
      responses:
        "200":
          description: Session deleted
```

---

## Request types

A very small common request model:

```json
{
  "model": "string",
  "messages": [
    {
      "role": "system | user | assistant",
      "content": "string"
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "stop": ["</s>"],
  "stream": false
}
```

For plain generate:

```json
{
  "model": "string",
  "prompt": "string",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "stop": ["</s>"],
  "stream": false
}
```

---

## Error format

Keep errors uniform:

```json
{
  "error": {
    "type": "invalid_request",
    "message": "model is required",
    "code": "missing_model"
  }
}
```

Suggested status codes:

* `400` bad request
* `404` session/model not found
* `409` model busy
* `500` engine failure

---

## Why this is enough

This gives you:

* stateless chat
* optional stateful chat
* streaming tokens
* model listing
* clean future expansion

And it stays much smaller than copying OpenAI wholesale.

## Best design choice for llama.cpp

For `llama.cpp`, I’d recommend:

* make `/v1/chat` your main endpoint
* use SSE for streaming
* keep sessions optional
* keep tool calling out of v1 unless you really need it

That avoids overdesign while staying agent-friendly.

Here’s the absolute smallest useful version if you want to go even leaner:

```text
GET  /v1/health
GET  /v1/models
POST /v1/chat
POST /v1/sessions
POST /v1/sessions/{id}/chat
DELETE /v1/sessions/{id}
```

And if you want ultra-minimal:

```text
GET  /v1/health
POST /v1/chat
```

with `messages`, `stream`, and `model`.

I can turn this into a full OpenAPI 3.1 file next.
