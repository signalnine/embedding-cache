# vectorembed

Go client library for the vector-embed-cache API.

## Installation

```bash
go get github.com/signalnine/embedding-cache/clients/go/vectorembed
```

Requires Go 1.21 or later.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/signalnine/embedding-cache/clients/go/vectorembed"
)

func main() {
    // Create client (uses VECTOR_EMBED_API_KEY env var if not provided)
    client, err := vectorembed.NewClient("your-api-key")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    ctx := context.Background()

    // Generate embedding
    embedding, err := client.Embed(ctx, "hello world")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Dimensions: %d, Cached: %v\n", embedding.Dimensions, embedding.Cached)
}
```

## Usage

### Single Embedding

```go
embedding, err := client.Embed(ctx, "text to embed")
if err != nil {
    log.Fatal(err)
}
fmt.Println(embedding.Vector)    // []float64
fmt.Println(embedding.Cached)    // bool
fmt.Println(embedding.Dimensions) // int
```

### Batch Embeddings

```go
embeddings, err := client.EmbedBatch(ctx, []string{"hello", "world"})
if err != nil {
    log.Fatal(err)
}
for _, e := range embeddings {
    fmt.Println(e.Vector)
}
```

Maximum batch size is 100 texts.

### Similarity Search

```go
results, err := client.Search(ctx, "find similar text",
    vectorembed.WithTopK(10),
    vectorembed.WithMinScore(0.5),
    vectorembed.WithIncludeText(true),
)
if err != nil {
    log.Fatal(err)
}
for _, r := range results.Results {
    fmt.Printf("Score: %.2f, Hash: %s\n", r.Score, r.TextHash)
}
```

### Cache Statistics

```go
stats, err := client.Stats(ctx)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Hits: %d, Misses: %d\n", stats.CacheHits, stats.CacheMisses)
```

## Configuration

```go
client, err := vectorembed.NewClient("api-key",
    vectorembed.WithBaseURL("https://custom.api.com"),
    vectorembed.WithTimeout(60*time.Second),
    vectorembed.WithRetries(5),
    vectorembed.WithModel("nomic-v1.5"),
)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithBaseURL(url)` | `https://api.vector-embed-cache.com` | API base URL |
| `WithTimeout(d)` | `30s` | Request timeout |
| `WithRetries(n)` | `3` | Retry attempts (0 to disable) |
| `WithModel(m)` | `nomic-v1.5` | Default embedding model |
| `WithHTTPClient(c)` | default | Custom http.Client |

## Error Handling

The client returns typed errors that can be inspected with `errors.As`:

```go
embedding, err := client.Embed(ctx, "text")
if err != nil {
    var rateLimitErr *vectorembed.RateLimitError
    if errors.As(err, &rateLimitErr) {
        fmt.Printf("Rate limited. Retry after: %v\n", rateLimitErr.RetryAfter)
        return
    }

    var authErr *vectorembed.AuthenticationError
    if errors.As(err, &authErr) {
        log.Fatal("Invalid API key")
    }

    log.Fatal(err)
}
```

### Error Types

| Type | HTTP Status | Description |
|------|-------------|-------------|
| `AuthenticationError` | 401 | Invalid API key |
| `ValidationError` | 400 | Bad request |
| `RateLimitError` | 429 | Rate limit exceeded (has `RetryAfter` field) |
| `ServerError` | 5xx | Server error |
| `NetworkError` | - | Connection failure |

## Thread Safety

The `Client` is safe for concurrent use from multiple goroutines. A single client instance can be shared across your application.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VECTOR_EMBED_API_KEY` | API key (used if not passed to NewClient) |

## License

MIT
