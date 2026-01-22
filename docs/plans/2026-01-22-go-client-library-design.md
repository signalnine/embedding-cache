# Go Client Library Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Go client library for the vector-embed-cache API with zero external dependencies.

**Architecture:** Functional options pattern, custom error types with `errors.Is/As`, context-first methods, built-in retry with exponential backoff.

**Tech Stack:** Go 1.21+, net/http (stdlib only)

---

## Design Decisions (Consensus-Driven)

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Module path | `github.com/signalnine/embedding-cache/clients/go` | Monorepo consistency |
| Package name | `vectorembed` | Self-documenting, collision-resistant |
| HTTP client | net/http with custom wrapper | Zero dependencies, Go philosophy |
| Error handling | Custom types with `errors.Is/As` | Idiomatic Go, RateLimitError needs RetryAfter |
| API pattern | Functional options | Idiomatic Go (AWS SDK v2, gRPC pattern) |
| Context | First parameter on every method | Go standard, enables cancellation |
| Retry | Built-in, default 3 attempts | Match JS/Python clients, configurable |

---

## Public API

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/signalnine/embedding-cache/clients/go/vectorembed"
)

func main() {
    // Create client (reads VECTOR_EMBED_API_KEY env var if not provided)
    client, err := vectorembed.NewClient("your-api-key")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    ctx := context.Background()

    // Single embedding
    embedding, err := client.Embed(ctx, "hello world")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Vector: %v, Cached: %v\n", embedding.Vector[:3], embedding.Cached)

    // Batch embeddings
    embeddings, err := client.EmbedBatch(ctx, []string{"hello", "world"})
    if err != nil {
        log.Fatal(err)
    }

    // Similarity search
    results, err := client.Search(ctx, "find similar",
        vectorembed.WithTopK(10),
        vectorembed.WithMinScore(0.5),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Statistics
    stats, err := client.Stats(ctx)
    if err != nil {
        log.Fatal(err)
    }
}
```

### Client Options

```go
// Create with options
client, err := vectorembed.NewClient(apiKey,
    vectorembed.WithBaseURL("https://custom-api.example.com"),
    vectorembed.WithTimeout(60*time.Second),
    vectorembed.WithRetries(5),
    vectorembed.WithModel("nomic-v1.5"),
)

// Disable retries
client, err := vectorembed.NewClient(apiKey,
    vectorembed.WithRetries(0),
)

// Use environment variable for API key
client, err := vectorembed.NewClient("",  // Falls back to VECTOR_EMBED_API_KEY
    vectorembed.WithTimeout(30*time.Second),
)
```

### Option Types

```go
// ClientOption configures the client
type ClientOption func(*clientConfig)

func WithBaseURL(url string) ClientOption
func WithTimeout(d time.Duration) ClientOption
func WithRetries(n int) ClientOption
func WithModel(model string) ClientOption
func WithHTTPClient(c *http.Client) ClientOption  // For custom transport

// SearchOption configures search requests
type SearchOption func(*searchConfig)

func WithTopK(k int) SearchOption
func WithMinScore(score float64) SearchOption
func WithIncludeText(include bool) SearchOption
```

---

## Request Types

```go
// embedRequest is the payload for POST /v1/embed
type embedRequest struct {
    Text  string `json:"text"`
    Model string `json:"model,omitempty"`
}

// embedBatchRequest is the payload for POST /v1/embed/batch
type embedBatchRequest struct {
    Texts []string `json:"texts"`
    Model string   `json:"model,omitempty"`
}

// searchRequest is the payload for POST /v1/search
type searchRequest struct {
    Text        string  `json:"text"`
    TopK        int     `json:"top_k,omitempty"`
    MinScore    float64 `json:"min_score,omitempty"`
    IncludeText bool    `json:"include_text,omitempty"`
}
```

---

## Response Types

```go
// EmbedResponse represents an embedding result
type EmbedResponse struct {
    Vector     []float64 `json:"vector"`
    Cached     bool      `json:"cached"`
    Dimensions int       `json:"dimensions"`
}

// SearchResult represents a single search match
type SearchResult struct {
    TextHash string  `json:"text_hash"`
    Score    float64 `json:"score"`
    Text     *string `json:"text,omitempty"`
    Model    string  `json:"model"`
    HitCount int     `json:"hit_count"`
}

// SearchResponse represents search results
type SearchResponse struct {
    Results      []SearchResult `json:"results"`
    Total        int            `json:"total"`
    SearchTimeMs float64        `json:"search_time_ms"`
}

// StatsResponse represents cache statistics
type StatsResponse struct {
    CacheHits   int `json:"cache_hits"`
    CacheMisses int `json:"cache_misses"`
    TotalCached int `json:"total_cached"`
}
```

---

## Error Handling

```go
// Base error type
type APIError struct {
    Message    string
    Code       string
    StatusCode int
    RequestID  string  // If returned by server
}

func (e *APIError) Error() string

// Typed errors for errors.Is/As
type AuthenticationError struct{ APIError }
type RateLimitError struct {
    APIError
    RetryAfter time.Duration
}
type ValidationError struct{ APIError }
type ServerError struct{ APIError }
type NetworkError struct {
    Message string
    Cause   error
}

func (e *NetworkError) Error() string
func (e *NetworkError) Unwrap() error
```

### Error Usage

```go
embedding, err := client.Embed(ctx, "text")
if err != nil {
    var rateLimitErr *vectorembed.RateLimitError
    if errors.As(err, &rateLimitErr) {
        log.Printf("Rate limited. Retry after %v", rateLimitErr.RetryAfter)
        return
    }

    var authErr *vectorembed.AuthenticationError
    if errors.As(err, &authErr) {
        log.Fatal("Invalid API key")
    }

    log.Printf("Error: %v", err)
}
```

---

## Project Structure

```
clients/go/
├── vectorembed/
│   ├── client.go         # Client struct and NewClient
│   ├── client_test.go    # Client tests
│   ├── errors.go         # Error types
│   ├── errors_test.go    # Error tests
│   ├── http.go           # HTTP wrapper with retry
│   ├── http_test.go      # HTTP tests
│   ├── options.go        # Option types
│   ├── types.go          # Response types
│   └── doc.go            # Package documentation
├── go.mod
├── go.sum
└── README.md
```

---

## Retry Logic

### Configuration

```go
const (
    DefaultRetries   = 3
    DefaultBaseDelay = 1 * time.Second
    DefaultMaxDelay  = 30 * time.Second
    JitterFactor     = 0.2  // ±20%
)

var retryableStatuses = map[int]bool{
    429: true,  // Rate limit
    500: true,  // Internal server error
    502: true,  // Bad gateway
    503: true,  // Service unavailable
    504: true,  // Gateway timeout
}
```

### Backoff Calculation

```go
func calculateDelay(attempt int) time.Duration {
    exponential := float64(DefaultBaseDelay) * math.Pow(2, float64(attempt))
    capped := math.Min(exponential, float64(DefaultMaxDelay))
    jitter := capped * JitterFactor * (rand.Float64()*2 - 1)
    return time.Duration(capped + jitter)
}
```

### Retry Behavior

- Respects `Retry-After` header when present (429 responses)
- Does NOT retry 4xx errors (except 429)
- Respects context cancellation between retries
- Retries can be disabled with `WithRetries(0)`

---

## HTTP Headers

```go
headers := map[string]string{
    "Authorization": "Bearer " + apiKey,
    "Content-Type":  "application/json",
    "Accept":        "application/json",
    "User-Agent":    "vectorembed-go/" + Version,
}
```

---

## Client Interface

```go
// Version is the client library version
const Version = "0.1.0"

// Client is a thread-safe client for the vector-embed-cache API.
// A single Client can be shared across multiple goroutines.
type Client struct {
    // unexported fields
}

func NewClient(apiKey string, opts ...ClientOption) (*Client, error)

// Close releases resources. Currently a no-op but included for future
// compatibility if connection pooling configuration is added.
func (c *Client) Close() error

func (c *Client) Embed(ctx context.Context, text string) (*EmbedResponse, error)
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([]EmbedResponse, error)
func (c *Client) Search(ctx context.Context, text string, opts ...SearchOption) (*SearchResponse, error)
func (c *Client) Stats(ctx context.Context) (*StatsResponse, error)
```

---

## Thread Safety

The `Client` is safe for concurrent use from multiple goroutines. Internally it uses:
- Immutable configuration after construction
- `http.Client` which is safe for concurrent use
- Local `rand.Rand` with mutex for jitter calculation (avoids global rand lock)

---

## Batch Semantics

- **Max batch size**: 100 texts (matches server limit)
- **Order guaranteed**: Results slice matches input order
- **Partial failure**: All-or-nothing (server rejects entire batch on validation error)
- **Client validation**: Returns error immediately if batch exceeds 100 texts or contains empty strings

---

## Testing Strategy

### Unit Tests

- Mock HTTP responses using `httptest.Server`
- Test all error type mappings
- Test retry logic (success after retry, max retries exceeded)
- Test option application
- Test context cancellation
- Test thread safety with `-race` flag

### Integration Tests

Integration tests run against a real server (skipped by default):

```go
// Run with: go test -v -tags=integration ./...
// +build integration

func TestIntegration_Embed(t *testing.T) {
    apiKey := os.Getenv("VECTOR_EMBED_API_KEY")
    if apiKey == "" {
        t.Skip("VECTOR_EMBED_API_KEY not set")
    }

    client, err := vectorembed.NewClient(apiKey)
    require.NoError(t, err)
    defer client.Close()

    resp, err := client.Embed(context.Background(), "hello world")
    require.NoError(t, err)
    assert.True(t, len(resp.Vector) > 0)
}
```

### Table-Driven Tests

```go
func TestEmbed(t *testing.T) {
    tests := []struct {
        name       string
        response   string
        statusCode int
        wantErr    error
        want       *EmbedResponse
    }{
        {
            name:       "success",
            response:   `{"vector":[0.1,0.2],"cached":true,"dimensions":2}`,
            statusCode: 200,
            want:       &EmbedResponse{Vector: []float64{0.1, 0.2}, Cached: true, Dimensions: 2},
        },
        {
            name:       "unauthorized",
            response:   `{"error":"invalid api key"}`,
            statusCode: 401,
            wantErr:    &AuthenticationError{},
        },
    }
    // ...
}
```

---

## CI Integration

Add to `.github/workflows/ci.yml`:

```yaml
  test-go-client:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: clients/go
    strategy:
      matrix:
        go-version: ['1.21', '1.22', '1.23']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: ${{ matrix.go-version }}
      - name: Test
        run: go test -v -race ./...
      - name: Vet
        run: go vet ./...
      - name: Build
        run: go build ./...
```

---

## Environment Variable Support

```go
func NewClient(apiKey string, opts ...ClientOption) (*Client, error) {
    if apiKey == "" {
        apiKey = os.Getenv("VECTOR_EMBED_API_KEY")
    }
    if apiKey == "" {
        return nil, errors.New("API key required: pass apiKey or set VECTOR_EMBED_API_KEY")
    }
    // ...
}
```

---

## Comparison with Other Clients

| Feature | JS Client | Python Client | Go Client |
|---------|-----------|---------------|-----------|
| API pattern | Factory function | Class init | Functional options |
| Async support | Promise-based | Separate AsyncClient | Context-based |
| Error handling | Typed subclasses | Typed subclasses | errors.Is/As |
| Retry default | 3 | 3 | 3 |
| HTTP library | Native fetch | httpx | net/http |
| Dependencies | 0 | 1 (httpx) | 0 |

---

## Future Considerations

1. **Streaming** - Not needed for current API
2. **Connection pooling** - net/http handles automatically
3. **Middleware/hooks** - Could add request/response hooks via options
4. **Generics** - Consider generic response parsing if Go 1.18+ only
