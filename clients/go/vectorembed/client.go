package vectorembed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const (
	// EnvAPIKey is the environment variable name for the API key.
	EnvAPIKey = "VECTOR_EMBED_API_KEY"

	// MaxBatchSize is the maximum number of texts allowed in a batch request.
	MaxBatchSize = 100

	// UserAgent is the User-Agent header value sent with all requests.
	UserAgent = "vector-embed-cache-go/1.0"
)

// Client is the main client for interacting with the vector-embed-cache API.
type Client struct {
	apiKey     string
	baseURL    string
	timeout    time.Duration
	retries    int
	model      string
	httpClient *http.Client
}

// NewClient creates a new Client with the given API key and options.
// If apiKey is empty, it falls back to the VECTOR_EMBED_API_KEY environment variable.
// Returns an error if no API key is provided.
func NewClient(apiKey string, opts ...ClientOption) (*Client, error) {
	// Fall back to environment variable if apiKey is empty
	if apiKey == "" {
		apiKey = os.Getenv(EnvAPIKey)
	}

	// Require an API key
	if apiKey == "" {
		return nil, fmt.Errorf("API key required: provide apiKey argument or set %s environment variable", EnvAPIKey)
	}

	// Start with default configuration
	cfg := defaultClientConfig()

	// Apply options
	for _, opt := range opts {
		opt(&cfg)
	}

	// Create HTTP client if not provided
	httpClient := cfg.httpClient
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: cfg.timeout,
		}
	}

	return &Client{
		apiKey:     apiKey,
		baseURL:    cfg.baseURL,
		timeout:    cfg.timeout,
		retries:    cfg.retries,
		model:      cfg.model,
		httpClient: httpClient,
	}, nil
}

// Close releases any resources held by the client.
// Currently this is a no-op but is provided for future compatibility.
func (c *Client) Close() error {
	return nil
}

// Embed generates an embedding for a single text.
func (c *Client) Embed(ctx context.Context, text string) (*EmbedResponse, error) {
	reqBody := embedRequest{
		Text:  text,
		Model: c.model,
	}

	var resp EmbedResponse
	if err := c.doRequest(ctx, "POST", "/v1/embed", reqBody, &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}

// EmbedBatch generates embeddings for multiple texts.
// Returns an error if more than MaxBatchSize (100) texts are provided.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([]EmbedResponse, error) {
	if len(texts) > MaxBatchSize {
		return nil, &ValidationError{
			APIError: APIError{
				Message:    fmt.Sprintf("batch size %d exceeds maximum of %d", len(texts), MaxBatchSize),
				StatusCode: 400,
			},
		}
	}

	reqBody := embedBatchRequest{
		Texts: texts,
		Model: c.model,
	}

	var resp []EmbedResponse
	if err := c.doRequest(ctx, "POST", "/v1/embed/batch", reqBody, &resp); err != nil {
		return nil, err
	}

	return resp, nil
}

// Search performs a similarity search for the given text.
func (c *Client) Search(ctx context.Context, text string, opts ...SearchOption) (*SearchResponse, error) {
	cfg := defaultSearchConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	reqBody := searchRequest{
		Text:        text,
		TopK:        cfg.topK,
		MinScore:    cfg.minScore,
		IncludeText: cfg.includeText,
	}

	var resp SearchResponse
	if err := c.doRequest(ctx, "POST", "/v1/search", reqBody, &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}

// Stats retrieves cache statistics.
func (c *Client) Stats(ctx context.Context) (*StatsResponse, error) {
	var resp StatsResponse
	if err := c.doRequest(ctx, "GET", "/v1/stats", nil, &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}

// doRequest performs an HTTP request with retry logic.
func (c *Client) doRequest(ctx context.Context, method, path string, body interface{}, result interface{}) error {
	url := c.baseURL + path

	var bodyReader io.Reader
	if body != nil {
		bodyBytes, err := json.Marshal(body)
		if err != nil {
			return &NetworkError{
				Message: fmt.Sprintf("failed to marshal request body: %v", err),
				Cause:   err,
			}
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	var lastErr error
	maxAttempts := c.retries + 1 // retries + initial attempt

	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Reset body reader for retries
		if body != nil {
			bodyBytes, _ := json.Marshal(body)
			bodyReader = bytes.NewReader(bodyBytes)
		}

		req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
		if err != nil {
			return &NetworkError{
				Message: fmt.Sprintf("failed to create request: %v", err),
				Cause:   err,
			}
		}

		// Set headers
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("User-Agent", UserAgent)

		resp, err := c.httpClient.Do(req)
		if err != nil {
			// Check for context cancellation
			if ctx.Err() != nil {
				return &NetworkError{
					Message: fmt.Sprintf("request cancelled: %v", ctx.Err()),
					Cause:   ctx.Err(),
				}
			}
			lastErr = &NetworkError{
				Message: fmt.Sprintf("request failed: %v", err),
				Cause:   err,
			}
			// Network errors are retryable
			if attempt < maxAttempts-1 {
				time.Sleep(calculateDelay(attempt))
				continue
			}
			return lastErr
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = &NetworkError{
				Message: fmt.Sprintf("failed to read response body: %v", err),
				Cause:   err,
			}
			if attempt < maxAttempts-1 {
				time.Sleep(calculateDelay(attempt))
				continue
			}
			return lastErr
		}

		// Check for error status codes
		if resp.StatusCode >= 400 {
			lastErr = mapStatusToError(resp.StatusCode, respBody, resp.Header)

			// Check if this error is retryable
			if isRetryable(resp.StatusCode) && attempt < maxAttempts-1 {
				time.Sleep(calculateDelay(attempt))
				continue
			}
			return lastErr
		}

		// Success - decode response
		if err := json.Unmarshal(respBody, result); err != nil {
			return &NetworkError{
				Message: fmt.Sprintf("failed to decode response: %v", err),
				Cause:   err,
			}
		}

		return nil
	}

	// Should not reach here, but return last error if we do
	return lastErr
}
