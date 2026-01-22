package vectorembed

import (
	"net/http"
	"time"
)

// clientConfig holds configuration for the Client.
type clientConfig struct {
	baseURL    string
	timeout    time.Duration
	retries    int
	model      string
	httpClient *http.Client
}

// defaultClientConfig returns a clientConfig with default values.
func defaultClientConfig() clientConfig {
	return clientConfig{
		baseURL: "https://api.vector-embed-cache.com",
		timeout: 30 * time.Second,
		retries: 3,
		model:   "nomic-v1.5",
	}
}

// ClientOption configures the client.
type ClientOption func(*clientConfig)

// WithBaseURL sets the base URL for API requests.
func WithBaseURL(url string) ClientOption {
	return func(c *clientConfig) {
		c.baseURL = url
	}
}

// WithTimeout sets the request timeout.
func WithTimeout(d time.Duration) ClientOption {
	return func(c *clientConfig) {
		c.timeout = d
	}
}

// WithRetries sets the number of retry attempts for failed requests.
func WithRetries(n int) ClientOption {
	return func(c *clientConfig) {
		c.retries = n
	}
}

// WithModel sets the default embedding model.
func WithModel(model string) ClientOption {
	return func(c *clientConfig) {
		c.model = model
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *clientConfig) {
		c.httpClient = client
	}
}

// searchConfig holds configuration for search requests.
type searchConfig struct {
	topK        int
	minScore    float64
	includeText bool
}

// defaultSearchConfig returns a searchConfig with default values.
func defaultSearchConfig() searchConfig {
	return searchConfig{
		topK:        10,
		minScore:    0.0,
		includeText: false,
	}
}

// SearchOption configures search requests.
type SearchOption func(*searchConfig)

// WithTopK sets the number of results to return.
func WithTopK(k int) SearchOption {
	return func(c *searchConfig) {
		c.topK = k
	}
}

// WithMinScore sets the minimum similarity score threshold.
func WithMinScore(score float64) SearchOption {
	return func(c *searchConfig) {
		c.minScore = score
	}
}

// WithIncludeText sets whether to include the original text in results.
func WithIncludeText(include bool) SearchOption {
	return func(c *searchConfig) {
		c.includeText = include
	}
}
