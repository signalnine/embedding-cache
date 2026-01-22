package vectorembed

import (
	"encoding/json"
	"math/rand"
	"strconv"
	"sync"
	"time"
)

// Retry configuration constants.
const (
	DefaultRetries   = 3
	DefaultBaseDelay = 1 * time.Second
	DefaultMaxDelay  = 30 * time.Second
	JitterFactor     = 0.2 // ±20%
)

// retryableStatuses maps HTTP status codes that should trigger a retry.
var retryableStatuses = map[int]bool{
	429: true, // Too Many Requests
	500: true, // Internal Server Error
	502: true, // Bad Gateway
	503: true, // Service Unavailable
	504: true, // Gateway Timeout
}

// lockedRand provides thread-safe random number generation.
type lockedRand struct {
	mu   sync.Mutex
	rand *rand.Rand
}

// Float64 returns a random float64 in [0, 1) in a thread-safe manner.
func (r *lockedRand) Float64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rand.Float64()
}

// globalRand is the package-level thread-safe random generator.
var globalRand = &lockedRand{
	rand: rand.New(rand.NewSource(time.Now().UnixNano())),
}

// calculateDelay returns the delay for a retry attempt using exponential backoff with jitter.
// The base delay is doubled for each attempt, then jitter of ±JitterFactor is applied.
// The result is capped at DefaultMaxDelay (plus jitter).
func calculateDelay(attempt int) time.Duration {
	// Calculate base delay with exponential backoff: base * 2^attempt
	delay := DefaultBaseDelay * time.Duration(1<<uint(attempt))

	// Cap at max delay
	if delay > DefaultMaxDelay {
		delay = DefaultMaxDelay
	}

	// Apply jitter: delay * (1 + random(-JitterFactor, +JitterFactor))
	jitter := (globalRand.Float64()*2 - 1) * JitterFactor // Random in [-JitterFactor, +JitterFactor]
	delay = time.Duration(float64(delay) * (1 + jitter))

	return delay
}

// isRetryable returns true if the given HTTP status code should trigger a retry.
func isRetryable(statusCode int) bool {
	return retryableStatuses[statusCode]
}

// errorResponse represents the JSON error response from the API.
type errorResponse struct {
	Error string `json:"error"`
}

// mapStatusToError maps an HTTP status code to the appropriate error type.
// The body parameter contains the response body which may contain an error message.
// The headers parameter may contain a Retry-After header for rate limit errors.
func mapStatusToError(statusCode int, body []byte, headers map[string][]string) error {
	// Try to extract error message from JSON body
	var errResp errorResponse
	message := ""
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error != "" {
		message = errResp.Error
	}

	// Set default messages if not extracted from body
	if message == "" {
		switch {
		case statusCode == 400:
			message = "bad request"
		case statusCode == 401:
			message = "authentication failed"
		case statusCode == 429:
			message = "rate limit exceeded"
		case statusCode >= 500:
			message = "server error"
		default:
			message = "unknown error"
		}
	}

	baseErr := APIError{
		Message:    message,
		StatusCode: statusCode,
	}

	switch {
	case statusCode == 400:
		return &ValidationError{APIError: baseErr}
	case statusCode == 401:
		return &AuthenticationError{APIError: baseErr}
	case statusCode == 429:
		retryAfter := time.Duration(0)
		if headers != nil {
			if vals, ok := headers["Retry-After"]; ok && len(vals) > 0 {
				retryAfter = parseRetryAfter(vals[0])
			}
		}
		return &RateLimitError{
			APIError:   baseErr,
			RetryAfter: retryAfter,
		}
	case statusCode >= 500:
		return &ServerError{APIError: baseErr}
	default:
		return &APIError{
			Message:    message,
			StatusCode: statusCode,
		}
	}
}

// parseRetryAfter parses a Retry-After header value and returns the duration.
// Returns 0 if the header is empty or cannot be parsed.
// Note: This implementation only supports seconds format, not HTTP-date format.
func parseRetryAfter(header string) time.Duration {
	if header == "" {
		return 0
	}
	seconds, err := strconv.Atoi(header)
	if err != nil {
		return 0
	}
	return time.Duration(seconds) * time.Second
}
