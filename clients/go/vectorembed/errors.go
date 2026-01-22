package vectorembed

import "time"

// APIError is the base error type for all API errors.
// It contains the error message, an optional error code, and the HTTP status code.
type APIError struct {
	Message    string
	Code       string
	StatusCode int
}

// Error implements the error interface.
func (e *APIError) Error() string {
	return e.Message
}

// AuthenticationError is returned when authentication fails (HTTP 401).
type AuthenticationError struct {
	APIError
}

// RateLimitError is returned when the rate limit is exceeded (HTTP 429).
// RetryAfter indicates how long to wait before retrying.
type RateLimitError struct {
	APIError
	RetryAfter time.Duration
}

// ValidationError is returned when the request is invalid (HTTP 400).
type ValidationError struct {
	APIError
}

// ServerError is returned when the server encounters an error (HTTP 5xx).
type ServerError struct {
	APIError
}

// NetworkError is returned when a network-level error occurs,
// such as connection refused or timeout.
type NetworkError struct {
	Message string
	Cause   error
}

// Error implements the error interface.
func (e *NetworkError) Error() string {
	return e.Message
}

// Unwrap returns the underlying cause of the error, enabling errors.Is and errors.As
// to work with wrapped errors.
func (e *NetworkError) Unwrap() error {
	return e.Cause
}
