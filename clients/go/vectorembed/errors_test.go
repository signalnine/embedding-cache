package vectorembed

import (
	"errors"
	"testing"
	"time"
)

func TestAPIError_Error(t *testing.T) {
	err := &APIError{Message: "test error", Code: "test_code", StatusCode: 400}
	if got := err.Error(); got != "test error" {
		t.Errorf("Error() = %q, want %q", got, "test error")
	}
}

func TestAuthenticationError_ErrorsAs(t *testing.T) {
	err := &AuthenticationError{APIError: APIError{Message: "unauthorized", StatusCode: 401}}

	var authErr *AuthenticationError
	if !errors.As(err, &authErr) {
		t.Error("errors.As should match AuthenticationError")
	}
	if authErr.StatusCode != 401 {
		t.Errorf("StatusCode = %d, want 401", authErr.StatusCode)
	}
}

func TestRateLimitError_RetryAfter(t *testing.T) {
	err := &RateLimitError{
		APIError:   APIError{Message: "rate limited", StatusCode: 429},
		RetryAfter: 30 * time.Second,
	}

	if err.RetryAfter != 30*time.Second {
		t.Errorf("RetryAfter = %v, want %v", err.RetryAfter, 30*time.Second)
	}

	var rateErr *RateLimitError
	if !errors.As(err, &rateErr) {
		t.Error("errors.As should match RateLimitError")
	}
}

func TestValidationError_ErrorsAs(t *testing.T) {
	err := &ValidationError{APIError: APIError{Message: "bad request", StatusCode: 400}}

	var valErr *ValidationError
	if !errors.As(err, &valErr) {
		t.Error("errors.As should match ValidationError")
	}
}

func TestServerError_ErrorsAs(t *testing.T) {
	err := &ServerError{APIError: APIError{Message: "internal error", StatusCode: 500}}

	var srvErr *ServerError
	if !errors.As(err, &srvErr) {
		t.Error("errors.As should match ServerError")
	}
}

func TestNetworkError_Unwrap(t *testing.T) {
	cause := errors.New("connection refused")
	err := &NetworkError{Message: "network error", Cause: cause}

	if err.Error() != "network error" {
		t.Errorf("Error() = %q, want %q", err.Error(), "network error")
	}

	if !errors.Is(err, cause) {
		t.Error("errors.Is should find cause through Unwrap")
	}

	if err.Unwrap() != cause {
		t.Error("Unwrap should return cause")
	}
}

func TestNetworkError_NilCause(t *testing.T) {
	err := &NetworkError{Message: "timeout"}

	if err.Unwrap() != nil {
		t.Error("Unwrap should return nil when Cause is nil")
	}
}
