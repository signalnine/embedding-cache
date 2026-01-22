package vectorembed

import (
	"fmt"
	"testing"
	"time"
)

func TestCalculateDelay_Exponential(t *testing.T) {
	// Attempt 0: ~1s ±20%
	delay0 := calculateDelay(0)
	if delay0 < 800*time.Millisecond || delay0 > 1200*time.Millisecond {
		t.Errorf("delay(0) = %v, want ~1s ±20%%", delay0)
	}

	// Attempt 1: ~2s ±20%
	delay1 := calculateDelay(1)
	if delay1 < 1600*time.Millisecond || delay1 > 2400*time.Millisecond {
		t.Errorf("delay(1) = %v, want ~2s ±20%%", delay1)
	}

	// Attempt 2: ~4s ±20%
	delay2 := calculateDelay(2)
	if delay2 < 3200*time.Millisecond || delay2 > 4800*time.Millisecond {
		t.Errorf("delay(2) = %v, want ~4s ±20%%", delay2)
	}
}

func TestCalculateDelay_Capped(t *testing.T) {
	// Large attempt should be capped at MaxDelay
	delay := calculateDelay(10)
	maxWithJitter := DefaultMaxDelay + time.Duration(float64(DefaultMaxDelay)*JitterFactor)
	if delay > maxWithJitter {
		t.Errorf("delay(10) = %v, should be capped at ~%v", delay, DefaultMaxDelay)
	}
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		status int
		want   bool
	}{
		{200, false},
		{400, false},
		{401, false},
		{429, true},
		{500, true},
		{502, true},
		{503, true},
		{504, true},
	}
	for _, tt := range tests {
		if got := isRetryable(tt.status); got != tt.want {
			t.Errorf("isRetryable(%d) = %v, want %v", tt.status, got, tt.want)
		}
	}
}

func TestMapStatusToError(t *testing.T) {
	tests := []struct {
		status int
		errTyp string
	}{
		{400, "*vectorembed.ValidationError"},
		{401, "*vectorembed.AuthenticationError"},
		{429, "*vectorembed.RateLimitError"},
		{500, "*vectorembed.ServerError"},
		{503, "*vectorembed.ServerError"},
	}
	for _, tt := range tests {
		err := mapStatusToError(tt.status, []byte(`{"error":"test"}`), nil)
		got := fmt.Sprintf("%T", err)
		if got != tt.errTyp {
			t.Errorf("mapStatusToError(%d) type = %s, want %s", tt.status, got, tt.errTyp)
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	tests := []struct {
		header string
		want   time.Duration
	}{
		{"30", 30 * time.Second},
		{"0", 0},
		{"", 0},
		{"invalid", 0},
	}
	for _, tt := range tests {
		if got := parseRetryAfter(tt.header); got != tt.want {
			t.Errorf("parseRetryAfter(%q) = %v, want %v", tt.header, got, tt.want)
		}
	}
}

func TestMapStatusToError_MessageExtraction(t *testing.T) {
	// Test with JSON error body
	body := []byte(`{"error":"custom error message"}`)
	err := mapStatusToError(400, body, nil)
	if err.Error() != "custom error message" {
		t.Errorf("error message = %q, want %q", err.Error(), "custom error message")
	}

	// Test with invalid JSON body - should use default message
	invalidBody := []byte(`not json`)
	err = mapStatusToError(400, invalidBody, nil)
	if err.Error() != "bad request" {
		t.Errorf("error message = %q, want %q", err.Error(), "bad request")
	}
}

func TestMapStatusToError_RateLimitRetryAfter(t *testing.T) {
	// Test that RateLimitError includes RetryAfter from header
	header := make(map[string][]string)
	header["Retry-After"] = []string{"60"}
	err := mapStatusToError(429, []byte(`{"error":"rate limited"}`), header)

	rateLimitErr, ok := err.(*RateLimitError)
	if !ok {
		t.Fatalf("expected *RateLimitError, got %T", err)
	}
	if rateLimitErr.RetryAfter != 60*time.Second {
		t.Errorf("RetryAfter = %v, want %v", rateLimitErr.RetryAfter, 60*time.Second)
	}
}

func TestLockedRand_Concurrent(t *testing.T) {
	// Test that lockedRand is safe for concurrent use
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				v := globalRand.Float64()
				if v < 0 || v >= 1 {
					t.Errorf("Float64() = %v, want [0, 1)", v)
				}
			}
			done <- true
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
}
