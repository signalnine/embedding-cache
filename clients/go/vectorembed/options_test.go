package vectorembed

import (
	"net/http"
	"testing"
	"time"
)

func TestDefaultClientConfig(t *testing.T) {
	cfg := defaultClientConfig()
	if cfg.baseURL != "https://api.vector-embed-cache.com" {
		t.Errorf("baseURL = %q, want default", cfg.baseURL)
	}
	if cfg.timeout != 30*time.Second {
		t.Errorf("timeout = %v, want 30s", cfg.timeout)
	}
	if cfg.retries != 3 {
		t.Errorf("retries = %d, want 3", cfg.retries)
	}
	if cfg.model != "nomic-v1.5" {
		t.Errorf("model = %q, want nomic-v1.5", cfg.model)
	}
}

func TestWithBaseURL(t *testing.T) {
	cfg := defaultClientConfig()
	WithBaseURL("https://custom.example.com")(&cfg)
	if cfg.baseURL != "https://custom.example.com" {
		t.Errorf("baseURL = %q, want custom URL", cfg.baseURL)
	}
}

func TestWithTimeout(t *testing.T) {
	cfg := defaultClientConfig()
	WithTimeout(60 * time.Second)(&cfg)
	if cfg.timeout != 60*time.Second {
		t.Errorf("timeout = %v, want 60s", cfg.timeout)
	}
}

func TestWithRetries(t *testing.T) {
	cfg := defaultClientConfig()
	WithRetries(5)(&cfg)
	if cfg.retries != 5 {
		t.Errorf("retries = %d, want 5", cfg.retries)
	}
}

func TestWithRetriesZero(t *testing.T) {
	cfg := defaultClientConfig()
	WithRetries(0)(&cfg)
	if cfg.retries != 0 {
		t.Errorf("retries = %d, want 0", cfg.retries)
	}
}

func TestWithModel(t *testing.T) {
	cfg := defaultClientConfig()
	WithModel("custom-model")(&cfg)
	if cfg.model != "custom-model" {
		t.Errorf("model = %q, want custom-model", cfg.model)
	}
}

func TestWithHTTPClient(t *testing.T) {
	cfg := defaultClientConfig()
	custom := &http.Client{Timeout: 5 * time.Second}
	WithHTTPClient(custom)(&cfg)
	if cfg.httpClient != custom {
		t.Error("httpClient should be custom client")
	}
}

func TestDefaultSearchConfig(t *testing.T) {
	cfg := defaultSearchConfig()
	if cfg.topK != 10 {
		t.Errorf("topK = %d, want 10", cfg.topK)
	}
	if cfg.minScore != 0.0 {
		t.Errorf("minScore = %f, want 0.0", cfg.minScore)
	}
	if cfg.includeText {
		t.Error("includeText should be false by default")
	}
}

func TestWithTopK(t *testing.T) {
	cfg := defaultSearchConfig()
	WithTopK(20)(&cfg)
	if cfg.topK != 20 {
		t.Errorf("topK = %d, want 20", cfg.topK)
	}
}

func TestWithMinScore(t *testing.T) {
	cfg := defaultSearchConfig()
	WithMinScore(0.5)(&cfg)
	if cfg.minScore != 0.5 {
		t.Errorf("minScore = %f, want 0.5", cfg.minScore)
	}
}

func TestWithIncludeText(t *testing.T) {
	cfg := defaultSearchConfig()
	WithIncludeText(true)(&cfg)
	if !cfg.includeText {
		t.Error("includeText should be true")
	}
}
