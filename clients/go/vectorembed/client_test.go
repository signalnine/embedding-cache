package vectorembed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func TestNewClient_RequiresAPIKey(t *testing.T) {
	os.Unsetenv("VECTOR_EMBED_API_KEY")
	_, err := NewClient("")
	if err == nil {
		t.Error("expected error when no API key provided")
	}
}

func TestNewClient_UsesEnvVar(t *testing.T) {
	os.Setenv("VECTOR_EMBED_API_KEY", "test-key-from-env")
	defer os.Unsetenv("VECTOR_EMBED_API_KEY")

	client, err := NewClient("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client.apiKey != "test-key-from-env" {
		t.Errorf("apiKey = %q, want test-key-from-env", client.apiKey)
	}
}

func TestNewClient_ExplicitKeyOverridesEnv(t *testing.T) {
	os.Setenv("VECTOR_EMBED_API_KEY", "env-key")
	defer os.Unsetenv("VECTOR_EMBED_API_KEY")

	client, err := NewClient("explicit-key")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client.apiKey != "explicit-key" {
		t.Errorf("apiKey = %q, want explicit-key", client.apiKey)
	}
}

func TestClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/embed" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing or wrong Authorization header")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"vector":     []float64{0.1, 0.2, 0.3},
			"cached":     true,
			"dimensions": 3,
		})
	}))
	defer server.Close()

	client, _ := NewClient("test-key", WithBaseURL(server.URL))
	resp, err := client.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	if len(resp.Vector) != 3 {
		t.Errorf("Vector length = %d, want 3", len(resp.Vector))
	}
	if !resp.Cached {
		t.Error("Cached should be true")
	}
}

func TestClient_EmbedBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]map[string]interface{}{
			{"vector": []float64{0.1}, "cached": true, "dimensions": 1},
			{"vector": []float64{0.2}, "cached": false, "dimensions": 1},
		})
	}))
	defer server.Close()

	client, _ := NewClient("test-key", WithBaseURL(server.URL))
	resp, err := client.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedBatch error: %v", err)
	}
	if len(resp) != 2 {
		t.Errorf("response length = %d, want 2", len(resp))
	}
}

func TestClient_EmbedBatch_MaxSize(t *testing.T) {
	client, _ := NewClient("test-key")
	texts := make([]string, 101)
	for i := range texts {
		texts[i] = "text"
	}
	_, err := client.EmbedBatch(context.Background(), texts)
	if err == nil {
		t.Error("expected error for batch > 100")
	}
}

func TestClient_Search(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"results": []map[string]interface{}{
				{"text_hash": "abc", "score": 0.95, "model": "m", "hit_count": 1},
			},
			"total":          1,
			"search_time_ms": 5.0,
		})
	}))
	defer server.Close()

	client, _ := NewClient("test-key", WithBaseURL(server.URL))
	resp, err := client.Search(context.Background(), "query", WithTopK(10))
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(resp.Results) != 1 {
		t.Errorf("Results length = %d, want 1", len(resp.Results))
	}
}

func TestClient_Stats(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("Stats should be GET, got %s", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"cache_hits":   100,
			"cache_misses": 10,
			"total_cached": 110,
		})
	}))
	defer server.Close()

	client, _ := NewClient("test-key", WithBaseURL(server.URL))
	resp, err := client.Stats(context.Background())
	if err != nil {
		t.Fatalf("Stats error: %v", err)
	}
	if resp.CacheHits != 100 {
		t.Errorf("CacheHits = %d, want 100", resp.CacheHits)
	}
}

func TestClient_AuthenticationError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(401)
		json.NewEncoder(w).Encode(map[string]string{"error": "unauthorized"})
	}))
	defer server.Close()

	client, _ := NewClient("bad-key", WithBaseURL(server.URL), WithRetries(0))
	_, err := client.Embed(context.Background(), "test")
	if err == nil {
		t.Error("expected error")
	}
	var authErr *AuthenticationError
	if !errors.As(err, &authErr) {
		t.Errorf("expected AuthenticationError, got %T", err)
	}
}

func TestClient_RetryOnServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			w.WriteHeader(500)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"vector": []float64{0.1}, "cached": false, "dimensions": 1,
		})
	}))
	defer server.Close()

	client, _ := NewClient("test-key", WithBaseURL(server.URL), WithRetries(2))
	resp, err := client.Embed(context.Background(), "test")
	if err != nil {
		t.Fatalf("expected success after retry, got: %v", err)
	}
	if resp == nil {
		t.Error("expected response")
	}
	if attempts != 2 {
		t.Errorf("attempts = %d, want 2", attempts)
	}
}
