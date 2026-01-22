package vectorembed

import (
	"encoding/json"
	"testing"
)

func TestEmbedResponse_Unmarshal(t *testing.T) {
	data := `{"vector":[0.1,0.2,0.3],"cached":true,"dimensions":3}`
	var resp EmbedResponse
	if err := json.Unmarshal([]byte(data), &resp); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if len(resp.Vector) != 3 {
		t.Errorf("Vector length = %d, want 3", len(resp.Vector))
	}
	if !resp.Cached {
		t.Error("Cached should be true")
	}
	if resp.Dimensions != 3 {
		t.Errorf("Dimensions = %d, want 3", resp.Dimensions)
	}
}

func TestSearchResult_Unmarshal(t *testing.T) {
	data := `{"text_hash":"abc123","score":0.95,"text":"hello","model":"nomic-v1.5","hit_count":5}`
	var result SearchResult
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if result.TextHash != "abc123" {
		t.Errorf("TextHash = %q, want %q", result.TextHash, "abc123")
	}
	if result.Score != 0.95 {
		t.Errorf("Score = %f, want 0.95", result.Score)
	}
	if result.Text == nil || *result.Text != "hello" {
		t.Error("Text should be 'hello'")
	}
}

func TestSearchResult_UnmarshalNullText(t *testing.T) {
	data := `{"text_hash":"abc","score":0.9,"text":null,"model":"m","hit_count":1}`
	var result SearchResult
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if result.Text != nil {
		t.Error("Text should be nil")
	}
}

func TestSearchResponse_Unmarshal(t *testing.T) {
	data := `{"results":[{"text_hash":"a","score":0.9,"model":"m","hit_count":1}],"total":1,"search_time_ms":5.5}`
	var resp SearchResponse
	if err := json.Unmarshal([]byte(data), &resp); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if len(resp.Results) != 1 {
		t.Errorf("Results length = %d, want 1", len(resp.Results))
	}
	if resp.SearchTimeMs != 5.5 {
		t.Errorf("SearchTimeMs = %f, want 5.5", resp.SearchTimeMs)
	}
}

func TestStatsResponse_Unmarshal(t *testing.T) {
	data := `{"cache_hits":100,"cache_misses":10,"total_cached":110}`
	var resp StatsResponse
	if err := json.Unmarshal([]byte(data), &resp); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if resp.CacheHits != 100 {
		t.Errorf("CacheHits = %d, want 100", resp.CacheHits)
	}
}

func TestEmbedRequest_Marshal(t *testing.T) {
	req := embedRequest{Text: "hello", Model: "nomic-v1.5"}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	want := `{"text":"hello","model":"nomic-v1.5"}`
	if string(data) != want {
		t.Errorf("Marshal = %s, want %s", data, want)
	}
}

func TestEmbedRequest_MarshalOmitModel(t *testing.T) {
	req := embedRequest{Text: "hello"}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	want := `{"text":"hello"}`
	if string(data) != want {
		t.Errorf("Marshal = %s, want %s", data, want)
	}
}

func TestSearchRequest_Marshal(t *testing.T) {
	req := searchRequest{Text: "query", TopK: 10, MinScore: 0.5, IncludeText: true}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	// Verify key fields are present
	var m map[string]interface{}
	json.Unmarshal(data, &m)
	if m["text"] != "query" {
		t.Error("text field missing or wrong")
	}
	if m["top_k"] != float64(10) {
		t.Error("top_k field missing or wrong")
	}
}
