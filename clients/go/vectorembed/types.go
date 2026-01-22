package vectorembed

// Request types (unexported, used internally)

// embedRequest is the request body for single text embedding.
type embedRequest struct {
	Text  string `json:"text"`
	Model string `json:"model,omitempty"`
}

// embedBatchRequest is the request body for batch text embedding.
type embedBatchRequest struct {
	Texts []string `json:"texts"`
	Model string   `json:"model,omitempty"`
}

// searchRequest is the request body for similarity search.
type searchRequest struct {
	Text        string  `json:"text"`
	TopK        int     `json:"top_k,omitempty"`
	MinScore    float64 `json:"min_score,omitempty"`
	IncludeText bool    `json:"include_text,omitempty"`
}

// Response types (exported)

// EmbedResponse is the response from an embedding request.
type EmbedResponse struct {
	Vector     []float64 `json:"vector"`
	Cached     bool      `json:"cached"`
	Dimensions int       `json:"dimensions"`
}

// SearchResult represents a single result from a similarity search.
type SearchResult struct {
	TextHash string  `json:"text_hash"`
	Score    float64 `json:"score"`
	Text     *string `json:"text,omitempty"`
	Model    string  `json:"model"`
	HitCount int     `json:"hit_count"`
}

// SearchResponse is the response from a similarity search request.
type SearchResponse struct {
	Results      []SearchResult `json:"results"`
	Total        int            `json:"total"`
	SearchTimeMs float64        `json:"search_time_ms"`
}

// StatsResponse is the response from a cache stats request.
type StatsResponse struct {
	CacheHits   int `json:"cache_hits"`
	CacheMisses int `json:"cache_misses"`
	TotalCached int `json:"total_cached"`
}
