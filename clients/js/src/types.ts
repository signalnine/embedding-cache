/** Client configuration options */
export interface ClientOptions {
  /** API key for authentication (required) */
  apiKey: string;
  /** Base URL for API (optional, defaults to production) */
  baseUrl?: string;
  /** Default model for embeddings (optional, defaults to nomic-v1.5) */
  model?: string;
  /** Request timeout in milliseconds (optional, defaults to 30000) */
  timeout?: number;
  /** Number of retry attempts (optional, defaults to 3) */
  retries?: number;
}

/** Per-request options that override client defaults */
export interface RequestOptions {
  /** Override default model */
  model?: string;
  /** Override default timeout */
  timeout?: number;
  /** AbortSignal for request cancellation */
  signal?: AbortSignal;
}

/** Single embedding response */
export interface EmbedResponse {
  /** Embedding vector */
  vector: number[];
  /** Whether result came from cache */
  cached: boolean;
  /** Vector dimensions */
  dimensions: number;
}

/** Batch embedding response */
export interface EmbedBatchResponse {
  /** Array of embedding vectors */
  embeddings: number[][];
  /** Cache status for each embedding */
  cached: boolean[];
  /** Vector dimensions */
  dimensions: number;
}

/** Search request options */
export interface SearchOptions {
  /** Text to search for (will be embedded) */
  text?: string;
  /** Pre-computed vector to search with */
  vector?: number[];
  /** Model for embedding lookup */
  model?: string;
  /** Number of results (default 10, max 100) */
  topK?: number;
  /** Minimum similarity score (0-1) */
  minScore?: number;
  /** Include original text in results */
  includeText?: boolean;
  /** Include vectors in results */
  includeVectors?: boolean;
}

/** Individual search result */
export interface SearchResult {
  /** Hash of the original text */
  textHash: string;
  /** Similarity score (0-1) */
  score: number;
  /** Original text (if includeText was true) */
  text?: string;
  /** Model used for embedding */
  model: string;
  /** Number of cache hits */
  hitCount: number;
  /** Vector (if includeVectors was true) */
  vector?: number[];
}

/** Search response */
export interface SearchResponse {
  /** Search results */
  results: SearchResult[];
  /** Total number of results */
  total: number;
  /** Search duration in milliseconds */
  searchTimeMs: number;
  /** Model used */
  model: string;
  /** Vector dimensions */
  dimensions: number;
}

/** Cache statistics */
export interface StatsResponse {
  /** Number of cache hits */
  cacheHits: number;
  /** Number of cache misses */
  cacheMisses: number;
  /** Total cached embeddings */
  totalCached: number;
}

/** Client interface */
export interface VectorEmbedClient {
  /** Generate embedding for single text */
  embed(text: string, options?: RequestOptions): Promise<EmbedResponse>;
  /** Generate embeddings for multiple texts */
  embedBatch(texts: string[], options?: RequestOptions): Promise<EmbedBatchResponse>;
  /** Search for similar embeddings */
  search(options: SearchOptions & RequestOptions): Promise<SearchResponse>;
  /** Get cache statistics */
  stats(options?: RequestOptions): Promise<StatsResponse>;
}
