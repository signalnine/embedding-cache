import { createFetcher } from './fetch';
import { ValidationError } from './errors';
import type {
  ClientOptions,
  RequestOptions,
  EmbedResponse,
  EmbedBatchResponse,
  SearchOptions,
  SearchResponse,
  StatsResponse,
  VectorEmbedClient,
} from './types';

const DEFAULT_BASE_URL = 'https://api.vector-embed.io';
const DEFAULT_MODEL = 'nomic-v1.5';
const DEFAULT_TIMEOUT = 30000;
const DEFAULT_RETRIES = 3;
const MAX_BATCH_SIZE = 100;

interface ApiEmbedResponse {
  embedding: number[];
  cached: boolean;
  dimensions: number;
}

interface ApiEmbedBatchResponse {
  embeddings: number[][];
  cached: boolean[];
  dimensions: number;
}

interface ApiSearchResponse {
  results: Array<{
    text_hash: string;
    score: number;
    text?: string;
    model: string;
    hit_count: number;
    vector?: number[];
  }>;
  total: number;
  search_time_ms: number;
  model: string;
  dimensions: number;
}

interface ApiStatsResponse {
  cache_hits: number;
  cache_misses: number;
  total_cached: number;
}

export function createClient(options: ClientOptions): VectorEmbedClient {
  const config = {
    apiKey: options.apiKey,
    baseUrl: options.baseUrl ?? DEFAULT_BASE_URL,
    timeout: options.timeout ?? DEFAULT_TIMEOUT,
    retries: options.retries ?? DEFAULT_RETRIES,
  };

  const defaultModel = options.model ?? DEFAULT_MODEL;
  const fetcher = createFetcher(config);

  return {
    async embed(text: string, requestOptions?: RequestOptions): Promise<EmbedResponse> {
      if (!text || text.trim() === '') {
        throw new ValidationError('Text cannot be empty');
      }

      const model = requestOptions?.model ?? defaultModel;
      const response = await fetcher<ApiEmbedResponse>(
        '/v1/embed',
        {
          method: 'POST',
          body: JSON.stringify({ text, model }),
        },
        requestOptions
      );

      return {
        vector: response.embedding,
        cached: response.cached,
        dimensions: response.dimensions,
      };
    },

    async embedBatch(texts: string[], requestOptions?: RequestOptions): Promise<EmbedBatchResponse> {
      if (texts.length === 0) {
        throw new ValidationError('Texts array cannot be empty');
      }
      if (texts.length > MAX_BATCH_SIZE) {
        throw new ValidationError(`Batch size ${texts.length} exceeds maximum of ${MAX_BATCH_SIZE}`);
      }
      if (texts.some(t => !t || t.trim() === '')) {
        throw new ValidationError('All texts must be non-empty strings');
      }

      const model = requestOptions?.model ?? defaultModel;
      const response = await fetcher<ApiEmbedBatchResponse>(
        '/v1/embed/batch',
        {
          method: 'POST',
          body: JSON.stringify({ texts, model }),
        },
        requestOptions
      );

      return {
        embeddings: response.embeddings,
        cached: response.cached,
        dimensions: response.dimensions,
      };
    },

    async search(options: SearchOptions & RequestOptions): Promise<SearchResponse> {
      if (!options.text && !options.vector) {
        throw new ValidationError('Either text or vector must be provided');
      }
      if (options.text && options.vector) {
        throw new ValidationError('Cannot provide both text and vector');
      }

      const model = options.model ?? defaultModel;
      const body: Record<string, unknown> = {};

      if (options.text) body.query_text = options.text;
      if (options.vector) body.query_vector = options.vector;
      body.model = model;
      if (options.topK) body.top_k = options.topK;
      if (options.minScore !== undefined) body.min_score = options.minScore;
      if (options.includeText !== undefined) body.include_text = options.includeText;
      if (options.includeVectors !== undefined) body.include_vectors = options.includeVectors;

      const response = await fetcher<ApiSearchResponse>(
        '/v1/search',
        {
          method: 'POST',
          body: JSON.stringify(body),
        },
        options
      );

      return {
        results: response.results.map(r => ({
          textHash: r.text_hash,
          score: r.score,
          text: r.text,
          model: r.model,
          hitCount: r.hit_count,
          vector: r.vector,
        })),
        total: response.total,
        searchTimeMs: response.search_time_ms,
        model: response.model,
        dimensions: response.dimensions,
      };
    },

    async stats(requestOptions?: RequestOptions): Promise<StatsResponse> {
      const response = await fetcher<ApiStatsResponse>(
        '/v1/stats',
        { method: 'GET' },
        requestOptions
      );

      return {
        cacheHits: response.cache_hits,
        cacheMisses: response.cache_misses,
        totalCached: response.total_cached,
      };
    },
  };
}
