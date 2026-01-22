import {
  VectorEmbedError,
  AuthenticationError,
  RateLimitError,
  ValidationError,
  ServerError,
  NetworkError,
} from './errors';
import type { RequestOptions } from './types';

interface FetchConfig {
  apiKey: string;
  baseUrl: string;
  timeout: number;
  retries: number;
}

const RETRYABLE_STATUSES = [429, 500, 502, 503, 504];

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function calculateBackoff(attempt: number): number {
  const base = 1000;
  const max = 30000;
  const exponential = base * Math.pow(2, attempt);
  const capped = Math.min(exponential, max);
  const jitter = capped * 0.2 * (Math.random() * 2 - 1);
  return capped + jitter;
}

async function mapHttpError(response: Response): Promise<VectorEmbedError> {
  let body: { error?: string } = {};
  try {
    body = await response.json();
  } catch {
    // Ignore JSON parse errors
  }

  const message = body.error || `HTTP ${response.status}`;

  switch (response.status) {
    case 401:
      return new AuthenticationError(message);
    case 429: {
      const retryAfter = response.headers.get('Retry-After');
      return new RateLimitError(message, retryAfter ? parseInt(retryAfter, 10) : undefined);
    }
    case 400:
      return new ValidationError(message);
    default:
      if (response.status >= 500) {
        return new ServerError(message, response.status);
      }
      return new VectorEmbedError(message, 'http_error', response.status);
  }
}

export function createFetcher(config: FetchConfig) {
  return async function fetcher<T>(
    path: string,
    init: RequestInit,
    requestOptions?: RequestOptions
  ): Promise<T> {
    const url = `${config.baseUrl}${path}`;
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= config.retries; attempt++) {
      const controller = new AbortController();
      const timeout = requestOptions?.timeout ?? config.timeout;
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      // Check user signal before attempt
      const userSignal = requestOptions?.signal;
      if (userSignal?.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }

      // Link user signal to our controller
      const abortHandler = () => controller.abort();
      userSignal?.addEventListener('abort', abortHandler);

      try {
        const response = await fetch(url, {
          ...init,
          headers: {
            'Authorization': `Bearer ${config.apiKey}`,
            'Content-Type': 'application/json',
            'Accept-Version': 'v1',
            ...init.headers,
          },
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        userSignal?.removeEventListener('abort', abortHandler);

        if (!response.ok) {
          const error = await mapHttpError(response);

          // Don't retry client errors (except rate limit)
          if (!RETRYABLE_STATUSES.includes(response.status)) {
            throw error;
          }

          // Check if we should retry
          if (attempt < config.retries) {
            lastError = error;
            const delay = error instanceof RateLimitError && error.retryAfter
              ? error.retryAfter * 1000
              : calculateBackoff(attempt);
            await sleep(delay);
            continue;
          }

          throw error;
        }

        return await response.json();
      } catch (error) {
        clearTimeout(timeoutId);
        userSignal?.removeEventListener('abort', abortHandler);

        // Don't retry user cancellation
        if (userSignal?.aborted || (error instanceof DOMException && error.name === 'AbortError')) {
          throw error;
        }

        // Network errors are retryable
        if (error instanceof TypeError && attempt < config.retries) {
          lastError = new NetworkError('Network request failed', error);
          await sleep(calculateBackoff(attempt));
          continue;
        }

        if (error instanceof VectorEmbedError) {
          throw error;
        }

        throw new NetworkError('Network request failed', error instanceof Error ? error : undefined);
      }
    }

    throw lastError || new NetworkError('Request failed after retries');
  };
}
