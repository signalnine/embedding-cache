# JavaScript/TypeScript Client Library Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a TypeScript client library for the vector-embed-cache API that works in Node.js, browsers, and edge runtimes.

**Architecture:** Zero-dependency client using native fetch with factory pattern API. Dual CJS/ESM builds for maximum compatibility. Typed error classes for structured error handling.

**Tech Stack:** TypeScript, tsup (bundler), vitest (testing), native fetch

---

## Design Decisions (Consensus-Driven)

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Language priority | JS/TS first | Target users in AI/ML app ecosystem |
| Package structure | Dual CJS/ESM | Pragmatic compatibility, Node 18+ minimum |
| HTTP client | Native fetch | Zero dependencies, universal runtime support |
| API pattern | Factory | `createClient({apiKey})` - modern conventions |
| Error handling | Typed subclasses | `AuthenticationError`, `RateLimitError`, etc. |
| Repository | Monorepo `clients/js/` | API sync, atomic PRs, single CI |
| Package name | `@vector-embed/client` | Scoped, clarifies role, expandable |

---

## Public API

### Basic Usage

```typescript
import { createClient } from '@vector-embed/client';

const client = createClient({
  apiKey: 'vec_your_api_key',
  baseUrl: 'https://api.example.com', // optional, defaults to production
});

// Single embedding
const embedding = await client.embed('hello world');
console.log(embedding.vector);  // number[]
console.log(embedding.cached);  // boolean

// Batch embeddings
const results = await client.embedBatch(['hello', 'world']);

// Similarity search
const matches = await client.search({
  text: 'find similar',
  topK: 10,
  minScore: 0.5,
});

// Statistics
const stats = await client.stats();
```

### Configuration Options

```typescript
interface ClientOptions {
  apiKey: string;              // Required: API key for authentication
  baseUrl?: string;            // Optional: API base URL
  model?: string;              // Optional: default model (nomic-v1.5)
  timeout?: number;            // Optional: request timeout in ms (30000)
  retries?: number;            // Optional: retry count for failures (3)
}

// Per-request options (can override client defaults)
interface RequestOptions {
  model?: string;              // Override default model
  timeout?: number;            // Override default timeout
  signal?: AbortSignal;        // User-provided cancellation signal
}
```

### Cancellation Support

All async methods accept an optional `AbortSignal` for user-controlled cancellation:

```typescript
const controller = new AbortController();

// Cancel after 5 seconds
setTimeout(() => controller.abort(), 5000);

try {
  const result = await client.embed('hello', { signal: controller.signal });
} catch (error) {
  if (error.name === 'AbortError') {
    console.log('Request cancelled');
  }
}

// React cleanup example
useEffect(() => {
  const controller = new AbortController();
  client.embed(text, { signal: controller.signal }).then(setResult);
  return () => controller.abort();
}, [text]);
```

### Response Types

```typescript
interface EmbedResponse {
  vector: number[];
  cached: boolean;
  dimensions: number;
}

interface SearchResult {
  textHash: string;
  score: number;
  text?: string;             // Present only if includeText: true in request
  model: string;
  hitCount: number;
}

interface SearchResponse {
  results: SearchResult[];
  total: number;
  searchTimeMs: number;
}

interface StatsResponse {
  cacheHits: number;
  cacheMisses: number;
  totalCached: number;
}
```

---

## Error Handling

```typescript
class VectorEmbedError extends Error {
  code: string;
  status?: number;
  cause?: Error;
}

class AuthenticationError extends VectorEmbedError {}  // 401
class RateLimitError extends VectorEmbedError {        // 429
  retryAfter?: number;
}
class ValidationError extends VectorEmbedError {}      // 400
class ServerError extends VectorEmbedError {}          // 5xx
class NetworkError extends VectorEmbedError {}         // fetch failures
```

Usage:
```typescript
try {
  await client.embed('text');
} catch (error) {
  if (error instanceof RateLimitError) {
    console.log(`Rate limited. Retry after ${error.retryAfter}s`);
  } else if (error instanceof AuthenticationError) {
    console.log('Invalid API key');
  }
}
```

---

## Project Structure

```
clients/js/
├── src/
│   ├── index.ts          # Main exports
│   ├── client.ts         # createClient factory
│   ├── fetch.ts          # fetch wrapper (auth, errors, retries)
│   ├── errors.ts         # Error class definitions
│   └── types.ts          # TypeScript interfaces
├── tests/
│   ├── client.test.ts    # Unit tests
│   └── integration.test.ts # Integration tests (needs server)
├── package.json
├── tsconfig.json
├── tsup.config.ts        # Build configuration
├── vitest.config.ts      # Test configuration
└── README.md
```

---

## Build Configuration

### package.json

```json
{
  "name": "@vector-embed/client",
  "version": "0.1.0",
  "description": "JavaScript/TypeScript client for vector-embed-cache API",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "require": "./dist/index.cjs",
      "types": "./dist/index.d.ts"
    }
  },
  "files": ["dist"],
  "engines": {
    "node": ">=18.0.0"
  },
  "keywords": [
    "embeddings",
    "vector",
    "cache",
    "ai",
    "ml",
    "vector-embed-cache"
  ],
  "scripts": {
    "build": "tsup",
    "test": "vitest run",
    "test:watch": "vitest",
    "typecheck": "tsc --noEmit"
  },
  "devDependencies": {
    "tsup": "^8.0.0",
    "typescript": "^5.0.0",
    "vitest": "^1.0.0"
  }
}
```

### tsup.config.ts

```typescript
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  clean: true,
  minify: true,
});
```

---

## Internal Fetch Wrapper

The fetch wrapper handles:
- Authorization header injection
- API versioning header (`Accept-Version: v1`)
- JSON response parsing
- HTTP error to typed error mapping
- Timeout via AbortController (per-attempt, not total)
- Retry with exponential backoff + jitter
- User-provided AbortSignal support

### Retry Semantics

```typescript
const RETRY_CONFIG = {
  maxRetries: 3,
  baseDelay: 1000,        // 1 second
  maxDelay: 30000,        // 30 seconds cap
  jitterFactor: 0.2,      // ±20% randomization

  // Retry only on these conditions:
  retryableStatuses: [429, 500, 502, 503, 504],
  retryableErrors: ['ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND'],
};

function calculateDelay(attempt: number): number {
  const exponential = RETRY_CONFIG.baseDelay * Math.pow(2, attempt);
  const capped = Math.min(exponential, RETRY_CONFIG.maxDelay);
  const jitter = capped * RETRY_CONFIG.jitterFactor * (Math.random() * 2 - 1);
  return capped + jitter;
}
```

**Retry behavior:**
- Respects `Retry-After` header when present (429 responses)
- Does NOT retry 4xx errors (except 429) - these are client errors
- Timeout applies per-attempt, not total (3 retries = up to 3x timeout)
- User-provided AbortSignal cancels immediately, skipping retries

### Fetch Wrapper Implementation

```typescript
async function request<T>(
  url: string,
  options: RequestInit,
  config: ClientConfig,
  requestOptions?: RequestOptions
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= config.retries; attempt++) {
    const controller = new AbortController();
    const timeout = requestOptions?.timeout ?? config.timeout;
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    // Link user's signal to our controller
    const userSignal = requestOptions?.signal;
    if (userSignal?.aborted) throw new Error('Aborted');
    userSignal?.addEventListener('abort', () => controller.abort());

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Authorization': `Bearer ${config.apiKey}`,
          'Content-Type': 'application/json',
          'Accept-Version': 'v1',
          ...options.headers,
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await mapHttpError(response);
        if (shouldRetry(error, attempt, config.retries)) {
          lastError = error;
          await sleep(getRetryDelay(error, attempt));
          continue;
        }
        throw error;
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (userSignal?.aborted) throw error; // Don't retry user cancellation
      if (shouldRetry(error, attempt, config.retries)) {
        lastError = error;
        await sleep(calculateDelay(attempt));
        continue;
      }
      throw error;
    }
  }

  throw lastError;
}
```

---

## CI Integration

Add to `.github/workflows/ci.yml`:

```yaml
  test-js-client:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: clients/js
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: clients/js/package-lock.json
      - run: npm ci
      - run: npm run typecheck
      - run: npm test
      - run: npm run build
```

---

## Batch Semantics

Batch operations have specific behavior:

- **Max batch size**: 100 texts (matches server limit)
- **Order guaranteed**: Results array matches input order
- **Partial failure**: Currently all-or-nothing (server rejects entire batch on validation error)
- **Client validation**: Validates non-empty strings and batch size before sending

```typescript
// Batch returns array in same order as input
const results = await client.embedBatch(['a', 'b', 'c']);
// results[0] is embedding for 'a', etc.
```

---

## Testing Strategy

### Unit Tests (vitest)

- **Fetch mocking**: Use `vitest.mock` or `msw` (Mock Service Worker)
- **Error mapping**: Test all HTTP status codes map to correct error types
- **Retry logic**: Test exponential backoff, jitter, Retry-After header handling
- **Cancellation**: Test AbortSignal propagation and cleanup
- **Input validation**: Test apiKey format, empty strings, batch size limits

### Integration Tests

- Run against local server instance (started in CI)
- Test full request/response cycle
- Verify response types match TypeScript interfaces

### Cross-Runtime Testing

- Node.js 18, 20 in CI matrix
- Consider adding Deno/Bun smoke tests

---

## Future Considerations

1. **Go client** - After JS client ships and stabilizes
2. **Python note** - Consensus flagged Python's absence as strategic concern; may want to reconsider after JS
3. **OpenAPI generation** - If API stabilizes, could auto-generate clients
4. **Browser bundle** - Could add UMD build for `<script>` tag usage if requested
5. **Streaming** - Not needed for current API, but design allows adding later
