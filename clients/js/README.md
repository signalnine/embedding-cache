# @vector-embed/client

JavaScript/TypeScript client for the [vector-embed-cache](https://github.com/signalnine/embedding-cache) API.

## Installation

```bash
npm install @vector-embed/client
```

## Quick Start

```typescript
import { createClient } from '@vector-embed/client';

const client = createClient({
  apiKey: 'your_api_key',
});

// Single embedding
const result = await client.embed('hello world');
console.log(result.vector);  // number[]
console.log(result.cached);  // boolean

// Batch embeddings
const batch = await client.embedBatch(['hello', 'world']);
console.log(batch.embeddings);  // number[][]

// Similarity search
const matches = await client.search({
  text: 'find similar',
  topK: 10,
});

// Statistics
const stats = await client.stats();
```

## Configuration

```typescript
const client = createClient({
  apiKey: 'your_api_key',      // Required
  baseUrl: 'https://...',      // Optional: custom API URL
  model: 'nomic-v1.5',         // Optional: default model
  timeout: 30000,              // Optional: timeout in ms
  retries: 3,                  // Optional: retry attempts
});
```

## Per-Request Options

Override defaults on individual requests:

```typescript
const result = await client.embed('text', {
  model: 'openai:text-embedding-3-small',
  timeout: 60000,
  signal: controller.signal,  // AbortSignal for cancellation
});
```

## Cancellation

```typescript
const controller = new AbortController();

// Cancel after 5 seconds
setTimeout(() => controller.abort(), 5000);

try {
  await client.embed('text', { signal: controller.signal });
} catch (error) {
  if (error.name === 'AbortError') {
    console.log('Request cancelled');
  }
}
```

## Error Handling

```typescript
import {
  AuthenticationError,
  RateLimitError,
  ValidationError,
} from '@vector-embed/client';

try {
  await client.embed('text');
} catch (error) {
  if (error instanceof RateLimitError) {
    console.log(`Retry after ${error.retryAfter} seconds`);
  } else if (error instanceof AuthenticationError) {
    console.log('Invalid API key');
  }
}
```

## API Reference

### `createClient(options)`

Creates a new client instance.

**Options:**
- `apiKey` (required): API key for authentication
- `baseUrl`: API base URL (default: production)
- `model`: Default model (default: 'nomic-v1.5')
- `timeout`: Request timeout in ms (default: 30000)
- `retries`: Retry attempts (default: 3)

### `client.embed(text, options?)`

Generate embedding for a single text.

### `client.embedBatch(texts, options?)`

Generate embeddings for multiple texts (max 100).

### `client.search(options)`

Search for similar embeddings by text or vector.

### `client.stats(options?)`

Get cache statistics.

## Requirements

- Node.js 18+ (uses native fetch)
- Works in browsers and edge runtimes (Deno, Bun, Cloudflare Workers)

## License

MIT
