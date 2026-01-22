import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createClient } from '../src/client';

describe('createClient', () => {
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch);
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('creates client with required apiKey', () => {
    const client = createClient({ apiKey: 'test_key' });
    expect(client).toBeDefined();
    expect(typeof client.embed).toBe('function');
    expect(typeof client.embedBatch).toBe('function');
    expect(typeof client.search).toBe('function');
    expect(typeof client.stats).toBe('function');
  });

  it('embed sends correct request', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        embedding: [0.1, 0.2, 0.3],
        cached: false,
        dimensions: 3,
      }),
    });

    const client = createClient({
      apiKey: 'test_key',
      baseUrl: 'https://api.test.com',
    });

    const result = await client.embed('hello world');

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.test.com/v1/embed',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ text: 'hello world', model: 'nomic-v1.5' }),
      })
    );
    expect(result.vector).toEqual([0.1, 0.2, 0.3]);
  });

  it('embedBatch sends correct request', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        embeddings: [[0.1], [0.2]],
        cached: [false, true],
        dimensions: 1,
      }),
    });

    const client = createClient({ apiKey: 'test_key', baseUrl: 'https://api.test.com' });
    const result = await client.embedBatch(['hello', 'world']);

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.test.com/v1/embed/batch',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ texts: ['hello', 'world'], model: 'nomic-v1.5' }),
      })
    );
    expect(result.embeddings).toEqual([[0.1], [0.2]]);
  });

  it('search sends correct request', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        results: [],
        total: 0,
        search_time_ms: 5,
        model: 'nomic-v1.5',
        dimensions: 768,
      }),
    });

    const client = createClient({ apiKey: 'test_key', baseUrl: 'https://api.test.com' });
    await client.search({ text: 'query', topK: 5 });

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.test.com/v1/search',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          query_text: 'query',
          model: 'nomic-v1.5',
          top_k: 5,
        }),
      })
    );
  });

  it('stats sends correct request', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        cache_hits: 10,
        cache_misses: 5,
        total_cached: 15,
      }),
    });

    const client = createClient({ apiKey: 'test_key', baseUrl: 'https://api.test.com' });
    const result = await client.stats();

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.test.com/v1/stats',
      expect.objectContaining({ method: 'GET' })
    );
    expect(result.cacheHits).toBe(10);
  });

  it('validates empty text', async () => {
    const client = createClient({ apiKey: 'test_key' });
    await expect(client.embed('')).rejects.toThrow('Text cannot be empty');
  });

  it('validates batch size', async () => {
    const client = createClient({ apiKey: 'test_key' });
    const texts = Array(101).fill('text');
    await expect(client.embedBatch(texts)).rejects.toThrow('exceeds maximum');
  });
});
