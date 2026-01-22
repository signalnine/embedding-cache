import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createFetcher } from '../src/fetch';
import { AuthenticationError, RateLimitError, ServerError, NetworkError } from '../src/errors';

describe('Fetch Wrapper', () => {
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch);
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const config = {
    apiKey: 'test_key',
    baseUrl: 'https://api.example.com',
    timeout: 5000,
    retries: 2,
  };

  it('adds authorization header', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ data: 'test' }),
    });

    const fetcher = createFetcher(config);
    await fetcher('/v1/test', { method: 'GET' });

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.example.com/v1/test',
      expect.objectContaining({
        headers: expect.objectContaining({
          'Authorization': 'Bearer test_key',
        }),
      })
    );
  });

  it('throws AuthenticationError on 401', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: () => Promise.resolve({ error: 'Invalid API key' }),
    });

    const fetcher = createFetcher(config);
    await expect(fetcher('/v1/test', { method: 'GET' }))
      .rejects.toThrow(AuthenticationError);
  });

  it('throws RateLimitError on 429 with retryAfter', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 429,
      headers: new Headers({ 'Retry-After': '60' }),
      json: () => Promise.resolve({ error: 'Rate limited' }),
    });

    const fetcher = createFetcher({ ...config, retries: 0 });

    try {
      await fetcher('/v1/test', { method: 'GET' });
      expect.fail('Should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(RateLimitError);
      expect((error as RateLimitError).retryAfter).toBe(60);
    }
  });

  it('retries on 500 errors', async () => {
    mockFetch
      .mockResolvedValueOnce({ ok: false, status: 500, json: () => Promise.resolve({}) })
      .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({ data: 'success' }) });

    const fetcher = createFetcher({ ...config, retries: 1 });
    const result = await fetcher('/v1/test', { method: 'GET' });

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result).toEqual({ data: 'success' });
  });

  it('respects AbortSignal', async () => {
    const controller = new AbortController();
    controller.abort();

    const fetcher = createFetcher(config);

    await expect(fetcher('/v1/test', { method: 'GET' }, { signal: controller.signal }))
      .rejects.toThrow();
  });

  it('does not retry on 400 errors', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ error: 'Bad request' }),
    });

    const fetcher = createFetcher({ ...config, retries: 2 });

    await expect(fetcher('/v1/test', { method: 'GET' }))
      .rejects.toThrow();

    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});
