import { describe, it, expect } from 'vitest';
import {
  VectorEmbedError,
  AuthenticationError,
  RateLimitError,
  ValidationError,
  ServerError,
  NetworkError,
} from '../src/errors';

describe('Error Classes', () => {
  it('VectorEmbedError has code and status properties', () => {
    const error = new VectorEmbedError('test message', 'test_code', 400);
    expect(error.message).toBe('test message');
    expect(error.code).toBe('test_code');
    expect(error.status).toBe(400);
    expect(error).toBeInstanceOf(Error);
  });

  it('AuthenticationError is instance of VectorEmbedError', () => {
    const error = new AuthenticationError('invalid key');
    expect(error).toBeInstanceOf(VectorEmbedError);
    expect(error.status).toBe(401);
  });

  it('RateLimitError has retryAfter property', () => {
    const error = new RateLimitError('rate limited', 60);
    expect(error.retryAfter).toBe(60);
    expect(error.status).toBe(429);
  });

  it('ValidationError has status 400', () => {
    const error = new ValidationError('invalid input');
    expect(error.status).toBe(400);
  });

  it('ServerError has status 500', () => {
    const error = new ServerError('server error', 503);
    expect(error.status).toBe(503);
  });

  it('NetworkError wraps original error as cause', () => {
    const cause = new Error('ECONNREFUSED');
    const error = new NetworkError('connection failed', cause);
    expect(error.cause).toBe(cause);
  });
});
