import { describe, it, expect } from 'vitest';
import * as lib from '../src/index';

describe('Public Exports', () => {
  it('exports createClient', () => {
    expect(typeof lib.createClient).toBe('function');
  });

  it('exports error classes', () => {
    expect(lib.VectorEmbedError).toBeDefined();
    expect(lib.AuthenticationError).toBeDefined();
    expect(lib.RateLimitError).toBeDefined();
    expect(lib.ValidationError).toBeDefined();
    expect(lib.ServerError).toBeDefined();
    expect(lib.NetworkError).toBeDefined();
  });

  it('exports VERSION', () => {
    expect(typeof lib.VERSION).toBe('string');
  });
});
