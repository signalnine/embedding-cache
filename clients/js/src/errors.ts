export class VectorEmbedError extends Error {
  readonly code: string;
  readonly status?: number;
  readonly cause?: Error;

  constructor(message: string, code: string, status?: number, cause?: Error) {
    super(message);
    this.name = 'VectorEmbedError';
    this.code = code;
    this.status = status;
    this.cause = cause;
  }
}

export class AuthenticationError extends VectorEmbedError {
  constructor(message: string = 'Authentication failed') {
    super(message, 'authentication_error', 401);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends VectorEmbedError {
  readonly retryAfter?: number;

  constructor(message: string = 'Rate limit exceeded', retryAfter?: number) {
    super(message, 'rate_limit_error', 429);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class ValidationError extends VectorEmbedError {
  constructor(message: string = 'Validation failed') {
    super(message, 'validation_error', 400);
    this.name = 'ValidationError';
  }
}

export class ServerError extends VectorEmbedError {
  constructor(message: string = 'Server error', status: number = 500) {
    super(message, 'server_error', status);
    this.name = 'ServerError';
  }
}

export class NetworkError extends VectorEmbedError {
  constructor(message: string = 'Network error', cause?: Error) {
    super(message, 'network_error', undefined, cause);
    this.name = 'NetworkError';
  }
}
