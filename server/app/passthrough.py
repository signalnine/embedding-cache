# server/app/passthrough.py
import ipaddress
import httpx
from urllib.parse import urlparse
from jsonpath_ng import parse as jsonpath_parse
from app.crypto import decrypt_api_key


ALLOWED_HOSTS = [
    "api.openai.com",
    "api.cohere.ai",
    "api.voyageai.com",
    "api.mistral.ai",
    "api.together.xyz",
]


def is_allowed_host(host: str) -> bool:
    """Check if host is in whitelist and not a private IP."""
    # Check against whitelist
    if host in ALLOWED_HOSTS:
        return True

    # Block localhost
    if host in ("localhost", "127.0.0.1", "::1"):
        return False

    # Block private IP ranges
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        # Not an IP address, check if it's in whitelist
        pass

    return False


def validate_endpoint(endpoint: str) -> None:
    """Validate BYOK endpoint URL.

    Raises ValueError if invalid.
    """
    parsed = urlparse(endpoint)

    if parsed.scheme != "https":
        raise ValueError("HTTPS required for BYOK endpoints")

    if not is_allowed_host(parsed.hostname):
        raise ValueError(f"Host not in allowed list: {parsed.hostname}")


def extract_embedding(response: dict, json_path: str) -> list[float]:
    """Extract embedding from response using JSONPath."""
    expr = jsonpath_parse(json_path)
    matches = expr.find(response)

    if not matches:
        raise ValueError(f"No match for JSONPath: {json_path}")

    return matches[0].value


async def call_byok_provider(
    endpoint: str,
    api_key_encrypted: str,
    request_template: dict,
    response_path: str,
    text: str,
) -> list[float]:
    """Call BYOK provider and return embedding."""
    # Decrypt API key
    api_key = decrypt_api_key(api_key_encrypted)

    # Build request body from template
    body = _substitute_template(request_template, text)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            endpoint,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        response.raise_for_status()

    return extract_embedding(response.json(), response_path)


def _substitute_template(template: dict, text: str) -> dict:
    """Replace $TEXT placeholders in template."""
    result = {}
    for key, value in template.items():
        if isinstance(value, str):
            result[key] = value.replace("$TEXT", text)
        elif isinstance(value, dict):
            result[key] = _substitute_template(value, text)
        elif isinstance(value, list):
            result[key] = [
                v.replace("$TEXT", text) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value
    return result
