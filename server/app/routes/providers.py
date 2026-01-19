# server/app/routes/providers.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import secrets
from app.database import get_db
from app.models import ApiKey, Provider
from app.schemas import ProviderRequest, ProviderResponse
from app.auth import get_current_api_key
from app.crypto import encrypt_api_key
from app.passthrough import validate_endpoint

router = APIRouter(prefix="/v1", tags=["providers"])


@router.post("/provider", response_model=ProviderResponse)
def create_provider(
    request: ProviderRequest,
    api_key: ApiKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """Configure BYOK provider."""
    # Validate endpoint
    try:
        validate_endpoint(request.endpoint)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if user already has a provider (limit 1 for now)
    existing = db.query(Provider).filter(
        Provider.user_id == api_key.user_id
    ).first()

    if existing:
        # Update existing
        existing.name = request.name
        existing.endpoint = request.endpoint
        existing.api_key_encrypted = encrypt_api_key(request.api_key)
        existing.request_template = request.request_template
        existing.response_path = request.response_path
        db.commit()
        return ProviderResponse(provider_id=existing.id)

    # Create new
    provider = Provider(
        id=f"prov_{secrets.token_hex(12)}",
        user_id=api_key.user_id,
        name=request.name,
        endpoint=request.endpoint,
        api_key_encrypted=encrypt_api_key(request.api_key),
        request_template=request.request_template,
        response_path=request.response_path,
    )
    db.add(provider)
    db.commit()

    return ProviderResponse(provider_id=provider.id)
