"""Pre-seeded embeddings lookup."""

from pathlib import Path

# Database filename constant
PRESEED_DB_NAME = "preseed_v1.5.db"


def get_preseed_db_path() -> Path:
    """Get path to bundled preseed database.

    Returns:
        Path to preseed DB location (may or may not exist)
    """
    # Get path relative to this module
    module_dir = Path(__file__).parent
    return module_dir / "data" / PRESEED_DB_NAME


def preseed_db_exists() -> bool:
    """Check if preseed database exists.

    Returns:
        True if preseed DB file exists, False otherwise
    """
    return get_preseed_db_path().exists()
