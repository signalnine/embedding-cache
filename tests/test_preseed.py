"""Tests for preseed module."""

from unittest.mock import patch


class TestPreseedPaths:
    def test_get_preseed_db_path_returns_path_in_package(self):
        from vector_embed_cache.preseed import get_preseed_db_path, PRESEED_DB_NAME

        path = get_preseed_db_path()
        assert path is not None
        assert "vector_embed_cache" in str(path)
        assert path.name == PRESEED_DB_NAME

    def test_preseed_db_exists_returns_false_when_missing(self):
        from vector_embed_cache.preseed import preseed_db_exists

        # Mock Path.exists() to return False
        with patch("vector_embed_cache.preseed.get_preseed_db_path") as mock_path:
            mock_path.return_value.exists.return_value = False
            assert preseed_db_exists() is False

    def test_preseed_db_exists_returns_true_when_present(self):
        from vector_embed_cache.preseed import preseed_db_exists

        # Mock Path.exists() to return True
        with patch("vector_embed_cache.preseed.get_preseed_db_path") as mock_path:
            mock_path.return_value.exists.return_value = True
            assert preseed_db_exists() is True

    def test_get_preseed_db_path_is_in_data_directory(self):
        from vector_embed_cache.preseed import get_preseed_db_path

        path = get_preseed_db_path()
        assert path.parent.name == "data"
