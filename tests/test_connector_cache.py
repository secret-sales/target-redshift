"""Unit tests for the RedshiftConnector reflection cache.

These tests run without a live Redshift cluster: the SQLAlchemy engine and the
``Table`` autoload path are patched out, so they only exercise the caching and
cache-invalidation logic added for the performance optimisation work.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from singer_sdk.connectors import SQLConnector

from target_redshift.connector import RedshiftConnector

CONFIG = {
    "host": "localhost",
    "port": 5439,
    "dbname": "test",
    "user": "test",
    "password": "test",
    "ssl_enable": False,
    "ssl_mode": "verify-ca",
    "default_target_schema": "public",
}


@pytest.fixture()
def connector() -> RedshiftConnector:
    """Return a RedshiftConnector without creating a real engine/connection."""
    return RedshiftConnector(config=CONFIG)


def _fake_table(columns: dict[str, object] | None = None) -> MagicMock:
    """Build a stand-in for a reflected SQLAlchemy Table.

    ``table.columns[name].type`` resolves for any name present in ``columns``;
    missing names raise ``KeyError`` just like a real column collection.
    """
    table = MagicMock(name="Table")
    columns = columns or {}
    column_objs = {}
    for name, type_ in columns.items():
        col = MagicMock(name=f"Column({name})")
        col.type = type_
        column_objs[name] = col
    table.columns = column_objs
    return table


def test_get_table_reflects_once(connector: RedshiftConnector) -> None:
    """get_table reflects on first call and serves the cache thereafter."""
    sentinel_engine = MagicMock(name="engine")
    with patch.object(
        RedshiftConnector, "_engine", new_callable=PropertyMock, return_value=sentinel_engine
    ), patch("target_redshift.connector.Table", return_value=_fake_table()) as mock_table:
        first = connector.get_table("public.my_table")
        second = connector.get_table("public.my_table")

    assert first is second
    assert mock_table.call_count == 1


def test_cache_key_is_case_insensitive(connector: RedshiftConnector) -> None:
    """Cache keys fold to lowercase, matching Redshift identifier folding."""
    assert connector._cache_key("Public.My_Table") == ("public", "my_table")
    assert connector._cache_key("PUBLIC.MY_TABLE") == connector._cache_key("public.my_table")


def test_fully_qualified_name_matches_plain(connector: RedshiftConnector) -> None:
    """A dialect-quoted FullyQualifiedName resolves to the same key as a plain name.

    Skipped if the installed singer-sdk does not expose a constructable
    ``FullyQualifiedName`` (the runtime caching behaviour is unaffected).
    """
    try:
        from singer_sdk.connectors.sql import FullyQualifiedName

        fqn = FullyQualifiedName(table="My_Table", schema="Public")
    except Exception:  # noqa: BLE001 - any construction issue → behaviour unaffected, skip
        pytest.skip("FullyQualifiedName not constructable in this singer-sdk version")

    assert connector._cache_key(fqn) == connector._cache_key("public.my_table")


def test_get_column_type_uses_cache(connector: RedshiftConnector) -> None:
    """_get_column_type resolves from the cache without further reflection."""
    expected_type = object()
    connector._table_cache[("public", "my_table")] = _fake_table({"id": expected_type})

    with patch("target_redshift.connector.Table") as mock_table:
        result = connector._get_column_type("public.my_table", "id")

    assert result is expected_type
    mock_table.assert_not_called()


def test_get_column_type_missing_column_raises(connector: RedshiftConnector) -> None:
    """_get_column_type raises KeyError for an unknown column."""
    connector._table_cache[("public", "my_table")] = _fake_table({"id": object()})
    with pytest.raises(KeyError):
        connector._get_column_type("public.my_table", "does_not_exist")


def test_invalidate_table_cache_forces_re_reflection(connector: RedshiftConnector) -> None:
    """After invalidation the next get_table reflects again."""
    sentinel_engine = MagicMock(name="engine")
    with patch.object(
        RedshiftConnector, "_engine", new_callable=PropertyMock, return_value=sentinel_engine
    ), patch("target_redshift.connector.Table", side_effect=[_fake_table(), _fake_table()]) as mock_table:
        connector.get_table("public.my_table")
        connector.invalidate_table_cache("public.my_table")
        connector.get_table("public.my_table")

    assert mock_table.call_count == 2


def test_table_exists_positive_cache(connector: RedshiftConnector) -> None:
    """A cached table reports as existing without deferring to the base check."""
    connector._table_cache[("public", "my_table")] = _fake_table()
    with patch.object(SQLConnector, "table_exists") as mock_super:
        assert connector.table_exists("public.my_table") is True
    mock_super.assert_not_called()


def test_table_exists_defers_to_base_when_uncached(connector: RedshiftConnector) -> None:
    """An uncached table falls back to the base implementation."""
    with patch.object(SQLConnector, "table_exists", return_value=False) as mock_super:
        assert connector.table_exists("public.absent") is False
    mock_super.assert_called_once()
